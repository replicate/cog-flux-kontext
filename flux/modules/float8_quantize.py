from loguru import logger
import torch
import torch.nn as nn
from torch.nn import init
import math
from torch.compiler import is_compiling
from torch import __version__
from torch.version import cuda


IS_TORCH_2_4 = __version__ < (2, 4, 9)
LT_TORCH_2_4 = __version__ < (2, 4)
if LT_TORCH_2_4:
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError(
            "This version of PyTorch is not supported. Please upgrade to PyTorch 2.4 with CUDA 12.4 or later."
        )
CUDA_VERSION = float(cuda) if cuda else 0
if CUDA_VERSION < 12.4:
    raise RuntimeError(
        f"This version of PyTorch is not supported. Please upgrade to PyTorch 2.4 with CUDA 12.4 or later got torch version {__version__} and CUDA version {cuda}."
    )
try:
    from cublas_ops import CublasLinear
except ImportError:
    CublasLinear = type(None)


class F8Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=torch.float16,
        float8_dtype=torch.float8_e4m3fn,
        float_weight: torch.Tensor = None,
        float_bias: torch.Tensor = None,
        num_scale_trials: int = 12,
        input_float8_dtype=torch.float8_e5m2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.float8_dtype = float8_dtype
        self.input_float8_dtype = input_float8_dtype
        self.input_scale_initialized = False
        self.weight_initialized = False
        self.max_value = torch.finfo(self.float8_dtype).max
        self.input_max_value = torch.finfo(self.input_float8_dtype).max
        factory_kwargs = {"dtype": dtype, "device": device}
        if float_weight is None:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), **factory_kwargs)
            )
        else:
            self.weight = nn.Parameter(
                float_weight, requires_grad=float_weight.requires_grad
            )
        if float_bias is None:
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(out_features, **factory_kwargs),
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(float_bias, requires_grad=float_bias.requires_grad)
        self.num_scale_trials = num_scale_trials
        self.input_amax_trials = torch.zeros(
            num_scale_trials, requires_grad=False, device=device, dtype=torch.float32
        )
        self.trial_index = 0
        self.register_buffer("scale", None)
        self.register_buffer(
            "input_scale",
            None,
        )
        self.register_buffer(
            "float8_data",
            None,
        )
        self.scale_reciprocal = self.register_buffer("scale_reciprocal", None)
        self.input_scale_reciprocal = self.register_buffer(
            "input_scale_reciprocal", None
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        sd = {k.replace(prefix, ""): v for k, v in state_dict.items()}
        if "weight" in sd:
            if (
                "float8_data" not in sd
                or sd["float8_data"] is None
                and sd["weight"].shape == (self.out_features, self.in_features)
            ):
                # Initialize as if it's an F8Linear that needs to be quantized
                self._parameters["weight"] = nn.Parameter(
                    sd["weight"], requires_grad=False
                )
                if "bias" in sd:
                    self._parameters["bias"] = nn.Parameter(
                        sd["bias"], requires_grad=False
                    )
                self.quantize_weight()

            else:
                raise RuntimeError(
                    f"Weight tensor not found or has incorrect shape in state dict: {sd.keys()}"
                )
        else:
            raise RuntimeError(
                "Weight tensor not found or has incorrect shape in state dict"
            )

    def quantize_weight(self):
        if self.weight_initialized:
            return
        amax = torch.max(torch.abs(self.weight.data)).float()
        self.scale = self.amax_to_scale(amax, self.max_value)
        self.float8_data = self.to_fp8_saturated(
            self.weight.data, self.scale, self.max_value
        ).to(self.float8_dtype)
        self.scale_reciprocal = self.scale.reciprocal()
        self.weight.data = torch.zeros(
            1, dtype=self.weight.dtype, device=self.weight.device, requires_grad=False
        )
        self.weight_initialized = True

    def set_weight_tensor(self, tensor: torch.Tensor):
        self.weight.data = tensor
        self.weight_initialized = False
        self.quantize_weight()

    def amax_to_scale(self, amax, max_val):
        return (max_val / torch.clamp(amax, min=1e-12)).clamp(max=max_val)

    def to_fp8_saturated(self, x, scale, max_val):
        return (x * scale).clamp(-max_val, max_val)

    def dynamic_quantize_input(self, x: torch.Tensor):
        amax = torch.max(torch.abs(x)).float()
        scale = self.amax_to_scale(amax, self.input_max_value)
        x_fp8 = self.to_fp8_saturated(x, scale, self.input_max_value).to(
            self.input_float8_dtype
        )
        return x_fp8, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, x_scale = self.dynamic_quantize_input(x)
        x_scale_reciprocal = x_scale.reciprocal()

        prev_dims = x.shape[:-1]
        x = x.view(-1, self.in_features)

        out = torch._scaled_mm(
            x,
            self.float8_data.T,
            scale_a=x_scale_reciprocal,
            scale_b=self.scale_reciprocal,
            bias=self.bias,
            out_dtype=self.weight.dtype,
            use_fast_accum=True,
        )

        out = out.view(*prev_dims, self.out_features)
        return out


