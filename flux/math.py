import torch
from einops import rearrange
from torch import Tensor

from sageattention.core import sageattn_qk_int8_pv_fp8_cuda_sm90

# from torch.distributed.tensor.experimental._attention import _templated_ring_attention
# import torch.distributed as dist
# from para_attn.para_attn_interface import ring_attn_func

# import torch.distributed._functional_collectives as ft_c
import torch.distributed.distributed_c10d as c10d
from para_attn.para_attn_interface import _sdpa_input_all_to_all, _sdpa_output_all_to_all

def _ulysses_attn_func(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    mesh=None,
):

    mesh = c10d._get_default_group()

    query = _sdpa_input_all_to_all(query, mesh)
    key = _sdpa_input_all_to_all(key, mesh)
    value = _sdpa_input_all_to_all(value, mesh)

    out = sageattn_qk_int8_pv_fp8_cuda_sm90(query, key, value, is_causal=False)

    out = _sdpa_output_all_to_all(out, mesh)
    return out


def ulysses_attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    x = _ulysses_attn_func(q, k, v, is_causal=False)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


# def ring_attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
#     q, k = apply_rope(q, k, pe)
#     x = ring_attn_func(q, k, v, is_causal=False)
#     x = rearrange(x, "B H L D -> B L (H D)")
#     return x


def quantized_attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    x = sageattn_qk_int8_pv_fp8_cuda_sm90(q, k, v, is_causal=False)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x

from torch.nn.attention import sdpa_kernel, SDPBackend

def cudnn_attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION): 
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
