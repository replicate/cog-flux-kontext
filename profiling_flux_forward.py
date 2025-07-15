import torch

# Import the model definition
from flux.model import Flux, FluxParams
from safetensors.torch import load_file as load_sft
from flux.util import optionally_expand_state_dict


@torch.no_grad()
def benchmark_flux_forward():
    """Run a single forward pass of `Flux` under `torch.profiler` after warm-up."""

    device = torch.device("cuda")
    dtype = torch.bfloat16

    params = FluxParams(
        in_channels=64,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=True,
    )

    model = Flux(params).to(device).to(dtype)
    sd = load_sft("models/kontext/kontext-dev.sft", device=str(device))
    sd = optionally_expand_state_dict(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)

    # iterate over all the modules in the model
    # for name, module in model.named_modules():
    #     if isinstance(module, F8Linear):
    #         print(f"{name} has weight: {module.weight.shape}")

    model = torch.compile(model, dynamic=True)

    img = torch.randn(1, 8220, 64, device=device, dtype=dtype)
    img_ids = torch.randint(0, 1024, (1, 8220, 3), device=device, dtype=torch.float32)

    txt = torch.randn(1, 512, 4096, device=device, dtype=dtype)
    txt_ids = torch.randint(0, 1024, (1, 512, 3), device=device, dtype=torch.float32)

    # timesteps / guidance are floats in the original prints – stay within BF16 safe range
    timesteps = torch.rand(1, device=device, dtype=dtype) * 1000.0
    y = torch.randn(1, 768, device=device, dtype=dtype)

    # Even though guidance is unused (guidance_embed=False), we create it to
    # match the reference shapes so that printing inside `forward` succeeds.
    guidance = torch.rand(1, device=device, dtype=dtype) * 1000.0

    with torch.inference_mode():
        for _ in range(10):
            model(img, img_ids, txt, txt_ids, timesteps, y, guidance)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:
            model(img, img_ids, txt, txt_ids, timesteps, y, guidance)

        # Print a succinct table sorted by self CUDA time
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total", row_limit=20
            )
        )


if __name__ == "__main__":
    benchmark_flux_forward()
