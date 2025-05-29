
import os
import torch

from predict import load_kontext_model, TORCH_COMPILE_CACHE, ASPECT_RATIOS
from util import warm_up_model
torch._dynamo.config.recompile_limit = 50

from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
from torchao.quantization.granularity import PerTensor, PerRow

def generate_torch_compile_cache():
    
    device = torch.device("cuda")
    model = load_kontext_model(device)
    model = torch.compile(model, dynamic=False)
    # quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))

    if os.path.exists(TORCH_COMPILE_CACHE):
        print(f"Removing existing torch compile cache at {TORCH_COMPILE_CACHE}")
        os.remove(TORCH_COMPILE_CACHE)

    for (h,w) in ASPECT_RATIOS.values():
        if (h,w) == (None, None):
            continue
        warm_up_model(h, w, model)

    artifacts = torch.compiler.save_cache_artifacts()
    assert artifacts is not None
    artifact_bytes, cache_info = artifacts
    with open(TORCH_COMPILE_CACHE, "wb") as f:
        f.write(artifact_bytes)
    print(f"Saved torch compile cache to {TORCH_COMPILE_CACHE}")
    print(f"Cache info: {cache_info}")

if __name__ == "__main__":
    generate_torch_compile_cache()