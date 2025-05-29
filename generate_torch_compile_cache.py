
import os
import torch
import time

from predict import load_kontext_model, TORCH_COMPILE_CACHE, ASPECT_RATIOS, FP8_QUANTIZATION, quantize_filter_fn
from util import warm_up_model
torch._dynamo.config.recompile_limit = 50

from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
from torchao.quantization.granularity import PerTensor, PerRow

def generate_torch_compile_cache():
    device = torch.device("cuda")
    model = load_kontext_model(device)
    if FP8_QUANTIZATION:
        quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()), filter_fn=quantize_filter_fn)
    model = torch.compile(model, dynamic=False)
    
    if os.path.exists(TORCH_COMPILE_CACHE):
        print(f"Removing existing torch compile cache at {TORCH_COMPILE_CACHE}")
        os.remove(TORCH_COMPILE_CACHE)

    for (h,w) in ASPECT_RATIOS.values():
        if (h,w) == (None, None):
            continue
        t0 = time.time()
        warm_up_model(h, w, model)
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"Warm up time: {t1 - t0} seconds")

    artifacts = torch.compiler.save_cache_artifacts()
    assert artifacts is not None
    artifact_bytes, cache_info = artifacts
    with open(TORCH_COMPILE_CACHE, "wb") as f:
        f.write(artifact_bytes)
    print(f"Saved torch compile cache to {TORCH_COMPILE_CACHE}")
    print(f"Cache info: {cache_info}")

if __name__ == "__main__":
    generate_torch_compile_cache()