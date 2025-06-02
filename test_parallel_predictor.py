#!/usr/bin/env python3
"""
Test script for FluxDevKontextPredictor
This script imports the predictor, sets it up, and calls predict with all aspect ratios.
"""

# import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from flux.util import ASPECT_RATIOS
from dist_predictor import FluxDistPredictor
from cog import Path
import time

import torch.distributed as dist
import torch
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
# torch.random.manual_seed(0)


def main():
    predictor = FluxDistPredictor()
    
    t0 = time.time()
    predictor.setup()
    t1 = time.time()
    print(f"Setup time: {t1 - t0} seconds")
    
    input_image_path = "girl.jpeg"  # Replace with your actual image path

    # Test each aspect ratio
    for aspect_ratio in ASPECT_RATIOS.keys():
    # for aspect_ratio in ["1:1", "1:1", "1:1"]:
        print(f"\n{'='*50}")
        print(f"Testing aspect ratio: {aspect_ratio}")
        print(f"{'='*50}")
        
        result = predictor.predict(
            prompt="change her haircut to be a pixie haircut",
            input_image=Path(input_image_path),
            aspect_ratio=aspect_ratio,
            num_inference_steps=30,
            guidance=3.5,
            seed=42,  # Using fixed seed for consistency
            output_format="png",
            output_quality=80,
            disable_safety_checker=True,
        )
        
        # rename output file to include aspect ratio
        output_file = result.name
        output_file = output_file.replace(".png", f"_{aspect_ratio}.png")
        result.rename(output_file)

if __name__ == "__main__":
    main() 