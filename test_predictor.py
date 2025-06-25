#!/usr/bin/env python3
"""
Test script for FluxDevKontextPredictor
This script imports the predictor, sets it up, and calls predict with all aspect ratios.
"""

# import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from flux.util import ASPECT_RATIOS
from predict import FluxDevKontextPredictor
from cog import Path
import time

import torch.distributed as dist
import torch

def main():
    predictor = FluxDevKontextPredictor()
    
    t0 = time.time()
    predictor.setup()
    t1 = time.time()
    print(f"Setup time: {t1 - t0} seconds")
    
    input_image_path = "aja.jpg"  # Replace with your actual image path

    # Test each aspect ratio
    # for aspect_ratio in ASPECT_RATIOS.keys():
    for i in range(3):
        print(f"\n{'='*50}")
        # print(f"Testing aspect ratio: {aspect_ratio}")
        print(f"{'='*50}")
        
        result = predictor.predict(
            prompt="change her haircut to be a pixie haircut",
            input_image=Path(input_image_path),
            aspect_ratio="match_input_image",
            num_inference_steps=30,
            guidance=3.5,
            seed=42,  # Using fixed seed for consistency
            output_format="png",
            output_quality=80,
            disable_safety_checker=False,
        )
        
        # rename output file to include aspect ratio
        # output_file = result.name
        # output_file = output_file.replace(".png", f"_{aspect_ratio}.png")
        # result.rename(output_file)

if __name__ == "__main__":
    main() 