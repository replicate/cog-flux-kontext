#!/usr/bin/env python3
"""
Test script for FluxDevKontextPredictor
This script imports the predictor, sets it up, and calls predict with all aspect ratios.
"""

# import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from predict import FluxDevKontextPredictor
import time


def main():
    predictor = FluxDevKontextPredictor()

    t0 = time.time()
    predictor.setup()
    t1 = time.time()
    print(f"Setup time: {t1 - t0} seconds")

    model_runs = [
        {
            "input_image": "car.jpg",
            "go_fast": True,
            "prompt": "Change the car color to red, turn the headlights on",
        },
        {
            "input_image": "car.jpg",
            "go_fast": False,
            "prompt": "Change the car color to red, turn the headlights on",
        },
        {
            "input_image": "guy.jpeg",
            "go_fast": True,
            "prompt": "make him into an oil painting, exactly preserving his likeness and facial features",
        },
        {
            "input_image": "guy.jpeg",
            "go_fast": False,
            "prompt": "make him into an oil painting, exactly preserving his likeness and facial features",
        },
        {
            "input_image": "lady.png",
            "go_fast": True,
            "prompt": "change the text on her sweater to say 'sally sells sea shells by the sea shore'",
        },
        {
            "input_image": "lady.png",
            "go_fast": False,
            "prompt": "change the text on her sweater to say 'sally sells sea shells by the sea shore'",
        },
    ]

    for model_run in model_runs:
        result = predictor.predict(
            prompt=model_run["prompt"],
            input_image="input_images/" + model_run["input_image"],
            aspect_ratio="match_input_image",
            num_inference_steps=28,
            guidance=3.5,
            seed=42,  # Using fixed seed for consistency
            output_format="png",
            output_quality=80,
            disable_safety_checker=False,
            go_fast=model_run["go_fast"],
        )

        input_image_name = model_run["input_image"].split(".")[0]
        go_fast = model_run["go_fast"]
        output_file = f"output_images/{input_image_name}_{go_fast}.webp"
        result.rename(output_file)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
