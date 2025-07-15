
import time
from pathlib import Path
from predict import FluxDevKontextPredictor

def main():
    # Instantiate the predictor
    predictor = FluxDevKontextPredictor()

    # Call setup
    print("Setting up the predictor...")
    setup_start_time = time.time()
    predictor.setup()
    setup_end_time = time.time()
    print(f"Setup completed in {setup_end_time - setup_start_time:.2f} seconds.")

    # Explicitly define all arguments for the predict method
    predict_args = {
        "prompt": "a photo of a woman in the style of 80s cyberpunk",
        "input_image": Path("input_image.png"),
        "aspect_ratio": "1:1",
        "megapixels": "1",
        "num_inference_steps": 30,
        "guidance": 2.5,
        "seed": 123,
        "output_format": "png",
        "output_quality": 80,
        "disable_safety_checker": True,
        # "lora_weights": "https://replicate.delivery/xezq/89qeqISCarVyDaLJRVCzmd12Woh7k4jdTmjCg7AHN19o56fUA/flux-lora.tar",
        "lora_weights": None, 
        "lora_strength": 1.0,
    }

    # Call predict
    print("Running prediction...")
    predict_start_time = time.time()
    output_path = predictor.predict(**predict_args)
    predict_end_time = time.time()
    print(f"Prediction completed in {predict_end_time - predict_start_time:.2f} seconds.")

    print(f"Prediction saved to: {output_path}")

if __name__ == "__main__":
    main() 