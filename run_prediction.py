
import time
from pathlib import Path
from predict import FluxDevKontextPredictor
import os

def main():
    # Instantiate the predictor
    predictor = FluxDevKontextPredictor()

    # Call setup
    print("Setting up the predictor...")
    setup_start_time = time.time()
    predictor.setup()
    setup_end_time = time.time()
    print(f"Setup completed in {setup_end_time - setup_start_time:.2f} seconds.")

    lora_weights = [
        "https://replicate.delivery/xezq/89qeqISCarVyDaLJRVCzmd12Woh7k4jdTmjCg7AHN19o56fUA/flux-lora.tar",
        "https://huggingface.co/fal/Plushie-Kontext-Dev-LoRA/resolve/main/plushie-kontext-dev-lora.safetensors",
        "https://huggingface.co/gokaygokay/Bronze-Sculpture-Kontext-Dev-LoRA/resolve/main/bronze.safetensors"
    ]

    prompts = [
        "render this image like a ps1 game (no UI)",
        "Convert to plushie style",
        "Convert this image into bronze version"
    ]

    lora_keys = [
        'ps1',
        'plushie',
        'bronze'
    ]

    # Explicitly define all arguments for the predict method

    for i in range(len(prompts)):
        predict_args = {
            "prompt": prompts[i],
            "input_image": Path("IMG_3924.png"),
            "aspect_ratio": "1:1",
            "megapixels": "1",
            "num_inference_steps": 30,
            "guidance": 2.5,
            "seed": 123,
            "output_format": "png",
            "output_quality": 80,
            "disable_safety_checker": True,
            "lora_weights": lora_weights[i],
            "lora_strength": 1.0,
        }
        
        output_path = predictor.predict(**predict_args)
        renamed = f"output_{lora_keys[i]}.png"
        os.rename(output_path, renamed)
        print(f"Prediction saved to: {renamed}")

if __name__ == "__main__":
    main() 