import os
from cog import BasePredictor, Path, Input
from flux_dist_runner import FluxDistRunner
from flux.util import ASPECT_RATIOS

import time

class FluxDistPredictor(BasePredictor):
    """
    Flux.1 Kontext Distributed Predictor - Uses MPDistRunner for distributed inference
    """

    def setup(self) -> None:
        self.runner = FluxDistRunner()
        self.runner.start()

    def predict(
        self,
        prompt: str = Input(
            description="Text description of what you want to generate, or the instruction on how to edit the given image.",
        ),
        input_image: Path = Input(
            description="Image to use as reference. Must be jpeg, png, gif, or webp.",
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio of the generated image. Use 'match_input_image' to match the aspect ratio of the input image.",
            choices=list(ASPECT_RATIOS.keys()),
            default="match_input_image",
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps", default=30, ge=4, le=50
        ),
        guidance: float = Input(
            description="Guidance scale for generation", default=2.5, ge=0.0, le=10.0
        ),
        seed: int = Input(
            description="Random seed for reproducible generation. Leave blank for random.",
            default=None,
        ),
        output_format: str = Input(
            description="Output image format",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80,
            ge=0,
            le=100,
        ),
        disable_safety_checker: bool = Input(
            description="Disable NSFW safety checker", default=False
        ),
    ) -> Path:
        """
        Generate an image based on the text prompt and conditioning image using distributed FLUX.1 Kontext
        """
        t0 = time.time()

        result = self.runner(kwargs={
            "prompt": prompt,
            "input_image": input_image,
            "aspect_ratio": aspect_ratio,
            "num_inference_steps": num_inference_steps,
            "guidance": guidance,
            "seed": seed,
            "output_format": output_format,
            "output_quality": output_quality,
            "disable_safety_checker": disable_safety_checker,
        })
        t1 = time.time()
        print(f"Prediction time: {t1 - t0:.2f} seconds")
        return result

    def __del__(self):
        if hasattr(self, 'runner') and self.runner:
            try:
                self.runner.terminate()
            except Exception as e:
                print(f"Error during cleanup: {e}")
            finally:
                self.runner = None 