from PIL import Image
import torch
import numpy as np
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import (
    CLIPImageProcessor,
    AutoModelForImageClassification,
    ViTImageProcessor,
)
from cog import Path

from weights import download_weights_pget
from util import print_timing


SAFETY_CACHE = Path("./safety-cache")
FEATURE_EXTRACTOR = Path("./feature-extractor")
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"
MAX_IMAGE_SIZE = 1440

FALCON_MODEL_NAME = "Falconsai/nsfw_image_detection"
FALCON_MODEL_CACHE = Path("./falcon-cache")
FALCON_MODEL_URL = (
    "https://weights.replicate.delivery/default/falconai/nsfw-image-detection.tar"
)


class SafetyChecker:
    def __init__(self):
        with print_timing("Loading SDXL safety checker"):
            if not SAFETY_CACHE.exists():
                download_weights_pget(SAFETY_URL, SAFETY_CACHE)
            self.sdxl_safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                SAFETY_CACHE, torch_dtype=torch.float16
            ).to("cuda")  # type: ignore
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                FEATURE_EXTRACTOR
            )

        with print_timing("Loading Falcon safety checker"):
            if not FALCON_MODEL_CACHE.exists():
                download_weights_pget(FALCON_MODEL_URL, FALCON_MODEL_CACHE)
            self.falcon_model = AutoModelForImageClassification.from_pretrained(
                FALCON_MODEL_NAME,
                cache_dir=FALCON_MODEL_CACHE,
            )
            self.falcon_processor = ViTImageProcessor.from_pretrained(FALCON_MODEL_NAME)

    def filter_images(self, images: list[Image.Image]) -> list[Image.Image]:
        has_nsfw_content = [False] * len(images)

        has_nsfw_content = self.run_sdxl_safety_checker(images)

        filtered = []
        for i, (img, is_nsfw) in enumerate(zip(images, has_nsfw_content)):
            if is_nsfw:
                try:
                    falcon_is_safe = self.run_falcon_safety_checker(img)
                except Exception as e:
                    print(f"Error running safety checker: {e}")
                    falcon_is_safe = False
                if not falcon_is_safe:
                    print(f"NSFW content detected in image {i}")
                    continue

            filtered.append(img)

        if not filtered:
            raise Exception(
                "All generated images contained NSFW content. Try running it again with a different prompt."
            )

        print(f"Total safe images: {len(filtered)} out of {len(images)}")

        return filtered

    def run_sdxl_safety_checker(self, images: list[Image.Image]) -> list[bool]:
        np_images = [np.array(img) for img in images]

        safety_checker_input = self.feature_extractor(images, return_tensors="pt").to(  # type: ignore
            "cuda"
        )
        _, has_nsfw_concept = self.sdxl_safety_checker(
            images=np_images,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return has_nsfw_concept

    def run_falcon_safety_checker(self, image):
        with torch.no_grad():
            inputs = self.falcon_processor(images=image, return_tensors="pt")  # type: ignore
            outputs = self.falcon_model(**inputs)
            logits = outputs.logits
            predicted_label = logits.argmax(-1).item()
            result = self.falcon_model.config.id2label[predicted_label]

        return result == "normal"
