import os
from lora import load_lora, unload_loras
import torch
from PIL import Image
from cog import BasePredictor, Path, Input
import time

from flux.sampling import denoise, get_schedule, prepare_kontext, unpack
from flux.util import (
    configs,
    load_clip,
    load_t5,
)
from flux.model import Flux
from flux.modules.autoencoder import AutoEncoder
from safetensors.torch import load_file as load_sft
from safety_checker import SafetyChecker
from util import print_timing
from weights import WeightsDownloadCache, download_weights_pget

# Environment setup
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch-inductor-cache-kontext"

# Kontext model configuration
KONTEXT_WEIGHTS_URL = "https://weights.replicate.delivery/default/black-forest-labs/kontext/release-candidate/kontext-dev.sft"
KONTEXT_WEIGHTS_PATH = "./models/kontext/preliminary-dev-kontext.sft"

# Model weights URLs
AE_WEIGHTS_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/safetensors/ae.safetensors"
AE_WEIGHTS_PATH = "./models/flux-dev/ae.safetensors"

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
    "match_input_image": (None, None),
}


class FluxDevKontextPredictor(BasePredictor):
    """
    Flux.1 Kontext Predictor - Image-to-image transformation model using FLUX.1-dev architecture
    """

    def setup(self) -> None:
        """Load model weights and initialize the pipeline"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Download all weights if needed
        download_model_weights()

        # Initialize models
        self.t5 = load_t5(self.device, max_length=512)
        self.clip = load_clip(self.device)
        self.model = load_kontext_model(device=self.device)
        self.ae = load_ae_local(device=self.device)

        self.cur_lora = None
        self.cur_strength = -10
        # Compile models for faster execution
        # print("Compiling models with torch.compile...")
        self.model = torch.compile(self.model, dynamic=True)
        start_time = time.time()
        self.predict(
            prompt="Make the hair blue",
            input_image=Path("input_image.png"),
            aspect_ratio="1:1",
            num_inference_steps=30,
            guidance=2.5,
            seed=42,
            output_format="png",
            output_quality=100,
            disable_safety_checker=True,
            lora_weights=None,
            lora_strength=1.0,
        )
        print(f"Compiled in {time.time() - start_time} seconds")
        # self.ae.decode = torch.compile(self.ae.decode, mode="max-autotune")

        # Initialize safety checker
        self.safety_checker = SafetyChecker()
        self.cache = WeightsDownloadCache()

        print("FluxDevKontextPredictor setup complete")

    def size_from_aspect_megapixels(
        self, aspect_ratio: str, megapixels: str = "1"
    ) -> tuple[int | None, int | None]:
        """Convert aspect ratio and megapixels to width and height"""
        width, height = ASPECT_RATIOS[aspect_ratio]
        if width is None or height is None:
            # For match_input_image, return None values
            return (None, None)
        if megapixels == "0.25":
            width, height = width // 2, height // 2
        return (width, height)

    def handle_lora(self, lora_weights: Path, lora_strength: float):
        if not lora_weights:
            if self.cur_lora is not None:
                unload_loras(self.model)
                self.cur_lora = None
                self.cur_strength = -10
            return
        
        lora_weights = str(lora_weights)
        lora_weights = self.cache.ensure(lora_weights)
        if lora_weights == self.cur_lora and lora_strength == self.cur_strength:
            print("Lora already loaded")
            return
        
        unload_loras(self.model)
        if lora_weights is not None:
            load_lora(self.model, lora_weights, lora_strength, store_clones=True)
            self.cur_lora = lora_weights
            self.cur_strength = lora_strength
        return

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
            default="1:1",
        ),
        megapixels: str = Input(
            description="Approximate number of megapixels for generated image",
            choices=["1", "0.25"],
            default="1",
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
        lora_weights: str = Input(
            description="Path to the lora weights", default=None
        ),
        lora_strength: float = Input(
            description="Strength of the lora", default=1.0
        ),
    ) -> Path:
        """
        Generate an image based on the text prompt and conditioning image using FLUX.1 Kontext
        """

        # handle loras 
        self.handle_lora(lora_weights, lora_strength)

        with torch.inference_mode(), print_timing("generate image"):
            seed = prepare_seed(seed)

            # Prepare target dimensions from aspect ratio and megapixels
            target_width, target_height = self.size_from_aspect_megapixels(
                aspect_ratio, megapixels
            )
            print(f"Target dimensions: {target_width}x{target_height}")

            # Prepare input for kontext sampling
            with print_timing("prepare input"):
                inp, final_height, final_width = prepare_kontext(
                    t5=self.t5,
                    clip=self.clip,
                    prompt=prompt,
                    ae=self.ae,
                    img_cond_path=str(input_image),
                    target_width=target_width,
                    target_height=target_height,
                    bs=1,
                    seed=seed,
                    device=self.device,
                )

            # Remove the original conditioning image from memory to save space
            inp.pop("img_cond_orig", None)

            # Get sampling schedule
            timesteps = get_schedule(
                num_inference_steps,
                inp["img"].shape[1],
                shift=True,  # flux-dev uses shift=True
            )

            # Generate image
            with print_timing("denoise"):
                x = denoise(self.model, **inp, timesteps=timesteps, guidance=guidance)

            # Decode latents to pixel space
            with print_timing("decode"):
                x = unpack(x.float(), final_height, final_width)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    x = self.ae.decode(x)

            with print_timing("convert to image"):
                x = x.clamp(-1, 1)
                x = (x + 1) / 2
                x = (x.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
                image = Image.fromarray(x[0])

            # Apply safety checking
            if not disable_safety_checker:
                with print_timing("Running safety checker"):
                    images = self.safety_checker.filter_images([image])
                    if not images:
                        raise Exception(
                            "Generated image contained NSFW content. Try running it again with a different prompt."
                        )
                    image = images[0]

            # Save image
            output_path = f"output.{output_format}"
            if output_format == "png":
                image.save(output_path)
            elif output_format == "webp":
                image.save(output_path, format="WEBP", quality=output_quality, optimize=True)
            else:  # jpg
                image.save(output_path, format="JPEG", quality=output_quality, optimize=True)

            # Return the output path
            return Path(output_path)


def download_model_weights():
    """Download all required model weights if they don't exist"""
    # Download kontext weights
    if not os.path.exists(KONTEXT_WEIGHTS_PATH):
        print("Kontext weights not found, downloading...")
        download_weights_pget(KONTEXT_WEIGHTS_URL, Path(KONTEXT_WEIGHTS_PATH))
        print("Kontext weights downloaded successfully")
    else:
        print("Kontext weights already exist")

    # Download autoencoder weights
    if not os.path.exists(AE_WEIGHTS_PATH):
        print("Autoencoder weights not found, downloading...")
        download_weights_pget(AE_WEIGHTS_URL, Path(AE_WEIGHTS_PATH))
        print("Autoencoder weights downloaded successfully")
    else:
        print("Autoencoder weights already exist")


def load_kontext_model(device: str | torch.device = "cuda"):
    """Load the kontext model with complete transformer weights"""
    # Use flux-dev config as base for kontext model
    config = configs["flux-dev"]

    print("Loading kontext model...")
    with torch.device("meta"):
        model = Flux(config.params).to(torch.bfloat16)

    # Load kontext weights (complete transformer)
    print(f"Loading kontext weights from {KONTEXT_WEIGHTS_PATH}")
    sd = load_sft(KONTEXT_WEIGHTS_PATH, device=str(device))
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)

    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    return model


def load_ae_local(device: str | torch.device = "cuda"):
    """Load autoencoder from local weights"""
    config = configs["flux-dev"]

    print("Loading autoencoder...")
    with torch.device("meta"):
        ae = AutoEncoder(config.ae_params)

    print(f"Loading autoencoder weights from {AE_WEIGHTS_PATH}")
    sd = load_sft(AE_WEIGHTS_PATH, device=str(device))
    missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)

    if missing:
        print(f"AE Missing keys: {missing}")
    if unexpected:
        print(f"AE Unexpected keys: {unexpected}")

    return ae

def prepare_seed(seed: int) -> int:
    if not seed:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")
    return seed
