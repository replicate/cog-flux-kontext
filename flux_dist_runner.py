from para_attn.distributed.mp_runner import MPDistRunner
import time
import torch
import torch.distributed as dist
import os
from PIL import Image
from cog import Path

from flux.sampling import denoise, get_schedule, prepare_kontext, unpack
from flux.util import (
    configs,
    load_clip,
    load_t5
)
from flux.model import Flux
from flux.modules.autoencoder import AutoEncoder
from safetensors.torch import load_file as load_sft
from safety_checker import SafetyChecker
from util import print_timing, get_sequence_length
from weights import download_weights
from flux.util import ASPECT_RATIOS

# Model weights URLs and paths
KONTEXT_WEIGHTS_URL = "https://weights.replicate.delivery/default/black-forest-labs/kontext/pre-release/preliminary-dev-kontext.sft"
KONTEXT_WEIGHTS_PATH = "/models/kontext/preliminary-dev-kontext.sft"
AE_WEIGHTS_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/safetensors/ae.safetensors"
AE_WEIGHTS_PATH = "/models/flux-dev/ae.safetensors"

class FluxDistRunner(MPDistRunner):

    @property
    def world_size(self):
        return torch.cuda.device_count()

    
    def init_processor(self):
        """Initialize the Flux model on each device/rank"""
        self.device = torch.device(f"cuda:{dist.get_rank()}")
        self.rank = dist.get_rank()
        
        # Download all weights if needed (only rank 0)
        if self.rank == 0:
            self._download_model_weights()
        dist.barrier()

        # Initialize models on each rank
        self.t5 = load_t5(self.device, max_length=512)
        self.clip = load_clip(self.device)
        self.model = self._load_kontext_model(device=self.device)
        self.ae = self._load_ae_local(device=self.device)

        # Compile the model for better performance
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.model = torch.compile(self.model, dynamic=False)
        # TODO load torch compile cache and warmup

        # Initialize safety checker (only needed on rank 0 for final output)
        if self.rank == 0:
            self.safety_checker = SafetyChecker()
        else:
            self.safety_checker = None

        print(f"FluxDistRunner initialized on rank {self.rank}")

    
    @torch.inference_mode()
    def process_task(self,
                     prompt,
                     input_image,
                     aspect_ratio,
                     num_inference_steps,
                     guidance,
                     seed,
                     output_format,
                     output_quality,
                     disable_safety_checker):
        """Process a single inference task"""

        if aspect_ratio == "match_input_image":
            target_height, target_width = None, None
        else:
            target_height, target_width = ASPECT_RATIOS[aspect_ratio]

        # Prepare input for kontext sampling (only rank 0)
        if self.rank == 0:
            with print_timing("prepare_kontext"):
                inp, final_height, final_width = prepare_kontext(
                    t5=self.t5,
                    clip=self.clip,
                    prompt=prompt,
                    ae=self.ae,
                    img_cond_path=str(input_image),
                    target_height_width=(target_height, target_width),
                    bs=1,
                    seed=seed,
                    device=self.device,
                )
                inp.pop("img_cond_orig", None)
                img = inp["img"]
                img_ids = inp["img_ids"]
                txt = inp["txt"]
                txt_ids = inp["txt_ids"]
                vec = inp["vec"]
                img_cond_seq = inp["img_cond_seq"]
                img_cond_seq_ids = inp["img_cond_seq_ids"]

                image_seq_len, txt_seq_len = get_sequence_length(final_width, final_height)
                assert image_seq_len // 2 == img.shape[1], f"image sequence length calculation wrong for {final_width}x{final_height}"
                assert txt_seq_len == txt.shape[1], f"text sequence length calculation wrong for {final_width}x{final_height}"
        else:
            
            # figure out what the height and width should be
            if target_height is None:
                img_cond = Image.open(str(input_image)).convert("RGB")
                width, height = img_cond.size
            else:
                width, height = target_height, target_width
            
            aspect_ratio = width / height
            _, width, height = min((abs(aspect_ratio - w / h), w, h) for h, w in ASPECT_RATIOS.values())
            final_width = 16 * (width // 16)
            final_height = 16 * (height // 16)
            image_seq_len, txt_seq_len = get_sequence_length(final_width, final_height)
            config = configs["flux-dev"]
            id_dim = 3
            context_in_dim = config.params.context_in_dim
            single_image_seq_len = image_seq_len // 2

            img = torch.zeros(1, single_image_seq_len, config.params.in_channels, device=self.device, dtype=torch.bfloat16)
            img_ids = torch.zeros(1, single_image_seq_len, id_dim, device=self.device, dtype=torch.float32)
            txt = torch.zeros(1, txt_seq_len, context_in_dim, device=self.device, dtype=torch.bfloat16)
            txt_ids = torch.zeros(1, txt_seq_len, id_dim, device=self.device, dtype=torch.float32)
            vec = torch.zeros(1, config.params.vec_in_dim, device=self.device, dtype=torch.bfloat16)
            img_cond_seq = torch.zeros(1, single_image_seq_len, config.params.in_channels, device=self.device, dtype=torch.bfloat16)
            img_cond_seq_ids = torch.zeros(1, single_image_seq_len, id_dim, device=self.device, dtype=torch.float32)
            
        # Broadcast tensors from rank 0 to all other ranks
        
        dist.broadcast(img, src=0)
        dist.broadcast(img_ids, src=0)
        dist.broadcast(txt, src=0)
        dist.broadcast(txt_ids, src=0)
        dist.broadcast(vec, src=0)
        dist.broadcast(img_cond_seq, src=0)
        dist.broadcast(img_cond_seq_ids, src=0)

        # Broadcast final dimensions
        dims_tensor = torch.tensor([final_height, final_width], device=self.device, dtype=torch.int32)
        dist.broadcast(dims_tensor, src=0)
        final_height, final_width = dims_tensor[0].item(), dims_tensor[1].item()

        inp = {
            "img": img,
            "img_ids": img_ids,
            "txt": txt,
            "txt_ids": txt_ids,
            "vec": vec,
            "img_cond_seq": img_cond_seq,
            "img_cond_seq_ids": img_cond_seq_ids,
        }

        # Get sampling schedule
        timesteps = get_schedule(
            num_inference_steps,
            inp["img"].shape[1],
            shift=True,  # flux-dev uses shift=True
        )

        # Generate image (distributed across all ranks)
        with print_timing("denoise"):
            x = denoise(self.model, **inp, timesteps=timesteps, guidance=guidance)

        # Decode latents to pixel space (only on rank 0)
        if self.rank == 0:
            with print_timing("decode-postproc"):
                x = unpack(x.float(), final_height, final_width)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    x = self.ae.decode(x)

                # Convert to image
                x = x.clamp(-1, 1)
                x = (x + 1) / 2
                x = (x.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
                image = Image.fromarray(x[0])

                # Apply safety checking
                if not disable_safety_checker and self.safety_checker:
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
                    image.save(
                        output_path, format="WEBP", quality=output_quality, optimize=True
                    )
                else:  # jpg
                    image.save(
                        output_path, format="JPEG", quality=output_quality, optimize=True
                    )

                return Path(output_path)
        else:
            # Other ranks don't need to return anything
            return None

    def _download_model_weights(self):
        """Download all required model weights if they don't exist"""
        # Download kontext weights
        if not os.path.exists(KONTEXT_WEIGHTS_PATH):
            print("Kontext weights not found, downloading...")
            download_weights(KONTEXT_WEIGHTS_URL, Path(KONTEXT_WEIGHTS_PATH))
            print("Kontext weights downloaded successfully")
        else:
            print("Kontext weights already exist")

        # Download autoencoder weights
        if not os.path.exists(AE_WEIGHTS_PATH):
            print("Autoencoder weights not found, downloading...")
            download_weights(AE_WEIGHTS_URL, Path(AE_WEIGHTS_PATH))
            print("Autoencoder weights downloaded successfully")
        else:
            print("Autoencoder weights already exist")

    def _load_kontext_model(self, device: str | torch.device = "cuda"):
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

    def _load_ae_local(self, device: str | torch.device = "cuda"):
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



