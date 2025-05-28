"""
A handy utility for verifying image generation locally.
To set up, first run a local cog server using:
   cog run -p 5000 python -m cog.server.http
Then, in a separate terminal, generate samples
   python samples.py
"""

import base64
import os
import sys
import time
from pathlib import Path
import requests


def gen(output_fn, **kwargs):
    st = time.time()
    print("Generating", output_fn)
    url = "http://localhost:5000/predictions"
    response = requests.post(url, json={"input": kwargs})
    data = response.json()
    print("Generated in: ", time.time() - st)

    os.system(f"mv output.png {output_fn}")


def test_loras():
    """
    runs generations in fp8 and bf16 on the same node! wow!
    """
    gen(
        f"cyberpunk_lady_no_lora.png",
        prompt="a photo of a woman in the style of 80s cyberpunk",
        input_image="https://replicate.delivery/pbxt/N5DXcBZiATNE0n0Wu7ghgVh5i7VoNzzfYtyGoNdbKYnZic7L/replicate-prediction-f2d25rg6gnrma0cq257vdw2n4c.png",
        aspect_ratio="1:1",
        num_outputs=1,
        output_format="png",
        disable_safety_checker=True,
        seed=123,
        )
    for strength in range(1, 7):
        gen(
            f"cyberpunk_lady_{strength}.png",
            prompt="a photo of a woman in the style of 80s cyberpunk",
            input_image="https://replicate.delivery/pbxt/N5DXcBZiATNE0n0Wu7ghgVh5i7VoNzzfYtyGoNdbKYnZic7L/replicate-prediction-f2d25rg6gnrma0cq257vdw2n4c.png",
            aspect_ratio="1:1",
            num_outputs=1,
            output_format="png",
            disable_safety_checker=True,
            seed=123,
            lora_weights="fofr/flux-80s-cyberpunk",
            lora_strength=strength,
        )


if __name__ == "__main__":
    test_loras()
