from contextlib import contextmanager
import time
import torch

@contextmanager
def print_timing(operation_name: str):
    """Context manager to time and print the execution time of a block of code.

    Args:
        operation_name: A descriptive name for the operation being timed
    """
    start_time = time.time()
    try:
        print(f"Starting {operation_name}")
        yield
    finally:
        elapsed_time = time.time() - start_time
        print(f"{operation_name} took {elapsed_time:.2f} seconds")


def get_sequence_length(w, h):
    """
    calculate the models internal sequence length, given a width and height input
    """

    # if any of these fail the resolution is invalid
    assert w % 16 == 0
    assert h % 16 == 0
    assert (h * w) % 64 == 0

    # autoencoder downsamples height and width by 8, and creates 16 channels
    latent_height = h // 8
    latent_width = w // 8
    total_latent_size = latent_height * latent_width * 16
    assert total_latent_size % 64 == 0
    
    # and then does latent_image.reshape(-1, 64) to create an embedding dim of 64
    image_seq_len = total_latent_size // 64
    
    # img (the tensor that is being denoised) and img_cond (the tensor that provides conditioning)
    # are preprocessed the same way and then concatenated
    image_seq_len *= 2
    txt_seq_len = 512
    return image_seq_len, txt_seq_len

def warm_up_model(h, w, model, device):
    in_channels = model.params.in_channels
    assert in_channels == 64
    context_in_dim = model.params.context_in_dim
    batch_size = 1
    image_id_dim = 3
    image_seq_len, txt_seq_len = get_sequence_length(w, h)

    img_input = torch.rand(batch_size, image_seq_len, in_channels, device=device, dtype=torch.bfloat16)
    img_input_ids = torch.rand(batch_size, image_seq_len, image_id_dim, device=device, dtype=torch.float32) * 73.0
    txt = torch.rand(batch_size, txt_seq_len, context_in_dim, device=device, dtype=torch.bfloat16)
    txt_ids = torch.zeros(batch_size, txt_seq_len, image_id_dim, device=device, dtype=torch.float32)
    vec = torch.rand(1, 768, device=device, dtype=torch.bfloat16)
    t_vec = torch.tensor([0.82421875], device=device, dtype=torch.bfloat16)
    guidance_vec = torch.tensor([3.5], device=device, dtype=torch.bfloat16)

    with torch.no_grad():

        _ = model(
            img=img_input,
            img_ids=img_input_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=t_vec,
            y=vec,
            guidance=guidance_vec,
        )