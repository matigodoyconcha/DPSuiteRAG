import os

from dotenv import load_dotenv

load_dotenv()


def resolve_device() -> str:
    """Return the best available compute device, respecting CUDA_VISIBLE_DEVICES from .env.

    Priority: cuda > mps > cpu.
    Set CUDA_VISIBLE_DEVICES in .env to control which GPU(s) are used.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
