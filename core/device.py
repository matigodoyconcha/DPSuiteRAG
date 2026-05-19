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


def model_device_kwargs() -> dict:
    """Return from_pretrained() kwargs for optimal device placement.

    CUDA (single or multi-GPU): uses device_map="auto" so transformers+accelerate
    spreads the model across all GPUs listed in CUDA_VISIBLE_DEVICES.
    MPS / CPU: returns {} — caller must call .to(resolve_device()) afterwards.
    """
    import torch

    if torch.cuda.is_available():
        return {"device_map": "auto"}
    return {}
