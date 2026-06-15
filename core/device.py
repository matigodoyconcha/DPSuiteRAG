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


def free_memory() -> None:
    """Release cached GPU and host memory between runs to avoid OOM.

    Runs the Python garbage collector and empties the CUDA / MPS allocator
    caches when available. Safe to call on any backend (no-op on CPU). Call it
    between epsilon iterations and, for generation-heavy methods, between queries.
    """
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available() and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
