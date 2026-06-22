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


def is_oom_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is an out-of-memory error (CUDA device or host RAM).

    Detects ``torch.cuda.OutOfMemoryError``, Python's builtin ``MemoryError``,
    and the generic ``RuntimeError("CUDA out of memory ...")`` some paths raise.
    The runners use this to skip an OOM (model, dataset) combination and continue
    the sweep instead of aborting everything. NOTE: an OS-level OOM-kill (SIGKILL)
    cannot be caught from Python — only in-process OOM exceptions are recoverable.
    """
    import torch

    if isinstance(exc, MemoryError):
        return True
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False
