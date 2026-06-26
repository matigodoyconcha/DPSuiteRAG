import logging
from contextlib import contextmanager
from typing import Iterator

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("benchdp.device")


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


def _cuda_memory_summary() -> str:
    """Human-readable free/total CUDA memory, or '' when unavailable."""
    import torch

    try:
        free, total = torch.cuda.mem_get_info()
        gib = 1024 ** 3
        return f"{free / gib:.2f} GiB free / {total / gib:.2f} GiB total"
    except Exception:
        return ""


def warn_if_cuda_oom(
    exc: BaseException, *, model_name: str | None = None, device: str | None = None
) -> bool:
    """Log a warning when ``exc`` is a CUDA out-of-memory error for a GPU placement.

    The single, centralized place that decides "this model did not fit on CUDA"
    and emits the warning (model name + the GPU's free/total memory). Returns
    ``True`` when a warning was emitted, ``False`` otherwise — so callers that
    silently fall back to CPU can still surface the reason.

    Args:
        exc: the exception raised by the allocation.
        model_name: identifier shown in the warning (defaults to "model").
        device: the *intended* device. The warning only fires for CUDA placements
            (``"cuda"``/``"cuda:N"``/``"auto"``); pass ``None`` to fire on any OOM.
    """
    if device is not None:
        dev = str(device)
        if not (dev.startswith("cuda") or dev == "auto"):
            return False
    if not is_oom_error(exc):
        return False
    where = f" on {device}" if device else ""
    mem = _cuda_memory_summary()
    suffix = f" ({mem})" if mem else ""
    logger.warning(
        "CUDA out of memory allocating '%s'%s — model does not fit%s",
        model_name or "model",
        where,
        suffix,
    )
    return True


@contextmanager
def cuda_oom_warning(
    model_name: str | None = None, device: str | None = None
) -> Iterator[None]:
    """Context manager that warns (via :func:`warn_if_cuda_oom`) on a CUDA OOM.

    Wrap any attempt to place a model on CUDA — a constructor that loads weights
    onto the GPU (e.g. ``SentenceTransformer(...)``) or an explicit
    ``.to("cuda")`` — so an out-of-memory failure is logged with the model name
    *before* the exception propagates. The exception is re-raised unchanged, so
    the runners' OOM-skip logic (:func:`is_oom_error`) still applies. Use
    :func:`warn_if_cuda_oom` directly instead at call sites that catch the OOM
    themselves to fall back to CPU.

    Args:
        model_name: identifier shown in the warning (defaults to "model").
        device: intended device; defaults to :func:`resolve_device`.
    """
    if device is None:
        device = resolve_device()
    try:
        yield
    except BaseException as exc:
        warn_if_cuda_oom(exc, model_name=model_name, device=device)
        raise
