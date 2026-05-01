"""
Shared index caching utility for all experiment runners.

Builds a FAISS index once per process and saves it to a temporary directory
whose name is derived from the dataset + pipeline configuration. On process
exit the temporary directory is deleted automatically via :mod:`atexit`, so
disk space is never permanently consumed by cached indices.

Typical usage::

    cache_dir = index_cache_dir(
        dataset_name="triviaqa",
        max_docs=200,
        embedder_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=256,
        chunk_overlap=32,
        seed=42,
    )
    load_or_build_index(
        indexer=indexer,
        build_fn=lambda: pipeline.index(documents),
        cache_dir=cache_dir,
    )
"""

from __future__ import annotations

import atexit
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable


def index_cache_dir(
    *,
    dataset_name: str,
    max_docs: int | None,
    embedder_model: str,
    chunk_size: int,
    chunk_overlap: int,
    seed: int,
    no_chunk: bool = False,
) -> Path:
    """Return a stable temporary path for a given index configuration.

    The path is deterministic: the same arguments always map to the same
    directory, so a second process with identical settings can reuse an
    index built by the first one while it is still running.

    The directory is located in the system's temp folder and is named
    ``dp_rag_idx_<12-char-hex-hash>``.  It is *not* created here; call
    :func:`load_or_build_index` to actually populate it.

    Args:
        dataset_name: Identifier of the dataset (e.g. ``"triviaqa"``).
        max_docs: Maximum number of documents loaded into the index, or
            ``None`` if no limit was applied.
        embedder_model: HuggingFace model name used to embed chunks.
        chunk_size: Number of tokens/characters per chunk (ignored when
            ``no_chunk=True``).
        chunk_overlap: Overlap between consecutive chunks (ignored when
            ``no_chunk=True``).
        seed: Random seed used when loading the dataset.
        no_chunk: When ``True`` each document is indexed as a single entry
            (one document = one privacy unit, as required for correct DP
            guarantees in DP-RAG).  The cache key differs from any chunked
            configuration so old cached indices are not reused.

    Returns:
        A :class:`~pathlib.Path` pointing to the cache directory (may not
        exist yet).
    """
    key = json.dumps(
        {
            "dataset_name": dataset_name,
            "max_docs": max_docs,
            "embedder_model": embedder_model,
            "no_chunk": no_chunk,
            "chunk_size": None if no_chunk else chunk_size,
            "chunk_overlap": None if no_chunk else chunk_overlap,
            "seed": seed,
        },
        sort_keys=True,
    )
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"dp_rag_idx_{digest}"


def load_or_build_index(
    indexer,
    build_fn: Callable[[], None],
    cache_dir: Path,
) -> None:
    """Load the FAISS index from *cache_dir* or build it from scratch.

    If *cache_dir* already exists (because a concurrent or prior run with
    the same configuration created it), the index is loaded directly,
    skipping the embedding + indexing step.

    Otherwise *build_fn* is called to build the index (which must populate
    *indexer* as a side-effect), the result is persisted to *cache_dir*,
    and an :mod:`atexit` hook is registered to delete the directory when the
    current process exits — whether normally, via an exception, or via
    ``KeyboardInterrupt``.

    Args:
        indexer: A :class:`~vanilla.indexer.FAISSIndexer` instance whose
            ``load`` / ``save`` methods are used for persistence.
        build_fn: Zero-argument callable that builds the index in *indexer*
            (e.g. ``lambda: pipeline.index(documents)``).
        cache_dir: Directory path returned by :func:`index_cache_dir`.
    """
    if cache_dir.exists():
        print(f"Loading cached index from {cache_dir} ...")
        indexer.load(str(cache_dir))
    else:
        build_fn()
        indexer.save(str(cache_dir))
        atexit.register(shutil.rmtree, str(cache_dir), ignore_errors=True)
        print(f"Index cached at {cache_dir} (will be deleted on exit).")
