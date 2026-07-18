"""
Shared index caching utility for all experiment runners.

Builds a FAISS index once per process and saves it to a temporary directory
whose name is derived from the dataset + pipeline configuration. On process
exit the temporary directory is deleted automatically via :mod:`atexit`, so
disk space is never permanently consumed by cached indices.

Typical usage::

    cache_dir = index_cache_dir(
        dataset_name="triviaqa",
        corpus_id="dpr_psgs_w100",
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
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable


def index_cache_dir(
    *,
    dataset_name: str,
    corpus_id: str | None = None,
    max_docs: int | None,
    embedder_model: str,
    chunk_size: int,
    chunk_overlap: int,
    seed: int,
    no_chunk: bool = False,
    store_embeddings_matrix: bool = True,
) -> Path:
    """Return a stable temporary path for a given index configuration.

    The path is deterministic within a run: the same arguments (and the same
    ``BENCHDP_RUN_ID``) always map to the same directory, so sibling processes
    of one run reuse the index the first one builds. Distinct runs get distinct
    directories, so a concurrent run never deletes an index still in use.

    The directory is located in the system's temp folder and is named
    ``dp_rag_idx_<12-char-hex-hash>``.  It is *not* created here; call
    :func:`load_or_build_index` to actually populate it.

    Args:
        dataset_name: Identifier of the dataset (e.g. ``"triviaqa"``).
        corpus_id: Identifier of the underlying document corpus. When two
            datasets read from the same corpus with identical settings they map
            to the same directory and reuse one index (e.g. Natural Questions
            and TriviaQA both retrieve from the DPR ``psgs_w100`` corpus). When
            ``None`` the key falls back to *dataset_name*, so each dataset keeps
            its own directory.
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
            # Identical corpora share one directory: datasets that pass the same
            # corpus_id (e.g. NQ and TriviaQA over the DPR psgs_w100 corpus) with
            # equal settings collide here and reuse a single index. Falls back to
            # dataset_name when corpus_id is None (each dataset stays separate).
            "corpus": corpus_id or dataset_name,
            "max_docs": max_docs,
            "embedder_model": embedder_model,
            "no_chunk": no_chunk,
            "chunk_size": None if no_chunk else chunk_size,
            "chunk_overlap": None if no_chunk else chunk_overlap,
            "seed": seed,
            # Distinct dirs for matrix vs matrix-free indices, so a method that needs
            # embeddings_matrix (DP-RAG / Private-RAG) never loads a matrix-free index
            # built by a FAISS-top-k method sharing the same corpus, and vice versa.
            "store_embeddings_matrix": store_embeddings_matrix,
            # Scope the directory to one orchestrated run so concurrent runs with
            # identical settings never share — and therefore never delete — each
            # other's index. Unset (standalone) keeps the old shared, self-cleaning
            # path.
            "run_id": os.environ.get("BENCHDP_RUN_ID"),
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
    *indexer* as a side-effect) and the result is persisted to *cache_dir*.

    Cleanup depends on who owns the index lifecycle:

    * **Standalone** (the default): an :mod:`atexit` hook removes the freshly
      built directory when the process exits — whether normally, via an
      exception, or via ``KeyboardInterrupt`` — so a lone method run never
      leaves an index behind.
    * **Orchestrated**: when the ``BENCHDP_INDEX_MANIFEST`` environment variable
      names a file, the index is left on disk (so sibling method/LLM
      subprocesses for the same dataset reuse it) and its path is appended to
      that manifest. Whoever set the variable deletes the listed directories
      once every method for the dataset has finished, keeping at most one
      dataset's indices on disk at a time.

    Args:
        indexer: A :class:`~vanilla.indexer.FAISSIndexer` instance whose
            ``load`` / ``save`` methods are used for persistence.
        build_fn: Zero-argument callable that builds the index in *indexer*
            (e.g. ``lambda: pipeline.index(documents)``).
        cache_dir: Directory path returned by :func:`index_cache_dir`.
    """
    manifest = os.environ.get("BENCHDP_INDEX_MANIFEST")
    if manifest:
        # Record before building so a crash mid-build still leaves a cleanup entry
        # (the directory may be created and only partially written).
        _record_in_manifest(manifest, cache_dir)
    if cache_dir.exists():
        print(f"Loading cached index from {cache_dir} ...")
        indexer.load(str(cache_dir))
    else:
        build_fn()
        indexer.save(str(cache_dir))
        if manifest:
            print(f"Index cached at {cache_dir} (removed once the dataset finishes).")
        else:
            atexit.register(shutil.rmtree, str(cache_dir), ignore_errors=True)
            print(f"Index cached at {cache_dir} (will be deleted on exit).")


def _record_in_manifest(manifest_path: str, cache_dir: Path) -> None:
    """Append *cache_dir* to the caller's per-dataset cleanup manifest.

    One directory per line. Appends of a short line are atomic, so concurrent
    method/LLM subprocesses can record safely; duplicate lines are harmless
    because the reader deduplicates before deleting.
    """
    try:
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(f"{cache_dir}\n")
    except OSError as exc:
        print(f"[WARN] could not record index {cache_dir} in manifest {manifest_path}: {exc}")
