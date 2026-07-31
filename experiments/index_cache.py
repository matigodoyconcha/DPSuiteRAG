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
    max_document_words: int | None = None,
    store_embeddings_matrix: bool = True,
    build_faiss_index: bool = True,
    loader_kwargs: dict | None = None,
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
        loader_kwargs: Every argument the dataset loader was constructed with.
            Two entries naming the same dataset but filtering it differently — a
            relevance band, a catalogue cap, a support binning, a different
            sampling seed — build genuinely different corpora, and keying only on
            the dataset name would hand the second one the first one's index and
            evaluate it against a corpus it never selected. Nothing would fail;
            the numbers would simply belong to the wrong corpus.

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
            # Truncating documents changes the indexed text without changing how many
            # entries there are, and chunk_size drops out of the key when documents are
            # indexed whole, so the cap needs a key of its own.
            "max_document_words": max_document_words,
            # Distinct dirs for matrix vs matrix-free indices, so a method that needs
            # embeddings_matrix (DP-RAG / Private-RAG) never loads a matrix-free index
            # built by a FAISS-top-k method sharing the same corpus, and vice versa.
            "store_embeddings_matrix": store_embeddings_matrix,
            # Likewise separate dirs for indices built without a FAISS index (DP-RAG /
            # Private-RAG skip it and score via the matrix): a FAISS-search retriever
            # must never load a matrix-only cache that has no index.faiss.
            "build_faiss_index": build_faiss_index,
            # Scope the directory to one orchestrated run so concurrent runs with
            # identical settings never share — and therefore never delete — each
            # other's index. Unset (standalone) keeps the old shared, self-cleaning
            # path.
            "run_id": os.environ.get("BENCHDP_RUN_ID"),
            # Which documents the loader actually selected. Serialised with sorted
            # keys at every depth so a nested block written in a different order
            # still maps to one directory; `default` keeps an unserialisable
            # argument contributing to the key instead of raising.
            "loader_kwargs": (
                json.dumps(loader_kwargs, sort_keys=True, default=repr)
                if loader_kwargs
                else None
            ),
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
    # The standalone MIA run (BENCHDP_MIA_ONLY) indexes only a member subset of the
    # corpus, whose identity is not part of the cache key (which keys on max_docs,
    # not document contents). Building fresh each time avoids serving a full-corpus
    # or a different method's member index; member corpora are small, so this is
    # cheap and keeps the MIA measurement independent of the utility runs.
    if os.environ.get("BENCHDP_MIA_ONLY") == "1":
        build_fn()
        return

    manifest = os.environ.get("BENCHDP_INDEX_MANIFEST")
    if manifest:
        # Record before building so a crash mid-build still leaves a cleanup entry
        # (the directory may be created and only partially written).
        _record_in_manifest(manifest, cache_dir)
    if _is_complete(cache_dir):
        print(f"Loading cached index from {cache_dir} ...")
        indexer.load(str(cache_dir))
        if not getattr(indexer, "metadata", None):
            raise RuntimeError(
                f"cached index at {cache_dir} loaded no metadata; refusing to run "
                "every query against an empty corpus"
            )
    else:
        if cache_dir.exists():
            # Left behind by a build that died partway. The retriever returns an empty
            # list for such an index without raising, so the cell would run to
            # completion, write results scored against no documents at all, and exit 0.
            print(f"Discarding incomplete cached index at {cache_dir} ...")
            shutil.rmtree(cache_dir, ignore_errors=True)
        build_fn()
        indexer.save(str(cache_dir))
        _mark_complete(cache_dir)
        if manifest:
            print(f"Index cached at {cache_dir} (removed once the dataset finishes).")
        else:
            atexit.register(shutil.rmtree, str(cache_dir), ignore_errors=True)
            print(f"Index cached at {cache_dir} (will be deleted on exit).")


#: Written only after `indexer.save` returns, so a directory without it is the debris
#: of an interrupted build rather than a usable index.
_COMPLETION_MARKER = ".index_complete"


def _is_complete(cache_dir: Path) -> bool:
    return (cache_dir / _COMPLETION_MARKER).exists()


def _mark_complete(cache_dir: Path) -> None:
    try:
        (cache_dir / _COMPLETION_MARKER).touch()
    except OSError:
        # A cache that cannot be marked is simply rebuilt next time; failing the run
        # over it would be worse than the rebuild.
        pass


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
