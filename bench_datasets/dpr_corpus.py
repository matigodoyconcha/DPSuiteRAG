"""Shared DPR Wikipedia passage corpus (``psgs_w100``) for open-domain QA.

The paper-faithful retrieval corpus for Natural Questions and TriviaQA is the
DPR ``psgs_w100`` split: ~21M 100-word Wikipedia passages (Karpukhin et al.,
2020, "Dense Passage Retrieval for Open-Domain Question Answering"), the same
corpus both datasets retrieve from in the DP-KSA, DP-SparseVoteRAG and
Private-RAG papers.

By default the QA loaders retrieve from this corpus. Download it once with::

    python scripts/download_dpr_corpus.py

which writes the decompressed TSV to :data:`DEFAULT_DPR_CORPUS_PATH`. When that
file is absent the loaders fall back to their legacy per-dataset contexts (with
a warning), so smoke tests and offline runs still work without the ~13 GB file.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

# Project root = parent of the ``bench_datasets`` package.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Conventional on-disk location of the decompressed DPR corpus TSV.
DEFAULT_DPR_CORPUS_PATH = _PROJECT_ROOT / "data" / "dpr" / "psgs_w100.tsv"

# The psgs_w100.tsv fields can hold passages far longer than csv's default cap.
csv.field_size_limit(10_000_000)


def corpus_available(path: str | Path = DEFAULT_DPR_CORPUS_PATH) -> bool:
    """Return ``True`` iff the DPR corpus TSV exists on disk at *path*."""
    return Path(path).is_file()


def load_dpr_passages(
    path: str | Path = DEFAULT_DPR_CORPUS_PATH,
    max_docs: int | None = None,
    seed: int = 42,
) -> list[str]:
    """Load passages from the DPR ``psgs_w100`` TSV as indexable documents.

    The TSV has a header row ``id\\ttext\\ttitle`` followed by one passage per
    row. Each passage is returned as ``"{title}. {text}"`` (title dropped when
    empty). When *max_docs* is smaller than the corpus a **deterministic seeded
    reservoir sample** is returned, so the subset is a representative spread over
    all of Wikipedia rather than just the first rows. ``max_docs=None`` loads the
    whole corpus (~21M passages — memory-heavy; prefer a bounded subset).

    Args:
        path: Path to the decompressed ``psgs_w100.tsv``.
        max_docs: Maximum number of passages to return, or ``None`` for all.
        seed: Seed for the reservoir sample (deterministic subset).

    Returns:
        A list of passage strings.

    Raises:
        FileNotFoundError: If *path* does not exist. Callers that want a graceful
            fallback should gate on :func:`corpus_available` first.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"DPR corpus not found at {path}. Download it with "
            "`python scripts/download_dpr_corpus.py`."
        )

    rng = random.Random(seed)
    reservoir: list[str] = []
    n_valid = 0  # 0-based index of the current valid passage (reservoir sampling)

    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for i, row in enumerate(reader):
            if i == 0:
                continue  # header: ["id", "text", "title"]
            if len(row) < 3:
                continue
            text = row[1].strip()
            title = row[2].strip()
            if not text:
                continue
            doc = f"{title}. {text}" if title else text

            if max_docs is None:
                reservoir.append(doc)
            elif len(reservoir) < max_docs:
                reservoir.append(doc)
            else:
                j = rng.randint(0, n_valid)
                if j < max_docs:
                    reservoir[j] = doc
            n_valid += 1

    return reservoir
