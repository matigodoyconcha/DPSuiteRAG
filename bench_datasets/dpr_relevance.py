"""DPR question relevance counts for the DPVoteRAG relevance-binning filter.

Koga et al. ("Privacy-Preserving Retrieval-Augmented Generation with Differential
Privacy", arXiv:2412.04697v3) shape the Natural Questions / TriviaQA eval query
set by a question's *relevance*: the number of DPR gold positive passages for it,
i.e. ``len(positive_ctxs)`` in the DPR retriever dev files ``biencoder-nq-dev.json``
/ ``biencoder-trivia-dev.json``. The paper keeps questions with 20 <= relevance
< 60, splits them into 4 equal-width bins and samples 25 per bin. DP mechanisms
need many supporting documents, so low-relevance questions are structurally
unanswerable under DP; the filter removes them from evaluation.

Download the two dev files once with::

    python scripts/download_dpr_relevance.py

which writes them to :data:`DEFAULT_NQ_RELEVANCE_PATH` and
:data:`DEFAULT_TRIVIA_RELEVANCE_PATH`. When a file is absent the QA loaders skip
the relevance filter (with a warning) and select queries the ordinary way, so
smoke tests and offline runs still work without these files.
"""
from __future__ import annotations

import json
from pathlib import Path

# Project root = parent of the ``bench_datasets`` package.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Conventional on-disk location of the DPR Natural Questions dev relevance file.
DEFAULT_NQ_RELEVANCE_PATH = _PROJECT_ROOT / "data" / "dpr" / "biencoder-nq-dev.json"

#: Conventional on-disk location of the DPR TriviaQA dev relevance file.
DEFAULT_TRIVIA_RELEVANCE_PATH = _PROJECT_ROOT / "data" / "dpr" / "biencoder-trivia-dev.json"


def relevance_available(path: str | Path) -> bool:
    """Return ``True`` iff the DPR relevance JSON exists on disk at *path*."""
    return Path(path).is_file()


def normalize_question(text: str) -> str:
    """Canonicalize a question so it matches across sources.

    Lowercases, strips surrounding whitespace, drops a single trailing ``?`` and
    collapses internal whitespace runs to single spaces. Used to key relevance
    counts by question text when matching a loader's questions to the DPR file.
    """
    normalized = str(text).strip().lower()
    if normalized.endswith("?"):
        normalized = normalized[:-1]
    return " ".join(normalized.split())


def load_relevance_records(path: str | Path) -> list[dict]:
    """Load per-question relevance records from a DPR biencoder dev file.

    Each DPR biencoder record carries its own ``question``, ``answers`` list and
    ``positive_ctxs`` gold passages; a question's relevance is
    ``len(positive_ctxs)``. Koga et al. evaluate on the DPR dev questions
    themselves, so building the eval query set directly from these records pairs
    every question with its relevance and gold answers by construction (no
    cross-source question-text matching needed). A top-level list and a
    ``{key: list}`` wrapper are both accepted (the first list value is used for
    the wrapper).

    Args:
        path: Path to ``biencoder-nq-dev.json`` or ``biencoder-trivia-dev.json``.

    Returns:
        A list of ``{"question": str, "answers": list[str], "relevance": int}``,
        one entry per record with a non-empty question.

    Raises:
        FileNotFoundError: If *path* does not exist. Callers that want a graceful
            fallback should gate on :func:`relevance_available` first.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"DPR relevance file not found at {path}. Download it with "
            "`python scripts/download_dpr_relevance.py`."
        )

    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        raw_records: list = next((v for v in data.values() if isinstance(v, list)), [])
    elif isinstance(data, list):
        raw_records = data
    else:
        raw_records = []

    records: list[dict] = []
    for record in raw_records:
        if not isinstance(record, dict):
            continue
        question = record.get("question")
        if not question:
            continue
        positive_ctxs = record.get("positive_ctxs") or []
        answers = [str(a).strip() for a in (record.get("answers") or []) if str(a).strip()]
        records.append({
            "question": str(question),
            "answers": answers,
            "relevance": len(positive_ctxs),
        })
    return records


def load_relevance_counts(path: str | Path) -> dict[str, int]:
    """Load per-question relevance counts from a DPR biencoder dev file.

    The file is a list of records, each with a ``question`` string and a
    ``positive_ctxs`` list of gold passages; a question's relevance is
    ``len(positive_ctxs)``. A top-level list and a ``{key: list}`` wrapper are
    both accepted (the first list value is used for the wrapper).

    Args:
        path: Path to ``biencoder-nq-dev.json`` or ``biencoder-trivia-dev.json``.

    Returns:
        A mapping ``{normalize_question(question): relevance_count}``.

    Raises:
        FileNotFoundError: If *path* does not exist. Callers that want a graceful
            fallback should gate on :func:`relevance_available` first.
    """
    return {
        normalize_question(record["question"]): record["relevance"]
        for record in load_relevance_records(path)
    }
