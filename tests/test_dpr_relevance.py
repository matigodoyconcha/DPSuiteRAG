"""Unit tests for the DPR question-relevance loader (relevance-binning filter).

These run fully offline against a tiny synthetic biencoder JSON (no download).
"""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench_datasets.dpr_relevance import (
    load_relevance_counts,
    load_relevance_records,
    normalize_question,
    relevance_available,
)
from bench_datasets.natural_questions import NaturalQuestionsDataset
from bench_datasets.triviaqa import TriviaQADataset

_RECORDS = [
    {"question": "Who won the Cup?", "answers": ["Team A", "A"], "positive_ctxs": [1, 2, 3]},
    {"question": "Where is X ?", "answers": ["Nowhere"], "positive_ctxs": []},   # -> 0
    {"question": "No positives key", "answers": []},             # missing -> 0
    {"positive_ctxs": [1]},                                      # no question -> skipped
    "garbage",                                                   # non-dict -> skipped
]
_EXPECTED = {"who won the cup": 3, "where is x": 0, "no positives key": 0}


def _write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_normalize_question():
    assert normalize_question("  Who WON the  Cup? ") == "who won the cup"
    assert normalize_question("no trailing q") == "no trailing q"
    assert normalize_question("A?B?") == "a?b"  # only one trailing '?' dropped


def test_load_relevance_counts_top_level_list(tmp_path):
    counts = load_relevance_counts(_write_json(tmp_path / "dev.json", _RECORDS))
    assert counts == _EXPECTED


def test_load_relevance_counts_dict_wrapper(tmp_path):
    counts = load_relevance_counts(_write_json(tmp_path / "wrap.json", {"data": _RECORDS}))
    assert counts == _EXPECTED


def test_relevance_available(tmp_path):
    present = _write_json(tmp_path / "dev.json", _RECORDS)
    assert relevance_available(present) is True
    assert relevance_available(tmp_path / "missing.json") is False


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_relevance_counts(tmp_path / "nope.json")


def test_load_relevance_records(tmp_path):
    records = load_relevance_records(_write_json(tmp_path / "dev.json", _RECORDS))
    # One entry per record with a non-empty question; ordering preserved.
    assert [(r["question"], r["relevance"], r["answers"]) for r in records] == [
        ("Who won the Cup?", 3, ["Team A", "A"]),
        ("Where is X ?", 0, ["Nowhere"]),
        ("No positives key", 0, []),
    ]


def test_load_relevance_records_dict_wrapper(tmp_path):
    records = load_relevance_records(_write_json(tmp_path / "wrap.json", {"data": _RECORDS}))
    assert [r["relevance"] for r in records] == [3, 0, 0]


def test_load_relevance_records_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_relevance_records(tmp_path / "nope.json")


# --- Relevance-binning query construction (offline; no HF/network) ------------
# These drive _relevance_binned_queries directly on synthetic biencoder records,
# so load() (which hits HuggingFace) is never called.

SEED = 42


def _ds(cls, **kw):
    return cls(random_seed=SEED, max_test_queries=None, relevance_min=20,
              relevance_max=60, n_relevance_bins=4, per_bin=25, **kw)


@pytest.mark.parametrize("cls", [NaturalQuestionsDataset, TriviaQADataset])
def test_binned_queries_in_band_and_shape(cls):
    records = (
        [{"question": f"q{i}", "answers": [f"a{i}"], "relevance": 25} for i in range(30)]  # bin 0
        + [{"question": "low", "answers": ["x"], "relevance": 5}]     # below band -> dropped
        + [{"question": "high", "answers": ["y"], "relevance": 65}]   # above band -> dropped
        + [{"question": "b3", "answers": [], "relevance": 55}]        # bin 3, no answers
    )
    queries = _ds(cls)._relevance_binned_queries(records)
    # bin 0 capped at per_bin=25, bin 3 contributes its single record.
    assert len(queries) == 26
    for q in queries:
        assert set(q) == {"query", "answer", "answers", "context"}
        assert q["context"] == ""
    b3 = next(q for q in queries if q["query"] == "b3")
    assert b3["answer"] == "" and b3["answers"] == [""]  # empty answers -> [""]
    q0 = next(q for q in queries if q["query"] == "q0")
    assert q0["answer"] == "a0" and q0["answers"] == ["a0"]  # answers carried from record


@pytest.mark.parametrize("cls", [NaturalQuestionsDataset, TriviaQADataset])
def test_binned_queries_deterministic_under_seed(cls):
    records = [
        {"question": f"q{i}", "answers": [f"a{i}"], "relevance": 20 + (i % 40)}
        for i in range(400)
    ]
    q1 = _ds(cls)._relevance_binned_queries(records)
    q2 = _ds(cls)._relevance_binned_queries(records)
    assert [q["query"] for q in q1] == [q["query"] for q in q2]
    assert len(q1) == 100  # 25 per bin x 4 bins, all bins over-full


@pytest.mark.parametrize("cls", [NaturalQuestionsDataset, TriviaQADataset])
def test_binned_queries_shortfall_warns(cls):
    records = [{"question": "only", "answers": ["a"], "relevance": 22}]  # bin 0, 1 < per_bin
    with pytest.warns(UserWarning):
        queries = _ds(cls)._relevance_binned_queries(records)
    assert len(queries) == 1
