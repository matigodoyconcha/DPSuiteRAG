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
    normalize_question,
    relevance_available,
)

_RECORDS = [
    {"question": "Who won the Cup?", "positive_ctxs": [1, 2, 3]},
    {"question": "Where is X ?", "positive_ctxs": []},          # -> 0
    {"question": "No positives key"},                            # missing -> 0
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
