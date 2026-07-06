"""Unit tests for the shared DPR psgs_w100 corpus loader.

These run fully offline against a tiny synthetic TSV (no ~13 GB download).
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench_datasets.dpr_corpus import corpus_available, load_dpr_passages


def _write_tsv(path: Path, rows: list[list[str]], header: bool = True) -> None:
    lines = []
    if header:
        lines.append("\t".join(["id", "text", "title"]))
    for r in rows:
        lines.append("\t".join(r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _corpus(tmp_path: Path) -> Path:
    p = tmp_path / "psgs_w100.tsv"
    _write_tsv(
        p,
        [
            ["1", "Aaron was a prophet.", "Aaron"],
            ["2", "Paris is the capital of France.", "Paris"],
            ["3", "", "EmptyText"],          # empty text -> skipped
            ["4", "No title passage.", ""],   # empty title -> text only
            ["5", "London is in England.", "London"],
        ],
    )
    return p


def test_loads_all_passages_formatted(tmp_path):
    docs = load_dpr_passages(_corpus(tmp_path), max_docs=None)
    assert docs == [
        "Aaron. Aaron was a prophet.",
        "Paris. Paris is the capital of France.",
        "No title passage.",             # empty title -> no "title. " prefix
        "London. London is in England.",
    ]  # the empty-text row is dropped, header is skipped


def test_max_docs_subset_is_bounded_and_deterministic(tmp_path):
    p = _corpus(tmp_path)
    a = load_dpr_passages(p, max_docs=2, seed=7)
    b = load_dpr_passages(p, max_docs=2, seed=7)
    assert len(a) == 2
    assert a == b                      # same seed -> identical reservoir sample
    assert set(a).issubset(set(load_dpr_passages(p, max_docs=None)))


def test_corpus_available(tmp_path):
    assert corpus_available(_corpus(tmp_path)) is True
    assert corpus_available(tmp_path / "missing.tsv") is False


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dpr_passages(tmp_path / "nope.tsv")
