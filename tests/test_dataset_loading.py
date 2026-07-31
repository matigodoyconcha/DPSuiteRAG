"""One loader path for every method, so a filter applies wherever it is declared."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench_datasets.movielens import MovielensDataset
from experiments.common import (
    dataset_loader_kwargs,
    load_dataset_from_config,
    normalise_dataset_config,
)

RUNNERS = [
    "run_dp_rag", "run_dpvoterag", "run_dpsparsevoterag", "run_dpksa",
    "run_private_rag", "run_dp_synrag",
]
CONFIGS = sorted(
    p for d in ("full", "paper_faithful")
    for p in (PROJECT_ROOT / "experiments" / "configs" / d).glob("*.yaml")
)


# ---------------------------------------------------------------------------
# Key aliases and the corpus/question split
# ---------------------------------------------------------------------------

def test_the_two_spellings_of_the_corpus_cap_resolve_to_one():
    assert normalise_dataset_config({"max_samples": 20000}) == {"max_index_docs": 20000}


def test_normalising_leaves_other_keys_alone():
    resolved = normalise_dataset_config({"name": "movielens", "top_movies": 30})
    assert resolved == {"name": "movielens", "top_movies": 30}


def test_no_config_still_uses_the_old_spelling():
    for path in CONFIGS:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = [config.get("dataset") or {}, *(config.get("datasets") or [])]
        for entry in entries:
            assert "max_samples" not in (entry or {}), (
                f"{path} still spells the corpus cap max_samples; two names for one "
                "quantity means the same corpus gets built twice"
            )


def test_the_question_count_does_not_default_to_the_corpus_size():
    # The old fallback asked for one question per indexed document: a 20000-document
    # corpus quietly became a 20000-question evaluation.
    resolved = normalise_dataset_config({"max_index_docs": 20000})
    assert "max_test_queries" not in resolved


# ---------------------------------------------------------------------------
# Filters reach the loader whichever method is running
# ---------------------------------------------------------------------------

def test_a_filter_declared_in_the_entry_reaches_the_loader():
    ds_cfg = {"name": "movielens", "top_movies": 30, "max_test_queries": 10}
    kwargs = dataset_loader_kwargs(MovielensDataset, normalise_dataset_config(ds_cfg))
    assert kwargs["top_movies"] == 30


@pytest.mark.parametrize("runner", RUNNERS)
def test_every_runner_loads_through_the_shared_path(runner):
    source = (PROJECT_ROOT / "experiments" / f"{runner}.py").read_text(encoding="utf-8")
    assert "load_dataset_from_config(" in source, (
        f"{runner} builds its own loader call, so a filter it was not written to know "
        "about is dropped without a word"
    )


@pytest.mark.parametrize("runner", RUNNERS)
def test_no_runner_keeps_a_private_loader(runner):
    source = (PROJECT_ROOT / "experiments" / f"{runner}.py").read_text(encoding="utf-8")
    assert "def load_dataset(" not in source
    assert "def _resolve_dataset(" not in source


def test_an_unknown_dataset_is_named_in_the_error():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        load_dataset_from_config({"name": "not_a_dataset"}, 42)


# ---------------------------------------------------------------------------
# A single delta for the common protocol
# ---------------------------------------------------------------------------

_FULL = sorted((PROJECT_ROOT / "experiments" / "configs" / "full").glob("*.yaml"))


@pytest.mark.parametrize("path", _FULL, ids=lambda p: p.stem)
def test_the_common_protocol_uses_one_delta(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    declared = [
        config.get(key) for key in ("delta", "delta_total") if config.get(key) is not None
    ]
    # Private-RAG is pure DP: delta = 0 is stronger than the target, not weaker.
    if not declared:
        return
    for value in declared:
        assert float(value) == pytest.approx(1e-5), (
            f"{path} reports delta={value}; the common axis is delta=1e-5 over the "
            "whole workload"
        )


@pytest.mark.parametrize("method", ["dpvoterag", "dpsparsevoterag"])
def test_the_per_token_delta_fits_the_workload(method):
    path = PROJECT_ROOT / "experiments" / "configs" / "full" / f"{method}.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    delta_token = float(config[method]["delta_token"])
    # Additive mass: charged once per token, per query. At the paper's 1e-6 a hundred
    # 50-token queries spend 5e-3, five hundred times the target.
    assert delta_token * 50 * 100 <= 1e-5
