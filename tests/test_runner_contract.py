"""Tests for the contract every runner shares: config sweeping, loader kwargs, schema.

The reference condition is scored by the same offline scorer as the DP methods, so
the shape of what a runner writes is part of the benchmark's correctness, not a
presentation detail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench_datasets.movielens import MovielensDataset
from bench_datasets.natural_questions import NaturalQuestionsDataset
from experiments.common import (
    dataset_loader_kwargs,
    iter_dataset_configs,
    iter_llm_configs,
    llm_label,
)
from experiments.score_results import _extract_preds_refs


# ---------------------------------------------------------------------------
# Config sweeping
# ---------------------------------------------------------------------------

def test_dataset_entries_are_merged_over_the_singular_block():
    config = {
        "dataset": {"name": "squad", "max_index_docs": 200, "index_split": "train"},
        "datasets": [{"name": "chatdoctor"}, {"name": "movielens", "max_index_docs": 800}],
    }
    entries = iter_dataset_configs(config)
    assert [e["name"] for e in entries] == ["chatdoctor", "movielens"]
    # Defaults are inherited, overrides win.
    assert entries[0]["max_index_docs"] == 200
    assert entries[0]["index_split"] == "train"
    assert entries[1]["max_index_docs"] == 800


def test_singular_block_is_used_when_there_is_no_list():
    config = {"dataset": {"name": "squad", "max_index_docs": 200}}
    assert iter_dataset_configs(config) == [{"name": "squad", "max_index_docs": 200}]


def test_llm_entries_are_merged_the_same_way():
    config = {
        "llm": {"name": "huggingface", "model_name": "sshleifer/tiny-gpt2"},
        "llms": [{"model_name": "facebook/opt-1.3b"}],
    }
    entries = iter_llm_configs(config)
    assert entries == [{"name": "huggingface", "model_name": "facebook/opt-1.3b"}]
    assert llm_label(entries[0]) == "facebook__opt-1.3b"


# ---------------------------------------------------------------------------
# Loader kwargs: a filter declared in the YAML must reach the loader
# ---------------------------------------------------------------------------

def test_loader_receives_the_filters_its_constructor_accepts():
    ds_cfg = {
        "name": "movielens",
        "max_index_docs": 800,
        "top_movies": 30,
        "min_high_ratings": 5,
    }
    kwargs = dataset_loader_kwargs(MovielensDataset, ds_cfg)
    assert kwargs["top_movies"] == 30
    assert kwargs["min_high_ratings"] == 5
    assert kwargs["max_index_docs"] == 800
    assert "name" not in kwargs


def test_loader_does_not_receive_keys_it_cannot_accept():
    ds_cfg = {"name": "natural_questions", "relevance_min": 20, "top_movies": 30}
    kwargs = dataset_loader_kwargs(NaturalQuestionsDataset, ds_cfg)
    assert kwargs["relevance_min"] == 20
    # top_movies belongs to another loader; forwarding it would be a TypeError.
    assert "top_movies" not in kwargs


def test_forwarded_kwargs_construct_the_loader():
    ds_cfg = {"name": "movielens", "top_movies": 30, "max_test_queries": 10}
    MovielensDataset(**dataset_loader_kwargs(MovielensDataset, ds_cfg))


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

def _vanilla_payload(items: list[dict]) -> dict:
    return {
        "method": "vanilla",
        "dataset": "squad",
        "results": items,
        "predictions": [it["prediction"] for it in items],
        "references": [it["reference"] for it in items],
    }


def test_scorer_reads_the_predictions_a_run_writes():
    payload = _vanilla_payload(
        [
            {"query": "q1", "prediction": "Paris", "reference": "Paris"},
            {"query": "q2", "prediction": "1945", "reference": "1945"},
        ]
    )
    preds, refs = _extract_preds_refs(payload)
    assert preds == ["Paris", "1945"]
    assert refs == ["Paris", "1945"]


def test_an_answer_keyed_payload_scores_as_empty():
    # Guards the failure this schema replaced: the scorer reads `prediction`, so a
    # payload that only carries `answer` is silently scored against empty strings —
    # it does not raise, it reports a perfect run as zero.
    payload = {
        "method": "vanilla",
        "results": [{"query": "q1", "answer": "Paris", "reference": "Paris"}],
    }
    preds, _ = _extract_preds_refs(payload)
    assert preds == [""]


def test_dataset_is_a_string_not_a_config_block():
    # The CSV writes this field verbatim; a dict there lands in the summary as a dict.
    payload = _vanilla_payload([{"query": "q", "prediction": "a", "reference": "a"}])
    assert isinstance(payload["dataset"], str)


@pytest.mark.parametrize("key", ["favorites", "retrieved_ids"])
def test_set_valued_gold_and_retrieved_ids_survive_into_the_items(key):
    item = {
        "query": "q",
        "prediction": "Toy Story",
        "reference": "Toy Story",
        "favorites": ["Toy Story", "Fargo"],
        "retrieved_ids": [3, 7, 11],
    }
    payload = _vanilla_payload([item])
    assert payload["results"][0][key]
