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

def test_a_dataset_override_drops_the_list_so_the_flag_has_an_effect():
    import argparse

    import experiments.run_vanilla as rv

    config = {
        "dataset": {"name": "squad", "max_index_docs": 200},
        "datasets": [{"name": "chatdoctor"}, {"name": "triviaqa"}],
        "llm": {}, "models": {}, "pipeline": {}, "generation": {},
    }
    args = argparse.Namespace(
        **{k: None for k in (
            "dataset", "llm_name", "llm_model", "embedder_model", "k", "max_new_tokens",
            "seed", "output", "index_split", "test_split", "max_index_docs",
            "max_test_queries", "chunk_size", "chunk_overlap", "docs_path", "queries_path",
        )}
    )
    args.dataset = "triviaqa"
    out = rv.apply_cli_overrides(config, args)
    assert iter_dataset_configs(out) == [{"name": "triviaqa", "max_index_docs": 200}]


def test_a_dataset_scoped_override_reaches_every_entry():
    import argparse

    import experiments.run_vanilla as rv

    config = {
        "dataset": {"name": "squad", "max_index_docs": 200},
        "datasets": [{"name": "chatdoctor"}, {"name": "triviaqa", "max_index_docs": 50}],
        "llm": {}, "models": {}, "pipeline": {}, "generation": {},
    }
    args = argparse.Namespace(
        **{k: None for k in (
            "dataset", "llm_name", "llm_model", "embedder_model", "k", "max_new_tokens",
            "seed", "output", "index_split", "test_split", "max_index_docs",
            "max_test_queries", "chunk_size", "chunk_overlap", "docs_path", "queries_path",
        )}
    )
    args.max_index_docs = 20000
    out = rv.apply_cli_overrides(config, args)
    # An entry's own key wins the merge, so writing only to the singular block would
    # leave the flag with no effect wherever an entry set the same key.
    assert [e["max_index_docs"] for e in iter_dataset_configs(out)] == [20000, 20000]


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
    with pytest.warns(RuntimeWarning, match="top_movies"):
        kwargs = dataset_loader_kwargs(NaturalQuestionsDataset, ds_cfg)
    assert kwargs["relevance_min"] == 20
    # top_movies belongs to another loader; forwarding it would be a TypeError.
    assert "top_movies" not in kwargs


def test_a_key_no_loader_accepts_is_reported_rather_than_dropped():
    # A corpus cap under the wrong key, or a mistyped filter name, would otherwise
    # leave the run at its defaults while looking configured.
    with pytest.warns(RuntimeWarning, match="max_samples"):
        kwargs = dataset_loader_kwargs(
            NaturalQuestionsDataset, {"name": "natural_questions", "max_samples": 20000}
        )
    assert kwargs == {}


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


def test_the_runner_writes_a_payload_the_scorer_can_read(monkeypatch, tmp_path):
    """Exercise run_experiment itself, with the pipeline stubbed out.

    Asserting on a hand-built dict would only restate the test's own literals; the
    contract worth protecting is what run_vanilla actually emits, so the model, the
    embedder and the index are replaced and the real payload-building code runs.
    """
    import experiments.run_vanilla as rv

    class _Chunk(dict):
        pass

    monkeypatch.setattr(rv, "load_llm", lambda cfg: _StubLLM(cfg))
    monkeypatch.setattr(rv, "HFEmbedder", lambda model_name: _StubEmbedder())
    monkeypatch.setattr(rv, "FAISSIndexer", _StubIndexer)
    monkeypatch.setattr(rv, "CosineSimilarityRetriever", lambda indexer: _StubRetriever())
    monkeypatch.setattr(rv, "load_or_build_index", lambda **kw: None)
    monkeypatch.setattr(
        rv,
        "load_data_sources",
        lambda ds_cfg, seed: (
            ["doc one", "doc two"],
            [{"query": "q1", "answer": "Paris", "favorites": ["Paris", "Rome"]}],
            "{context}\n{query}",
        ),
    )

    config = rv.default_config()
    config["output_path"] = str(tmp_path / "out.json")
    payload = rv.run_experiment(config, {"name": "movielens"}, config["llm"])

    # The scorer must find predictions where it looks for them.
    preds, refs = _extract_preds_refs(payload)
    assert preds == ["stub answer"]
    assert refs == ["Paris"]
    # dataset is a plain name: the CSV writes this field verbatim.
    assert payload["dataset"] == "movielens"
    # Set-valued gold and document ids survive, so hit_rate and retrieval precision
    # have their inputs.
    item = payload["results"][0]
    assert item["favorites"] == ["Paris", "Rome"]
    assert item["retrieved_ids"] == [0, 1]
    assert all(i is not None for i in item["retrieved_ids"])


class _StubLLM:
    def __init__(self, cfg):
        self._cfg = {"name": "stub", "model_name": cfg.get("model_name", "stub")}

    def get_config(self):
        return self._cfg

    def build_generator(self, **_):
        return _StubGenerator()


class _StubGenerator:
    def generate(self, prompt):
        return "stub answer"


class _StubEmbedder:
    def embed_query(self, query):
        return [0.0, 1.0]


class _StubIndexer:
    def __init__(self, *a, **kw):
        self.index = None
        self.metadata = []


class _StubRetriever:
    def retrieve(self, query_vector, k):
        return [
            {"doc_id": 0, "chunk_id": 0, "text": "doc one", "score": 0.9},
            {"doc_id": 1, "chunk_id": 0, "text": "doc two", "score": 0.5},
        ]
