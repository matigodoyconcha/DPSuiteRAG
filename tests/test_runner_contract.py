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
    corpus_fingerprint,
    dataset_loader_kwargs,
    iter_dataset_configs,
    iter_llm_configs,
    llm_label,
)
from experiments.index_cache import index_cache_dir
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
# Index cache: same name + different filters must not share a corpus
# ---------------------------------------------------------------------------

def _cache_kwargs(ds_cfg: dict, **over):
    base = dict(
        dataset_name=str(ds_cfg["name"]),
        corpus_id=None,
        max_docs=ds_cfg.get("max_index_docs"),
        embedder_model="Snowflake/snowflake-arctic-embed-m-v1.5",
        chunk_size=200,
        chunk_overlap=20,
        seed=42,
        loader_kwargs=corpus_fingerprint(ds_cfg),
    )
    base.update(over)
    return base


def test_the_same_filters_map_to_one_index():
    cfg = {"name": "movielens", "max_index_docs": 800, "top_movies": 30}
    assert index_cache_dir(**_cache_kwargs(cfg)) == index_cache_dir(**_cache_kwargs(cfg))


def test_a_different_corpus_filter_gets_its_own_index():
    # Two entries naming one dataset but selecting different documents. Sharing an
    # index here would evaluate the second against a corpus it never selected —
    # silently, with no error and plausible-looking numbers.
    wide = {"name": "movielens", "max_index_docs": 800}
    narrow = {"name": "movielens", "max_index_docs": 800, "top_movies": 30}
    assert index_cache_dir(**_cache_kwargs(wide)) != index_cache_dir(**_cache_kwargs(narrow))


def test_a_different_sampling_seed_gets_its_own_index():
    a = {"name": "chatdoctor", "max_index_docs": 20000, "random_seed": 1}
    b = {"name": "chatdoctor", "max_index_docs": 20000, "random_seed": 2}
    assert index_cache_dir(**_cache_kwargs(a)) != index_cache_dir(**_cache_kwargs(b))


@pytest.mark.parametrize(
    "dataset,key,other",
    [
        # SearchQA builds its corpus out of the very questions it keeps, so both the
        # count and the support binning move the documents.
        ("searchqa", "max_test_queries", 100),
        ("searchqa", "per_bin", 17),
        # MovieLens reserves the query users and indexes the rest; anything that
        # shifts that partition — including whether a user yields a query at all —
        # changes the corpus.
        ("movielens", "max_test_queries", 200),
        ("movielens", "top_k_favorites", 3),
        ("movielens", "seed_size", 5),
        # SQuAD seeds the index with each test question's gold context.
        ("squad", "max_test_queries", 100),
        ("squad", "test_split", "train"),
    ],
)
def test_question_side_keys_still_separate_a_question_dependent_corpus(dataset, key, other):
    base = {"name": dataset, "max_index_docs": 2000, key: 40}
    variant = {**base, key: other}
    assert index_cache_dir(**_cache_kwargs(base)) != index_cache_dir(**_cache_kwargs(variant))


def test_the_two_names_for_a_corpus_cap_map_to_one_index():
    # dp_rag says max_samples and everyone else says max_index_docs. Left unnormalised
    # the same corpus is built twice, which for the DPR passages is ~20 h and ~34 GB.
    old = {"name": "natural_questions", "max_samples": 20000}
    new = {"name": "natural_questions", "max_index_docs": 20000}
    shared = dict(corpus_id="dpr_psgs_w100", max_docs=20000)
    assert index_cache_dir(**_cache_kwargs(old, **shared)) == index_cache_dir(
        **_cache_kwargs(new, **shared)
    )


def test_a_reordered_config_block_maps_to_one_index():
    a = {"name": "chatdoctor", "overrides": {"a": 1, "b": 2}}
    b = {"name": "chatdoctor", "overrides": {"b": 2, "a": 1}}
    assert index_cache_dir(**_cache_kwargs(a)) == index_cache_dir(**_cache_kwargs(b))


def test_a_document_cap_separates_the_index():
    from experiments.common import chunking_fingerprint

    whole = chunking_fingerprint({"chunking": {"enabled": False}})
    capped = chunking_fingerprint({"chunking": {"enabled": False, "max_document_words": 400}})
    kwargs = _cache_kwargs({"name": "chatdoctor", "max_index_docs": 2000})
    kwargs.pop("chunk_size"), kwargs.pop("chunk_overlap")
    assert index_cache_dir(**kwargs, **whole) != index_cache_dir(**kwargs, **capped)


def test_query_side_filters_still_share_the_dpr_index():
    # Natural Questions and TriviaQA both retrieve from DPR psgs_w100 and differ only
    # in their relevance filters, which select test questions rather than documents.
    # Keying on those would build the same ~34 GB index twice.
    nq = {
        "name": "natural_questions", "max_index_docs": None,
        "relevance_min": 20, "relevance_max": 60, "per_bin": 25,
        "relevance_path": "data/dpr/biencoder-nq-dev.json",
        "max_test_queries": 100,
    }
    tqa = {
        "name": "triviaqa", "max_index_docs": None,
        "relevance_min": 20, "relevance_max": 60, "per_bin": 25,
        "relevance_path": "data/dpr/biencoder-trivia-dev.json",
        "max_test_queries": 40,
    }
    shared = dict(corpus_id="dpr_psgs_w100")
    assert index_cache_dir(**_cache_kwargs(nq, **shared)) == index_cache_dir(
        **_cache_kwargs(tqa, **shared)
    )


def test_searchqa_support_filters_are_treated_as_corpus_affecting():
    # SearchQA builds its corpus and its queries from the same questions, so a filter
    # that changes which questions are kept also changes what gets indexed.
    a = {"name": "searchqa", "max_index_docs": 1410}
    b = {"name": "searchqa", "max_index_docs": 1410, "min_answer_snippets": 40}
    assert index_cache_dir(**_cache_kwargs(a)) != index_cache_dir(**_cache_kwargs(b))


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


def test_a_machine_can_be_given_the_subset_of_datasets_it_can_hold():
    """`--datasets` splits the grid; a name no method runs is a typo, not an empty run.

    The corpora differ in what they need — Natural Questions and TriviaQA index the
    DPR corpus and the rest are self-contained — so the grid is split across machines
    by naming datasets. Silently accepting an unknown name would produce a run that
    looks complete and covers nothing.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "experiments/runner.py",
         "--datasets", "no_such_dataset", "--methods", "vanilla"],
        capture_output=True, text=True, cwd=root,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "no configured method runs ['no_such_dataset']" in combined
    # The message must name the real ones, or the reader cannot fix the typo.
    assert "chatdoctor" in combined and "triviaqa" in combined
