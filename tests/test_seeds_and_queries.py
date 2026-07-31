"""Repetitions vary the mechanism, and every method answers the same questions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (
    corpus_fingerprint,
    query_id,
    resolve_seeds,
    seed_label,
    select_queries,
)
from experiments.index_cache import index_cache_dir


# ---------------------------------------------------------------------------
# data_seed vs run_seed
# ---------------------------------------------------------------------------

def test_one_seed_in_the_config_drives_both(monkeypatch):
    monkeypatch.delenv("BENCHDP_RUN_SEED", raising=False)
    assert resolve_seeds({"random_seed": 7}) == (7, 7)


def test_a_seed_nested_in_a_method_block_is_still_the_run_seed(monkeypatch):
    # Four runners keep their seed inside the method's own block rather than at the
    # top level; where the YAML puts it must not change what it means.
    monkeypatch.delenv("BENCHDP_RUN_SEED", raising=False)
    assert resolve_seeds({"dpvoterag": {"random_seed": 5}}) == (5, 5)
    assert resolve_seeds({"private_rag": {"random_seed": 9}}) == (9, 9)


def test_the_default_is_used_when_nothing_declares_a_seed(monkeypatch):
    monkeypatch.delenv("BENCHDP_RUN_SEED", raising=False)
    assert resolve_seeds({}) == (42, 42)


def test_a_repetition_moves_the_noise_and_leaves_the_corpus_alone(monkeypatch):
    monkeypatch.setenv("BENCHDP_RUN_SEED", "3")
    data_seed, run_seed = resolve_seeds({"random_seed": 42})
    assert run_seed == 3
    # The corpus and the questions must not move, or the repetition would resample
    # the data and rebuild the index instead of re-rolling the mechanism.
    assert data_seed == 42


def test_every_repetition_reuses_one_index(monkeypatch):
    ds_cfg = {"name": "chatdoctor", "max_index_docs": 20000}

    def cache_dir_for(run_seed: str) -> Path:
        monkeypatch.setenv("BENCHDP_RUN_SEED", run_seed)
        data_seed, _ = resolve_seeds({"random_seed": 42})
        return index_cache_dir(
            dataset_name="chatdoctor",
            max_docs=20000,
            embedder_model="e",
            chunk_size=256,
            chunk_overlap=32,
            seed=data_seed,
            loader_kwargs=corpus_fingerprint(ds_cfg),
        )

    # Three repetitions of a cell must not trigger three builds of the same corpus.
    assert cache_dir_for("1") == cache_dir_for("2") == cache_dir_for("3")


def test_data_seed_can_be_moved_on_its_own(monkeypatch):
    monkeypatch.delenv("BENCHDP_RUN_SEED", raising=False)
    assert resolve_seeds({"random_seed": 42, "data_seed": 7}) == (7, 42)


def test_the_environment_wins_over_the_config(monkeypatch):
    monkeypatch.setenv("BENCHDP_RUN_SEED", "11")
    assert resolve_seeds({"random_seed": 42, "run_seed": 99})[1] == 11


def test_seed_label_names_the_repetition(monkeypatch):
    monkeypatch.setenv("BENCHDP_RUN_SEED", "3")
    assert seed_label() == "seed_3"
    monkeypatch.delenv("BENCHDP_RUN_SEED")
    assert seed_label() is None


# ---------------------------------------------------------------------------
# Shared question selection
# ---------------------------------------------------------------------------

_QUERIES = [{"query": f"question {i}", "answer": str(i)} for i in range(10)]


def test_the_cut_takes_a_prefix_in_order():
    picked = select_queries(_QUERIES, max_queries=3)
    assert [q["query"] for q in picked] == ["question 0", "question 1", "question 2"]


def test_no_cap_keeps_everything():
    assert len(select_queries(_QUERIES, max_queries=None)) == 10


def test_blank_questions_are_dropped_before_the_cut():
    # Dropping them afterwards would silently shorten the set, so two methods with
    # different blank handling would answer different numbers of questions.
    noisy = [{"query": "  "}, *_QUERIES]
    assert len(select_queries(noisy, max_queries=3)) == 3


def test_every_method_gets_the_same_set_and_the_same_order():
    a = select_queries(_QUERIES, max_queries=5)
    b = select_queries(list(_QUERIES), max_queries=5)
    assert [q["query_id"] for q in a] == [q["query_id"] for q in b]


def test_the_id_follows_the_question_not_its_position():
    full = select_queries(_QUERIES, max_queries=10)
    short = select_queries(_QUERIES, max_queries=3)
    # Same question, same id, so two runs can be compared question by question even
    # when one of them answered fewer.
    assert full[1]["query_id"] == short[1]["query_id"]
    assert full[0]["query_id"] != full[1]["query_id"]


def test_the_id_ignores_surrounding_whitespace():
    assert query_id({"query": " what is dp? "}) == query_id({"query": "what is dp?"})


def test_the_original_fields_survive():
    picked = select_queries([{"query": "q", "answer": "a", "favorites": ["x"]}], max_queries=1)
    assert picked[0]["answer"] == "a"
    assert picked[0]["favorites"] == ["x"]


def test_tagging_can_be_skipped():
    picked = select_queries(_QUERIES, max_queries=2, tag_ids=False)
    assert "query_id" not in picked[0]


@pytest.mark.parametrize(
    "runner",
    [
        "run_dp_rag", "run_dpvoterag", "run_dpsparsevoterag", "run_dpksa",
        "run_private_rag", "run_dp_synrag", "run_vanilla",
    ],
)
def test_every_runner_uses_the_shared_selection(runner):
    source = (PROJECT_ROOT / "experiments" / f"{runner}.py").read_text(encoding="utf-8")
    assert "select_queries(" in source, (
        f"{runner} cuts its own question set, so it can drift from the others"
    )
    assert "resolve_seeds(" in source, (
        f"{runner} does not separate the data seed from the run seed, so a repetition "
        "would resample its corpus"
    )
