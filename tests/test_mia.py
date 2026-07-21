"""Tests for the real membership inference attacks (S2MIA + Anderson Yes/No).

Covers the offline scorers (metrics/mia.py), the online probe driver
(experiments/mia_probe.py) and the member/non-member split (BaseDataset generic
default + the ChatDoctor QA-native override).
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.mia import (  # noqa: E402
    bleu_similarity,
    max_bleu_against_corpus,
    roc_auc,
    tpr_at_fpr,
    yes_no_score,
    score_mia_records,
)
from experiments.mia_probe import (  # noqa: E402
    anderson_probe_prompt,
    resolve_mia_params,
    run_mia_probes,
)
from bench_datasets.base_dataset import BaseDataset  # noqa: E402
from bench_datasets.chatdoctor import ChatDoctorDataset  # noqa: E402


# --------------------------------------------------------------------------- #
# roc_auc / tpr_at_fpr
# --------------------------------------------------------------------------- #

def test_roc_auc_perfect_separation():
    assert roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0


def test_roc_auc_reversed_is_zero():
    assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0


def test_roc_auc_all_ties_is_half():
    # All-equal scores (e.g. degenerate empty outputs) settle at 0.5, not an
    # index-order artifact.
    assert roc_auc([1, 1, 0, 0], [0.0, 0.0, 0.0, 0.0]) == 0.5


def test_roc_auc_single_class_is_half():
    assert roc_auc([1, 1, 1], [0.9, 0.5, 0.1]) == 0.5


def test_tpr_at_fpr_perfect_separation():
    # Members strictly above non-members -> TPR 1.0 at FPR 0.
    assert tpr_at_fpr([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], target_fpr=0.0) == 1.0


def test_tpr_at_fpr_no_separation():
    assert tpr_at_fpr([1, 0], [0.5, 0.5], target_fpr=0.0) == 0.0


# --------------------------------------------------------------------------- #
# bleu / yes-no signals
# --------------------------------------------------------------------------- #

def test_bleu_identical_is_high():
    assert bleu_similarity("paris and lyon are cities", "paris and lyon are cities") > 0.9


def test_bleu_empty_is_zero():
    assert bleu_similarity("", "something") == 0.0
    assert bleu_similarity("something", "") == 0.0


def test_max_bleu_against_corpus_picks_best_match():
    corpus = [["totally", "unrelated"], ["paris", "and", "lyon", "are", "cities"]]
    assert max_bleu_against_corpus("paris and lyon are cities", corpus) > 0.9
    assert max_bleu_against_corpus("nothing matches here at all", corpus) < 0.3


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Yes", 1.0),
        ("yes it does appear", 1.0),
        ("No", 0.0),
        ("no it is not present", 0.0),
        ("I am not sure", 0.0),   # no yes/no token -> non-member
        ("", 0.0),
        ("No, but yes", 0.0),     # first token decides
    ],
)
def test_yes_no_score(text, expected):
    assert yes_no_score(text) == expected


# --------------------------------------------------------------------------- #
# score_mia_records
# --------------------------------------------------------------------------- #

def _leaky_records():
    """Members reproduce their gold and say Yes; non-members do neither."""
    members = [
        {"is_member": True, "gold": f"answer number {i} exactly",
         "s2mia_answer": f"answer number {i} exactly",
         "anderson_answer": "Yes", "doc": f"doc {i} answer number {i} exactly"}
        for i in range(5)
    ]
    non_members = [
        {"is_member": False, "gold": f"held answer {i}",
         "s2mia_answer": "unrelated generated text here",
         "anderson_answer": "No", "doc": f"held doc {i} held answer {i}"}
        for i in range(5)
    ]
    return members + non_members


def test_score_mia_records_leaky_is_high_auc():
    m = score_mia_records(_leaky_records())
    assert m["s2mia_auc"] == 1.0
    assert m["rag_mia_auc"] == 1.0
    assert m["s2mia_tpr_at_fpr01"] == 1.0


def test_score_mia_records_no_signal_is_half():
    # Identical behaviour for members and non-members -> no leakage.
    recs = [
        {"is_member": i < 5, "gold": "same gold", "s2mia_answer": "same answer",
         "anderson_answer": "No", "doc": "same doc"}
        for i in range(10)
    ]
    m = score_mia_records(recs)
    assert m["s2mia_auc"] == 0.5
    assert m["rag_mia_auc"] == 0.5


def test_score_mia_records_empty():
    assert score_mia_records([]) == {}


def test_score_mia_records_synth_retention():
    # Member input docs appear verbatim in the generated corpus; non-members do not.
    recs = _leaky_records()
    synth = [r["doc"] for r in recs if r["is_member"]] + ["some other generated text"]
    m = score_mia_records(recs, synthetic_texts=synth)
    assert "synth_retention_auc" in m
    assert m["synth_retention_auc"] == 1.0


def test_score_mia_records_no_synth_key_when_absent():
    m = score_mia_records(_leaky_records())
    assert "synth_retention_auc" not in m


# --------------------------------------------------------------------------- #
# run_mia_probes / prompt / params
# --------------------------------------------------------------------------- #

def test_run_mia_probes_shapes_and_labels():
    split = {
        "index_documents": ["a b c d", "e f g h"],
        "members": [{"query": "q1", "gold": "g1", "doc": "d1"}],
        "non_members": [{"query": "q2", "gold": "g2", "doc": "d2"}],
    }
    calls = []

    def answer_fn(q):
        calls.append(q)
        return "Yes" if q.startswith("Does this") else "some answer"

    records = run_mia_probes(answer_fn, split)
    assert len(records) == 2               # one member + one non-member
    assert len(calls) == 4                 # 2 probes (s2mia + anderson) per target
    assert [r["is_member"] for r in records] == [True, False]
    assert set(records[0]) == {"is_member", "query", "gold", "doc",
                               "s2mia_answer", "anderson_answer"}


def test_anderson_prompt_truncates_and_asks_yes_no():
    prompt = anderson_probe_prompt("word " * 100, maxlen=30)
    assert "Yes or No" in prompt
    assert len(prompt) < 120  # doc part truncated to ~30 chars


def test_resolve_mia_params_defaults_and_overrides():
    assert resolve_mia_params({}) == {
        "enabled": True, "n_per_class": 100, "index_size": 2000,
        "seed": 42, "anderson_doc_maxlen": 512,
    }
    p = resolve_mia_params({"mia": {"n_per_class": 25, "index_size": None, "enabled": False}})
    assert p["n_per_class"] == 25
    assert p["index_size"] is None
    assert p["enabled"] is False


# --------------------------------------------------------------------------- #
# get_mia_split — generic default and invariants
# --------------------------------------------------------------------------- #

class _StubDataset(BaseDataset):
    def __init__(self, docs):
        self._docs = docs

    def load(self):
        pass

    def get_documents(self):
        return self._docs

    def get_queries(self):
        return []


def test_generic_split_invariants():
    docs = [f"document number {i} has alpha beta gamma delta words {i}" for i in range(40)]
    split = _StubDataset(docs).get_mia_split(n_per_class=8, index_size=None, seed=1)
    idx = set(split["index_documents"])
    assert len(split["members"]) == 8
    assert len(split["non_members"]) == 8
    # The load-bearing invariant of a real MIA:
    assert all(m["doc"] in idx for m in split["members"])
    assert all(nm["doc"] not in idx for nm in split["non_members"])
    # Generic split: query is a prefix, gold is the (non-empty) continuation.
    for probe in split["members"]:
        assert probe["gold"]
        assert probe["query"]


def test_generic_split_too_few_docs_returns_none():
    assert _StubDataset(["only one doc here"]).get_mia_split(n_per_class=5) is None


def test_generic_split_index_size_cap():
    docs = [f"doc {i} alpha beta gamma delta {i}" for i in range(100)]
    split = _StubDataset(docs).get_mia_split(n_per_class=5, index_size=20, seed=0)
    # 5 held out as non-members, the member pool capped at index_size.
    assert len(split["index_documents"]) <= 20


# --------------------------------------------------------------------------- #
# ChatDoctor QA-native override (no network: inject the loaded rows directly)
# --------------------------------------------------------------------------- #

def test_chatdoctor_split_qa_native():
    ds = ChatDoctorDataset()
    ds._documents = [f"Q: patient question {i}?\nA: doctor advice number {i} here." for i in range(10)]
    ds._queries = [{"query": f"held question {i}?", "answer": f"held advice {i}"} for i in range(10)]

    split = ds.get_mia_split(n_per_class=3, seed=1)
    idx = set(split["index_documents"])
    assert split["index_documents"] == ds._documents
    assert all(m["doc"] in idx for m in split["members"])
    assert all(nm["doc"] not in idx for nm in split["non_members"])
    # Query = patient question, gold = doctor answer.
    probe = split["members"][0]
    assert probe["query"].endswith("?")
    assert probe["gold"].startswith("doctor advice")


def test_chatdoctor_split_qa_doc_parsing():
    q, a = ChatDoctorDataset._split_qa_doc("Q: what is this?\nA: it is a test.")
    assert q == "what is this?"
    assert a == "it is a test."
