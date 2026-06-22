"""Smoke tests for all bench_datasets — verifies loading and BaseDataset interface.

Without internet access, these tests will FAIL with ConnectionError (expected behavior).
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench_datasets.squad import SQuADDataset
from bench_datasets.chatdoctor import ChatDoctorDataset
from bench_datasets.natural_questions import NaturalQuestionsDataset
from bench_datasets.triviaqa import TriviaQADataset
from bench_datasets.hotpotqa import HotpotQADataset
from bench_datasets.medical_dirichlet_phi3 import MedicalDirichletPhi3Dataset

MAX_DOCS = 5
MAX_QUERIES = 3
SEED = 42


def _check_dataset(dataset_instance):
    """Verifies the BaseDataset interface on any concrete instance."""
    dataset_instance.load()

    docs = dataset_instance.get_documents()
    assert isinstance(docs, list), "get_documents() must return a list"
    assert len(docs) > 0, "get_documents() must not be empty"
    assert all(isinstance(d, str) for d in docs), "All documents must be str"

    queries = dataset_instance.get_queries()
    assert isinstance(queries, list), "get_queries() must return a list"
    assert len(queries) > 0, "get_queries() must not be empty"
    for q in queries:
        assert isinstance(q, dict), "Each query must be a dict"
        assert "query" in q, "Each query must have a 'query' key"
        assert "answer" in q, "Each query must have an 'answer' key"
        assert isinstance(q["query"], str), "'query' must be str"


def test_squad_loads():
    _check_dataset(SQuADDataset(
        max_index_docs=MAX_DOCS,
        max_test_queries=MAX_QUERIES,
        random_seed=SEED,
    ))


def test_chatdoctor_loads():
    _check_dataset(ChatDoctorDataset(
        max_index_docs=MAX_DOCS,
        max_test_queries=MAX_QUERIES,
        random_seed=SEED,
    ))


def test_natural_questions_loads():
    _check_dataset(NaturalQuestionsDataset(
        max_index_docs=MAX_DOCS,
        max_test_queries=MAX_QUERIES,
        random_seed=SEED,
    ))


def test_triviaqa_loads():
    _check_dataset(TriviaQADataset(
        max_index_docs=MAX_DOCS,
        max_test_queries=MAX_QUERIES,
        random_seed=SEED,
    ))


def test_hotpotqa_loads():
    _check_dataset(HotpotQADataset(
        max_index_docs=MAX_DOCS,
        max_test_queries=MAX_QUERIES,
        random_seed=SEED,
    ))


def test_medical_dirichlet_phi3_loads():
    _check_dataset(MedicalDirichletPhi3Dataset(
        max_index_docs=MAX_DOCS,
        max_test_queries=MAX_QUERIES,
        random_seed=SEED,
    ))
