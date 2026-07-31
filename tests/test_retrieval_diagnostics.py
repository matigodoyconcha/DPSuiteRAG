"""The gold ranking and the stream prefix curve that the scaling figure reads."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import nonprivate_ranking


class _MatrixIndex:
    def __init__(self, matrix):
        self.embeddings_matrix = matrix
        self.metadata = [{"doc_id": i, "chunk_id": 0} for i in range(len(matrix))]
        self.index = None


class _FaissIndex:
    """Mimics the FAISS path: search returns (scores, indices), -1 padding included."""

    def __init__(self, matrix):
        self.embeddings_matrix = None
        self.metadata = [{"doc_id": i, "chunk_id": 0} for i in range(len(matrix))]
        self._matrix = matrix
        self.index = self

    def search(self, query, k):
        scores = self._matrix @ query[0]
        order = np.argsort(-scores)[:k]
        padded = np.full((1, k), -1, dtype=np.int64)
        padded[0, : len(order)] = order
        return np.zeros((1, k), dtype=np.float32), padded


@pytest.fixture
def corpus():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(200, 16)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    query = rng.normal(size=16).astype(np.float32)
    return matrix, query


def test_the_ranking_is_the_true_top_k(corpus):
    matrix, query = corpus
    expected = list(np.argsort(-(matrix @ (query / np.linalg.norm(query))))[:10])
    assert nonprivate_ranking(_MatrixIndex(matrix), query, 10) == [int(i) for i in expected]


def test_the_ranking_is_ordered_best_first(corpus):
    matrix, query = corpus
    ids = nonprivate_ranking(_MatrixIndex(matrix), query, 10)
    scores = matrix @ (query / np.linalg.norm(query))
    assert list(scores[ids]) == sorted(scores[ids], reverse=True)


def test_both_index_layouts_agree(corpus):
    # Some methods keep the embedding matrix and score directly; others go through
    # FAISS. The gold must not depend on which one the run happened to build.
    matrix, query = corpus
    assert nonprivate_ranking(_MatrixIndex(matrix), query, 10) == nonprivate_ranking(
        _FaissIndex(matrix), query, 10
    )


def test_an_unnormalised_query_ranks_the_same(corpus):
    # Cosine order is invariant to the query's magnitude; a caller passing a raw
    # embedding must not get a different gold than one passing a normalised vector.
    matrix, query = corpus
    assert nonprivate_ranking(_MatrixIndex(matrix), query, 10) == nonprivate_ranking(
        _MatrixIndex(matrix), query * 17.0, 10
    )


def test_asking_for_more_than_the_corpus_holds(corpus):
    matrix, query = corpus
    assert len(nonprivate_ranking(_MatrixIndex(matrix), query, 10_000)) == len(matrix)


def test_faiss_padding_is_dropped(corpus):
    matrix, query = corpus
    ids = nonprivate_ranking(_FaissIndex(matrix), query, 10_000)
    assert -1 not in ids
    assert len(ids) == len(matrix)


def test_an_index_without_either_layout_yields_no_gold():
    class _Empty:
        embeddings_matrix = None
        metadata = []
        index = None

    # A run missing the diagnostic still has to produce its utility numbers.
    assert nonprivate_ranking(_Empty(), np.zeros(4, dtype=np.float32), 10) == []


def test_ids_are_documents_not_index_positions():
    # The privacy unit is the document, so the gold has to speak in doc_ids; with
    # chunking on, several index rows share one.
    matrix = np.eye(4, dtype=np.float32)
    index = _MatrixIndex(matrix)
    index.metadata = [{"doc_id": 7}, {"doc_id": 7}, {"doc_id": 9}, {"doc_id": 9}]
    ids = nonprivate_ranking(index, np.array([1, 0, 0, 0], dtype=np.float32), 2)
    assert ids[0] == 7


def test_the_prefix_curve_is_recorded_per_cell():
    """MuRAG's scaling curve must not carry the previous grid cell's stream.

    The budget is a per-cell object, so a curve that accumulated across cells would
    show documents already exhausted at query 1 of every cell after the first.
    """
    import ast

    source = (PROJECT_ROOT / "experiments" / "run_private_rag.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    inits, appends = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "prefix_curve":
            inits.append(node.lineno)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and getattr(node.func.value, "id", "") == "prefix_curve"
        ):
            appends.append(node.lineno)
    assert inits and appends
    # The reset has to come before the appends that fill it, in the same cell body.
    assert min(inits) < min(appends)
