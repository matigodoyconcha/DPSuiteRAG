"""Unit tests for Private-RAG (MURAG / MURAG-ADA), arXiv:2511.07637."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dp_methods.per_query.private_rag.budget_manager import BudgetManager
from dp_methods.per_query.private_rag.retriever import (
    AdaptiveBudgetAwareRetriever,
    BudgetAwareRetriever,
)


class _FakeIndexer:
    """Minimal indexer exposing the attributes the retrievers read.

    ``embeddings_matrix`` is a (N, 1) column so that a query of ``[1.0]`` yields
    scores equal to the column entries — deterministic control for unit tests.
    """

    def __init__(self, scores: list[float]):
        self.embeddings_matrix = np.array([[s] for s in scores], dtype=np.float32)
        self.metadata = [{"doc_id": i, "text": f"d{i}"} for i in range(len(scores))]


_UNIT_QUERY = np.array([1.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# BudgetManager — individual pure-DP filter (E(z) = M * eps_q)
# ---------------------------------------------------------------------------

def test_budget_manager_starts_full() -> None:
    bm = BudgetManager(initial_budget=5.0)
    assert bm.remaining("doc0") == 5.0        # unseen -> full budget
    assert bm.can_query("doc0", cost=5.0)     # exactly enough
    assert not bm.can_query("doc0", cost=5.1)  # not quite


def test_budget_manager_consume_and_exhaust() -> None:
    bm = BudgetManager(initial_budget=6.0)
    bm.consume("z", 2.0)
    assert bm.remaining("z") == pytest.approx(4.0)
    assert bm.can_query("z", 4.0)
    bm.consume("z", 2.0)
    bm.consume("z", 2.0)
    assert bm.remaining("z") == pytest.approx(0.0)
    assert not bm.can_query("z", 2.0)  # exhausted -> dropped from active set


def test_budget_manager_floors_at_zero() -> None:
    bm = BudgetManager(initial_budget=1.0)
    bm.consume("z", 5.0)  # would go negative; floored
    assert bm.remaining("z") == 0.0


def test_budget_manager_per_document_independence() -> None:
    bm = BudgetManager(initial_budget=10.0)
    bm.consume("a", 10.0)          # exhaust a
    assert not bm.can_query("a", 1.0)
    assert bm.can_query("b", 10.0)  # b untouched


def test_budget_manager_active_ids_filters_by_cost() -> None:
    bm = BudgetManager(initial_budget=4.0)
    bm.consume("a", 4.0)   # exhausted
    bm.consume("b", 2.0)   # 2 left
    ids = ["a", "b", "c"]  # c unseen (full)
    assert bm.active_ids(cost=2.0, candidate_ids=ids) == ["b", "c"]
    assert bm.active_ids(cost=4.0, candidate_ids=ids) == ["c"]


def test_budget_manager_reset() -> None:
    bm = BudgetManager(initial_budget=3.0)
    bm.consume("a", 3.0)
    bm.reset()
    assert bm.remaining("a") == 3.0
    assert bm.can_query("a", 3.0)


def test_budget_manager_stats() -> None:
    bm = BudgetManager(initial_budget=5.0)
    bm.consume_many(["a", "b"], 5.0)  # both exhausted
    bm.consume("c", 1.0)
    s = bm.stats(cost=5.0)
    assert s["n_tracked"] == 3
    assert s["n_exhausted"] == 3  # a, b exhausted; c cannot pay 5 (has 4)
    assert s["total_consumed"] == pytest.approx(11.0)


def test_budget_manager_rejects_nonpositive_budget() -> None:
    with pytest.raises(ValueError):
        BudgetManager(initial_budget=0.0)


# ---------------------------------------------------------------------------
# BudgetAwareRetriever (MURAG) — fixed public threshold
# ---------------------------------------------------------------------------

def test_murag_returns_top_k_above_threshold() -> None:
    idx = _FakeIndexer([0.9, 0.8, 0.5, 0.1, -0.2])
    bm = BudgetManager(initial_budget=10.0)
    r = BudgetAwareRetriever(indexer=idx, budget_manager=bm, epsilon_q=2.0, threshold=0.3)
    out = r.retrieve(_UNIT_QUERY, k=2)
    assert [d["text"] for d in out] == ["d0", "d1"]          # top-2 by score
    assert all(out[i]["score"] >= out[i + 1]["score"] for i in range(len(out) - 1))


def test_murag_charges_all_above_threshold_not_just_top_k() -> None:
    # Docs 0,1,2 are above tau=0.3; k=2 returns only 0,1 but ALL THREE are charged
    # (paper: charge D_qt before Top-K). Docs 3,4 (below tau) are untouched.
    idx = _FakeIndexer([0.9, 0.8, 0.5, 0.1, -0.2])
    bm = BudgetManager(initial_budget=10.0)
    r = BudgetAwareRetriever(indexer=idx, budget_manager=bm, epsilon_q=2.0, threshold=0.3)
    r.retrieve(_UNIT_QUERY, k=2)
    assert bm.remaining(0) == pytest.approx(8.0)
    assert bm.remaining(1) == pytest.approx(8.0)
    assert bm.remaining(2) == pytest.approx(8.0)  # charged even though not returned
    assert bm.remaining(3) == 10.0                # below threshold: free
    assert bm.remaining(4) == 10.0
    assert r.last_run_stats["n_charged"] == 3
    assert r.last_run_stats["n_returned"] == 2


def test_murag_depletes_budget_and_drops_documents() -> None:
    idx = _FakeIndexer([0.9, 0.8, 0.5, 0.1, -0.2])
    bm = BudgetManager(initial_budget=6.0)  # allows 3 participations at eps_q=2
    r = BudgetAwareRetriever(indexer=idx, budget_manager=bm, epsilon_q=2.0, threshold=0.3)
    for _ in range(3):
        assert r.retrieve(_UNIT_QUERY, k=2)      # non-empty while budget lasts
    # Docs 0,1,2 now exhausted; nothing above threshold remains active.
    assert r.retrieve(_UNIT_QUERY, k=2) == []


def test_murag_is_deterministic() -> None:
    idx = _FakeIndexer([0.9, 0.8, 0.5, 0.1])
    out_a = BudgetAwareRetriever(
        indexer=idx, budget_manager=BudgetManager(10.0), epsilon_q=1.0, threshold=0.3
    ).retrieve(_UNIT_QUERY, k=2)
    out_b = BudgetAwareRetriever(
        indexer=idx, budget_manager=BudgetManager(10.0), epsilon_q=1.0, threshold=0.3
    ).retrieve(_UNIT_QUERY, k=2)
    assert [d["text"] for d in out_a] == [d["text"] for d in out_b]


def test_murag_empty_when_all_below_threshold() -> None:
    idx = _FakeIndexer([0.1, 0.05, -0.2])
    bm = BudgetManager(initial_budget=10.0)
    r = BudgetAwareRetriever(indexer=idx, budget_manager=bm, epsilon_q=1.0, threshold=0.5)
    assert r.retrieve(_UNIT_QUERY, k=3) == []
    assert bm.stats()["total_consumed"] == 0.0  # nothing charged


# ---------------------------------------------------------------------------
# AdaptiveBudgetAwareRetriever (MURAG-ADA) — noisy prefix-sum threshold
# ---------------------------------------------------------------------------

# Scores interior to 0.1-wide bins (avoids float bin-boundary ambiguity):
#   bin 0 [>0.9]: d0(0.95), d1(0.93)   bin 1 (0.8,0.9]: d2(0.85)
#   bin 4: d3(0.55)   bin 5: d4(0.45)   bin 8: d5(0.15)
_ADA_SCORES = [0.95, 0.93, 0.85, 0.55, 0.45, 0.15]


def _ada_retriever(bm: BudgetManager, **kw) -> AdaptiveBudgetAwareRetriever:
    # Large eps_thr -> Laplace scale 1/eps_thr is tiny -> prefix sum ~ true counts.
    params = dict(
        indexer=_FakeIndexer(_ADA_SCORES),
        budget_manager=bm,
        epsilon_q=200.0,
        epsilon_thr=100.0,
        score_min=0.0,
        score_max=1.0,
        bin_size=0.1,
    )
    params.update(kw)
    return AdaptiveBudgetAwareRetriever(**params)


def test_murag_ada_stops_when_prefix_sum_reaches_k() -> None:
    bm = BudgetManager(initial_budget=1000.0)
    r = _ada_retriever(bm)
    out = r.retrieve(_UNIT_QUERY, k=2)               # bin 0 alone has 2 docs -> stop
    assert [d["text"] for d in out] == ["d0", "d1"]
    assert r.last_run_stats["n_bins_processed"] == 1
    assert r.last_run_stats["tau_t"] == pytest.approx(0.9)  # lower edge of bin 0


def test_murag_ada_processes_more_bins_for_larger_k() -> None:
    bm = BudgetManager(initial_budget=1000.0)
    r = _ada_retriever(bm)
    out = r.retrieve(_UNIT_QUERY, k=3)               # bin0(2)+bin1(1)=3 -> stop after bin1
    assert [d["text"] for d in out] == ["d0", "d1", "d2"]
    assert r.last_run_stats["n_bins_processed"] == 2
    assert r.last_run_stats["tau_t"] == pytest.approx(0.8)


def test_murag_ada_charges_eps_thr_plus_eps_rag() -> None:
    # A retrieved doc pays eps_thr (its processed bin) + eps_RAG (active set) = eps_q.
    # A doc in an unprocessed (lower) bin pays nothing.
    bm = BudgetManager(initial_budget=1000.0)
    r = _ada_retriever(bm)
    r.retrieve(_UNIT_QUERY, k=2)                      # processes bin 0 only
    assert bm.remaining(0) == pytest.approx(1000.0 - 200.0)  # eps_thr+eps_RAG
    assert bm.remaining(1) == pytest.approx(1000.0 - 200.0)
    assert bm.remaining(2) == 1000.0                 # bin 1 never processed


def test_murag_ada_is_reproducible() -> None:
    a = _ada_retriever(BudgetManager(1000.0)).retrieve(_UNIT_QUERY, k=2)
    b = _ada_retriever(BudgetManager(1000.0)).retrieve(_UNIT_QUERY, k=2)
    assert [d["text"] for d in a] == [d["text"] for d in b]


def test_murag_ada_rejects_eps_thr_ge_eps_q() -> None:
    with pytest.raises(ValueError):
        AdaptiveBudgetAwareRetriever(
            indexer=_FakeIndexer(_ADA_SCORES),
            budget_manager=BudgetManager(10.0),
            epsilon_q=1.0,
            epsilon_thr=1.0,  # not < eps_q
        )
