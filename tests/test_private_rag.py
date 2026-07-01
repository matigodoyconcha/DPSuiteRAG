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
