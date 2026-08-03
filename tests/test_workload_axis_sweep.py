"""Sweeping the workload size is what turns the accounting into a measurement.

At a fixed total budget, a method that pays per query has to spend less on each one as
the stream grows, while a method whose guarantee already covers the stream does not
move. Running both at the same target is what makes that difference a utility curve
rather than an argument about accounting, and these tests fix the property the curve
depends on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.workload_accounting import (
    DEFAULT_DELTA,
    METHOD_KNOB_BUILDERS,
    METHOD_REGIMES,
    budget_for_target,
    method_epsilon_total,
)

WORKLOADS = [1, 5, 25, 100]
TARGET = 10.0

STRUCTURAL = {
    "dp_rag": dict(retrieval_ratio=2.8458, max_tokens=50),
    "dpvoterag": dict(max_tokens=50, delta_token=1e-6),
    "dpsparsevoterag": dict(max_private_tokens=50, delta_token=1e-6, sparse_fraction=0.5),
    "private_rag": dict(m_retrievals=1),
    "private_rag_ada": dict(m_retrievals=1),
    "dp_synrag": dict(),
}


@pytest.mark.parametrize("method", sorted(STRUCTURAL))
def test_the_derived_budget_reaches_the_target_at_every_workload(method):
    for q in WORKLOADS:
        knob = budget_for_target(
            method, target_epsilon=TARGET, queries=q, delta=DEFAULT_DELTA,
            **STRUCTURAL[method]
        )
        mechanisms = METHOD_KNOB_BUILDERS[method](knob, **STRUCTURAL[method])
        from metrics.workload_accounting import epsilon_total

        achieved = epsilon_total(
            mechanisms, queries=q, delta=DEFAULT_DELTA,
            regime=METHOD_REGIMES.get(method, "per_query"),
        ).epsilon
        assert achieved == pytest.approx(TARGET, rel=0.02), (
            f"{method} at Q={q} composes to {achieved:.3f}"
        )


@pytest.mark.parametrize("method", ["dp_rag", "dpvoterag", "dpsparsevoterag"])
def test_a_per_query_method_must_spend_less_on_a_longer_stream(method):
    """The whole point of the axis: serving more questions costs each one more noise."""
    budgets = [
        budget_for_target(method, target_epsilon=TARGET, queries=q, **STRUCTURAL[method])
        for q in WORKLOADS
    ]
    assert budgets == sorted(budgets, reverse=True), budgets
    assert budgets[0] / budgets[-1] > 5, (
        f"{method} barely moves across the workload: {budgets}"
    )


@pytest.mark.parametrize("method", ["private_rag", "private_rag_ada", "dp_synrag"])
def test_a_stream_covering_guarantee_does_not_move(method):
    """A per-document filter and an offline build pay once, whatever Q is."""
    budgets = [
        budget_for_target(method, target_epsilon=TARGET, queries=q, **STRUCTURAL[method])
        for q in WORKLOADS
    ]
    assert all(b == pytest.approx(budgets[0], rel=1e-6) for b in budgets), budgets


def test_every_method_with_a_registry_entry_can_be_inverted():
    assert set(METHOD_KNOB_BUILDERS) - {"dpksa"} == set(STRUCTURAL)


def test_an_unreachable_target_raises_rather_than_clamping():
    """A clamped budget would put a point on the axis the mechanism never honoured."""
    with pytest.raises(ValueError):
        # The per-token delta alone overruns this target long before epsilon does.
        budget_for_target(
            "dpvoterag", target_epsilon=1e-4, queries=100,
            max_tokens=50, delta_token=1e-3,
        )
