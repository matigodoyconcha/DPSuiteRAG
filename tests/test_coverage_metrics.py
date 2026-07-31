"""Tests for the metrics that separate a silent method from a wrong one."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.utility import (
    answer_rate,
    compute_utility_metrics,
    empty_rate,
    f1_score,
    retrieval_precision_at_k,
)


# ---------------------------------------------------------------------------
# answer_rate / empty_rate
# ---------------------------------------------------------------------------

def test_answer_rate_counts_non_empty_predictions():
    assert answer_rate(["Paris", "", "1945", "   "]) == pytest.approx(0.5)


def test_empty_rate_is_the_complement():
    preds = ["Paris", "", "1945", "   "]
    assert answer_rate(preds) + empty_rate(preds) == pytest.approx(1.0)


def test_rates_are_nan_without_predictions():
    assert math.isnan(answer_rate([]))
    assert math.isnan(empty_rate([]))


def test_a_silent_method_and_a_wrong_one_score_the_same_on_f1():
    # The reason the rate is needed: both of these are f1 = 0, and nothing in the
    # overlap metrics distinguishes the method that stopped answering from the one
    # that answered badly.
    refs = ["Paris", "Paris"]
    silent = f1_score(["", ""], refs)
    wrong = f1_score(["Berlin", "Berlin"], refs)
    assert silent == wrong == 0.0
    assert answer_rate(["", ""]) == 0.0
    assert answer_rate(["Berlin", "Berlin"]) == 1.0


def test_accuracy_among_answered_queries_is_recoverable():
    preds = ["Paris", "", "Paris", ""]
    refs = ["Paris", "Paris", "Paris", "Paris"]
    overall = f1_score(preds, refs)
    conditional = overall / answer_rate(preds)
    assert conditional == pytest.approx(1.0)


def test_answer_rate_is_computed_through_the_dispatcher():
    out = compute_utility_metrics(["a", ""], ["a", "b"], ["answer_rate", "empty_rate"])
    assert out["answer_rate"] == pytest.approx(0.5)
    assert out["empty_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# retrieval_precision_at_k
# ---------------------------------------------------------------------------

def test_perfect_retrieval_scores_one():
    assert retrieval_precision_at_k([[1, 2, 3]], [[1, 2, 3]], k=3) == pytest.approx(1.0)


def test_precision_is_the_share_of_the_non_private_top_k():
    assert retrieval_precision_at_k([[1, 2, 9, 8]], [[1, 2, 3, 4]], k=4) == pytest.approx(0.5)


def test_ids_compare_across_types():
    assert retrieval_precision_at_k([["1", "2"]], [[1, 2]], k=2) == pytest.approx(1.0)


def test_k_truncates_both_rankings():
    # Only the first two retrieved count, and only the first two are gold.
    assert retrieval_precision_at_k([[1, 2, 3, 4]], [[1, 2, 3, 4]], k=2) == pytest.approx(1.0)
    assert retrieval_precision_at_k([[3, 4]], [[1, 2, 3, 4]], k=2) == pytest.approx(0.0)


def test_an_exhausted_budget_scores_zero_not_missing():
    # A method whose per-document budget ran out returns nothing on purpose; that is
    # a total loss of precision, and the metric that hides it as a missing value is
    # exactly the one that makes budget exhaustion invisible.
    assert retrieval_precision_at_k([[]], [[1, 2, 3]], k=3) == pytest.approx(0.0)


def test_averages_over_queries():
    score = retrieval_precision_at_k([[1, 2], [9, 8]], [[1, 2], [1, 2]], k=2)
    assert score == pytest.approx(0.5)


def test_nan_without_a_gold_ranking():
    assert math.isnan(retrieval_precision_at_k([[1, 2]], []))
    assert math.isnan(retrieval_precision_at_k([], [[1, 2]]))


def test_precision_is_computed_through_the_dispatcher():
    out = compute_utility_metrics(
        ["a"], ["a"], ["retrieval_precision_at_k"],
        retrieved_ids=[[1, 2, 9, 8]], relevant_ids=[[1, 2, 3, 4]], precision_k=4,
    )
    assert out["retrieval_precision_at_k"] == pytest.approx(0.5)
