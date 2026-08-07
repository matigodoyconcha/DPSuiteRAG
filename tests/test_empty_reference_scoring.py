"""Answering nothing must never score as answering everything.

QA normalization strips punctuation and the articles a/an/the, so a gold of ``"The"``
— or a gold that was empty to begin with — reduces to the empty string. Exact match
and token-F1 then read an empty prediction as a perfect hit, and the methods that
return an empty string are exactly the ones this benchmark is measuring at the tight
end of the privacy axis: the halted vote, the declined propose-test-release, the
exhausted budget. The bias points at the result.

These tests pin the fix (a pair with nothing on either side is not scored) and, just as
importantly, the normalization it must not disturb.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.utility import (  # noqa: E402
    _em_pair,
    _token_f1_pair,
    compute_utility_metrics,
    contains_accuracy,
    f1_score,
    has_scoreable_reference,
    levenshtein,
    match_accuracy,
    rouge_1,
    rouge_l,
    scoring_coverage,
)

# Every metric that compares a prediction to a reference item by item. The property
# below has to hold for all of them, not just the two that were caught.
PAIR_METRICS = [match_accuracy, contains_accuracy, f1_score, rouge_l, rouge_1, levenshtein]


# ---------------------------------------------------------------------------
# the defect
# ---------------------------------------------------------------------------

def test_an_empty_answer_does_not_exactly_match_an_article_only_gold():
    assert _em_pair("", "The") == 0.0
    assert match_accuracy([""], ["The"]) == 0.0


def test_an_empty_answer_does_not_share_tokens_with_an_article_only_gold():
    assert _token_f1_pair("", "The") == 0.0
    assert f1_score([""], ["The"]) == 0.0


def test_punctuation_only_is_not_an_exact_match_for_an_article():
    assert _em_pair("!!!", "a") == 0.0
    assert match_accuracy(["!!!"], ["a"]) == 0.0


def test_nothing_against_nothing_is_not_a_match():
    assert _em_pair("", "") == 0.0
    assert _token_f1_pair("", "") == 0.0


@pytest.mark.parametrize("metric", PAIR_METRICS)
def test_no_pair_metric_credits_a_cell_that_answered_nothing(metric):
    # The audited cell: two empty predictions, references ["", "The"]. It used to take
    # the top exact-match score of the run.
    assert metric(["", ""], ["", "The"]) == 0.0


def test_a_realistic_all_empty_cell_scores_zero():
    # What a halted mechanism actually produces: no answers, real golds.
    preds = ["", "", ""]
    refs = ["Paris", "Rome", "Berlin"]
    for metric in PAIR_METRICS:
        assert metric(preds, refs) == 0.0
    # And nothing was dropped — these are questions the method got wrong, not
    # questions that could not be asked.
    assert scoring_coverage(preds, refs) == {"n_scored": 3, "n_unscoreable": 0}


# ---------------------------------------------------------------------------
# only the doubly-degenerate case is excluded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", PAIR_METRICS)
def test_a_real_answer_against_an_empty_gold_still_scores_zero(metric):
    assert metric(["Paris"], [""]) == 0.0


@pytest.mark.parametrize("metric", PAIR_METRICS)
def test_an_empty_answer_against_a_real_gold_still_scores_zero(metric):
    assert metric([""], ["Paris"]) == 0.0


def test_a_wrong_answer_is_not_excused_by_the_skip():
    # Dropping items must not become a way to lose wrong answers: only the pair with
    # nothing on either side leaves the denominator.
    assert match_accuracy(["Paris", "Berlin"], ["Paris", "Rome"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# the normalization the fix must not disturb
# ---------------------------------------------------------------------------

def test_trailing_punctuation_still_matches():
    assert match_accuracy(["David Seville."], ["David Seville"]) == 1.0


def test_a_leading_article_still_matches():
    assert match_accuracy(["The Phantom"], ["Phantom"]) == 1.0


def test_an_alias_list_still_matches_any_of_its_forms():
    assert match_accuracy(["nyc"], [["New York City", "NYC"]]) == 1.0
    assert contains_accuracy(["nyc"], [["New York City", "NYC"]]) == 1.0
    assert f1_score(["nyc"], [["New York City", "NYC"]]) == pytest.approx(1.0)


def test_an_alias_list_with_one_usable_gold_is_scoreable():
    # ["", "Paris"] has content; the item is asked and answered.
    assert has_scoreable_reference(["", "Paris"]) is True
    assert match_accuracy(["Paris"], [["", "Paris"]]) == 1.0
    assert match_accuracy([""], [["", "Paris"]]) == 0.0


def test_an_alias_list_with_no_usable_gold_is_not_scoreable():
    assert has_scoreable_reference(["", "the", "!!"]) is False


# ---------------------------------------------------------------------------
# the dropped items are visible
# ---------------------------------------------------------------------------

def test_the_denominator_and_the_dropped_count_travel_with_the_metrics():
    out = compute_utility_metrics(
        ["Paris", "", ""], ["Paris", "The", ""], ["match_accuracy", "f1_score"]
    )
    assert out["n_scored"] == 1
    assert out["n_unscoreable"] == 2
    # A mean over one item, and the payload says so rather than implying three.
    assert out["match_accuracy"] == 1.0


def test_the_coverage_counts_are_reported_even_when_nothing_is_dropped():
    out = compute_utility_metrics(["Paris"], ["Paris"], ["match_accuracy"])
    assert out["n_scored"] == 1
    assert out["n_unscoreable"] == 0


def test_the_coverage_counts_follow_the_alias_sets_the_metrics_scored_against():
    # With the alias list the item has a usable gold, so it is scored, not dropped.
    out = compute_utility_metrics(
        [""], ["The"], ["match_accuracy"], alias_sets=[["The", "Paris"]]
    )
    assert out["n_scored"] == 1
    assert out["n_unscoreable"] == 0
    assert out["match_accuracy"] == 0.0


def test_every_pair_metric_shares_one_denominator():
    # Two metrics reported for the same cell must have averaged over the same items,
    # or the cell's numbers cannot be read against each other.
    preds = ["Paris", ""]
    refs = ["Paris", "a"]
    assert scoring_coverage(preds, refs)["n_scored"] == 1
    for metric in PAIR_METRICS:
        assert metric(preds, refs) == pytest.approx(metric(["Paris"], ["Paris"]))


def test_a_call_without_pair_metrics_reports_no_denominator():
    # answer_rate is over every query the method was given; it has no dropped items,
    # and publishing a coverage count next to it would describe a different measurement.
    out = compute_utility_metrics(["", ""], ["a", "b"], ["answer_rate"])
    assert "n_scored" not in out
    assert "n_unscoreable" not in out


def test_the_coverage_counts_can_be_requested_by_name():
    out = compute_utility_metrics(["", "x"], ["The", "x"], ["n_scored", "n_unscoreable"])
    assert out == {"n_scored": 1, "n_unscoreable": 1}


def test_the_dropped_count_reaches_the_consolidated_csv():
    from experiments.score_results import CSV_FIELDS

    assert "n_scored" in CSV_FIELDS
    assert "n_unscoreable" in CSV_FIELDS
