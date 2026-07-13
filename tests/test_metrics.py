"""Smoke tests for utility metrics, privacy metrics and privacy accounting."""
import sys
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.utility import (
    match_accuracy,
    contains_accuracy,
    f1_score,
    rouge_l,
    rouge_1,
    levenshtein,
    hit_rate,
    compute_utility_metrics,
    bert_score_f1,
    recall_at_k,
    answer_faithfulness,
    answer_relevancy,
)
from metrics.privacy import (
    mia_auc_bleu,
    compute_privacy_metrics,
    bleu_membership_score,
    privacy_budget_consumption,
)
from metrics.privacy_accounting import pld_dp_rag_total


# ---------------------------------------------------------------------------
# match_accuracy
# ---------------------------------------------------------------------------

def test_match_accuracy_perfect():
    assert match_accuracy(["hello"], ["hello"]) == 1.0


def test_match_accuracy_zero():
    assert match_accuracy(["hello"], ["goodbye"]) == 0.0


def test_match_accuracy_case_insensitive():
    assert match_accuracy(["Hello World"], ["hello world"]) == 1.0


def test_match_accuracy_empty_predictions():
    assert match_accuracy([], ["ref"]) == 0.0


def test_match_accuracy_partial():
    score = match_accuracy(["right", "wrong"], ["right", "right"])
    assert score == 0.5


# ---------------------------------------------------------------------------
# rouge_l
# ---------------------------------------------------------------------------

def test_rouge_l_identical():
    score = rouge_l(["the cat eats fish"], ["the cat eats fish"])
    assert score == pytest.approx(1.0, abs=1e-4)


def test_rouge_l_empty_prediction():
    score = rouge_l([], ["reference"])
    assert score == 0.0


def test_rouge_l_returns_float_in_range():
    score = rouge_l(["something different here"], ["distinct reference"])
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# rouge_1 / levenshtein (DP-KSA paper metrics)
# ---------------------------------------------------------------------------

def test_rouge_1_identical():
    assert rouge_1(["the cat eats fish"], ["the cat eats fish"]) == pytest.approx(1.0, abs=1e-4)


def test_rouge_1_empty_prediction():
    assert rouge_1([], ["reference"]) == 0.0


def test_levenshtein_identical():
    assert levenshtein(["paris"], ["paris"]) == pytest.approx(1.0)


def test_levenshtein_disjoint():
    assert levenshtein(["abc"], ["xyz"]) == pytest.approx(0.0)


def test_levenshtein_one_edit():
    # one substitution over length 5 -> similarity 0.8
    assert levenshtein(["paris"], ["paras"]) == pytest.approx(0.8)


def test_levenshtein_both_empty():
    assert levenshtein([""], [""]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# hit_rate (set-valued / top-k gold)
# ---------------------------------------------------------------------------

def test_hit_rate_hit():
    assert hit_rate(["I recommend Toy Story"], [["Toy Story", "Aladdin"]]) == 1.0


def test_hit_rate_miss():
    assert hit_rate(["I recommend Heat"], [["Toy Story", "Aladdin"]]) == 0.0


def test_hit_rate_partial():
    score = hit_rate(["Toy Story", "nope"], [["Toy Story"], ["Aladdin"]])
    assert score == pytest.approx(0.5)


def test_hit_rate_empty_sets():
    assert hit_rate(["anything"], []) == 0.0


def test_hit_rate_dispatcher():
    result = compute_utility_metrics(
        ["Toy Story"], ["x"], ["hit_rate"], reference_sets=[["Toy Story"]]
    )
    assert result["hit_rate"] == 1.0


# ---------------------------------------------------------------------------
# compute_utility_metrics (dispatcher)
# ---------------------------------------------------------------------------

def test_compute_utility_metrics_runs():
    result = compute_utility_metrics(
        ["answer"], ["answer"], ["match_accuracy", "rouge_l"]
    )
    assert "match_accuracy" in result
    assert "rouge_l" in result


def test_compute_utility_metrics_unsupported_raises():
    with pytest.raises(ValueError):
        compute_utility_metrics(["pred"], ["ref"], ["bleu_inexistente"])


# ---------------------------------------------------------------------------
# mia_auc_bleu
# ---------------------------------------------------------------------------

def test_mia_auc_bleu_returns_float_in_range():
    preds = ["the dog runs", "the house is big", "it rains in the city"]
    refs  = ["the dog runs", "the house is small", "it is sunny"]
    auc = mia_auc_bleu(preds, refs)
    assert 0.0 <= auc <= 1.0


def test_mia_auc_bleu_empty():
    assert mia_auc_bleu([], []) == 0.5


# ---------------------------------------------------------------------------
# compute_privacy_metrics (dispatcher)
# ---------------------------------------------------------------------------

def test_compute_privacy_metrics_runs():
    result = compute_privacy_metrics(
        ["pred"], ["ref"],
        ["mia_auc_bleu"],
    )
    assert "mia_auc_bleu" in result


# ---------------------------------------------------------------------------
# privacy_accounting
# ---------------------------------------------------------------------------

def test_pld_dp_rag_total_finite_and_tighter_than_union_bound():
    eps = pld_dp_rag_total(
        eps_retrieval=0.5,
        token_epsilon=0.05,
        num_tokens=70,
        delta=1e-3,
    )
    union_bound = 0.5 + 70 * 0.05
    assert math.isfinite(eps)
    # PLD composition is tighter than the naive union bound but still spends
    # more than retrieval alone.
    assert 0.5 < eps < union_bound


def test_pld_dp_rag_total_degenerate_inputs():
    # No generation tokens -> only the retrieval budget remains.
    assert pld_dp_rag_total(0.5, 0.05, 0, 1e-3) == pytest.approx(0.5)
    assert pld_dp_rag_total(0.5, 0.0, 70, 1e-3) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# bert_score_f1
# ---------------------------------------------------------------------------

def test_bert_score_f1_identical():
    score = bert_score_f1(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert score == pytest.approx(1.0, abs=0.05)


def test_bert_score_f1_empty():
    assert bert_score_f1([], []) == 0.0


def test_bert_score_f1_in_range():
    score = bert_score_f1(["hello world"], ["goodbye planet"])
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

def test_recall_at_k_perfect():
    assert recall_at_k([[1, 2, 3]], [[1, 2]]) == pytest.approx(1.0)


def test_recall_at_k_zero():
    assert recall_at_k([[4, 5, 6]], [[1, 2]]) == pytest.approx(0.0)


def test_recall_at_k_partial():
    score = recall_at_k([[1, 2, 3]], [[1, 4]])
    assert score == pytest.approx(0.5)


def test_recall_at_k_empty():
    assert recall_at_k([], []) == 0.0


def test_recall_at_k_skips_empty_relevant():
    # Queries with no relevant docs are skipped, not penalised.
    score = recall_at_k([[1, 2], [3]], [[], [3]])
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# answer_faithfulness
# ---------------------------------------------------------------------------

def test_answer_faithfulness_high_when_answer_in_context():
    score = answer_faithfulness(
        ["the sky is blue"],
        [["the sky is blue and very clear"]],
    )
    assert score > 0.5


def test_answer_faithfulness_low_when_no_overlap():
    score = answer_faithfulness(
        ["quantum entanglement"],
        [["the weather is nice today"]],
    )
    assert score < 0.5


def test_answer_faithfulness_empty_predictions():
    assert answer_faithfulness([], []) == 0.0


def test_answer_faithfulness_dispatcher():
    result = compute_utility_metrics(
        ["sky is blue"],
        ["reference"],
        ["answer_faithfulness"],
        contexts=[["the sky is blue and clear"]],
    )
    assert "answer_faithfulness" in result
    assert 0.0 <= result["answer_faithfulness"] <= 1.0


# ---------------------------------------------------------------------------
# answer_relevancy
# ---------------------------------------------------------------------------

def test_answer_relevancy_high_for_matching():
    score = answer_relevancy(
        ["What is the capital of France?"],
        ["Paris is the capital of France."],
    )
    assert score > 0.5


def test_answer_relevancy_in_range():
    score = answer_relevancy(["Who invented the telephone?"], ["Alexander Graham Bell"])
    assert 0.0 <= score <= 1.0


def test_answer_relevancy_empty():
    assert answer_relevancy([], []) == 0.0


def test_answer_relevancy_dispatcher():
    result = compute_utility_metrics(
        ["Paris is the capital."],
        ["reference"],
        ["answer_relevancy"],
        queries=["What is the capital of France?"],
    )
    assert "answer_relevancy" in result
    assert 0.0 <= result["answer_relevancy"] <= 1.0


# ---------------------------------------------------------------------------
# bleu_membership_score
# ---------------------------------------------------------------------------

def test_bleu_membership_score_high_when_match():
    score = bleu_membership_score(
        ["the quick brown fox jumps"],
        ["the quick brown fox jumps over the lazy dog"],
    )
    assert score > 0.3


def test_bleu_membership_score_zero_empty_corpus():
    assert bleu_membership_score(["some text"], []) == 0.0


def test_bleu_membership_score_zero_empty_predictions():
    assert bleu_membership_score([], ["corpus doc"]) == 0.0


def test_bleu_membership_score_in_range():
    score = bleu_membership_score(
        ["random prediction here"],
        ["completely different content", "another document"],
    )
    assert 0.0 <= score <= 1.0


def test_bleu_membership_score_dispatcher():
    result = compute_privacy_metrics(
        ["test prediction"],
        ["reference"],
        ["bleu_membership_score"],
        corpus_docs=["test prediction corpus"],
    )
    assert "bleu_membership_score" in result
    assert 0.0 <= result["bleu_membership_score"] <= 1.0


# ---------------------------------------------------------------------------
# privacy_budget_consumption
# ---------------------------------------------------------------------------

def test_privacy_budget_consumption_average():
    dp_spent = [{"epsilon": 2.0}, {"epsilon": 4.0}]
    assert privacy_budget_consumption(dp_spent) == pytest.approx(3.0)


def test_privacy_budget_consumption_empty():
    assert privacy_budget_consumption([]) == 0.0


def test_privacy_budget_consumption_missing_key():
    dp_spent = [{"epsilon": 1.0}, {}]
    assert privacy_budget_consumption(dp_spent) == pytest.approx(0.5)


def test_privacy_budget_consumption_dispatcher():
    result = compute_privacy_metrics(
        ["pred"],
        ["ref"],
        ["privacy_budget_consumption"],
        dp_spent_per_query=[{"epsilon": 2.0}, {"epsilon": 4.0}],
    )
    assert result["privacy_budget_consumption"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# zCDP accounting (DP-SynRAG)
# ---------------------------------------------------------------------------

from metrics.privacy_accounting_zcdp import zcdp_dp_synrag, ZCDPCompositionResult


def test_zcdp_dp_synrag_formula_components():
    """Verify the per-component rho values match the paper Theorem 1.

    Paper formula:
        rho = K/(2 sigma_h^2) + L * [epsilon_theta_s^2/8 + 1/(2 sigma_mu^2) + T/2*(c/tau)^2]
    """
    import math

    K, sigma_h = 10, 1.0
    L = 2
    epsilon_theta_s = 2.0  # Non-unit value to catch epsilon^2/8 vs 1/(8*epsilon^2) bug
    sigma_mu = 0.5
    T, c, tau = 70, 1.0, 0.5
    delta = 1e-3

    r = zcdp_dp_synrag(
        K=K, sigma_h=sigma_h, L=L, epsilon_theta_s=epsilon_theta_s,
        sigma_mu=sigma_mu, T=T, c=c, tau=tau, delta=delta,
    )

    expected_rho_h = K / (2 * sigma_h ** 2)
    expected_rho_theta = (epsilon_theta_s ** 2) / 8.0  # epsilon^2 / 8, NOT 1/(8*epsilon^2)
    expected_rho_mu = 1.0 / (2.0 * sigma_mu ** 2)
    expected_rho_token = (T / 2.0) * (c / tau) ** 2
    expected_rho = expected_rho_h + L * (expected_rho_theta + expected_rho_mu + expected_rho_token)
    expected_epsilon = expected_rho + math.sqrt(4.0 * expected_rho * math.log(1.0 / delta))

    assert r.rho == pytest.approx(expected_rho, rel=1e-9)
    assert r.epsilon == pytest.approx(expected_epsilon, rel=1e-9)
    assert r.components["rho_h"] == pytest.approx(expected_rho_h, rel=1e-9)
    assert r.components["rho_theta_s"] == pytest.approx(expected_rho_theta, rel=1e-9)


def test_zcdp_dp_synrag_epsilon_exceeds_rho():
    """ε must be strictly larger than ρ (the sqrt term is always positive for δ < 1)."""
    r = zcdp_dp_synrag(
        K=5, sigma_h=2.0, L=1, epsilon_theta_s=2.0,
        sigma_mu=1.0, T=10, c=0.5, tau=1.0, delta=1e-3,
    )
    assert r.epsilon > r.rho


def test_zcdp_dp_synrag_greedy_raises():
    """tau<=1e-6 must raise ValueError (L1: no fallback, privacy proof requires tau>0)."""
    with pytest.raises(ValueError, match="tau="):
        zcdp_dp_synrag(
            K=10, sigma_h=1.0, L=2, epsilon_theta_s=1.0,
            sigma_mu=0.5, T=30, c=1.0, tau=0.0, delta=1e-3,
        )


def test_zcdp_dp_synrag_result_type():
    """Return type must be ZCDPCompositionResult with correct method field."""
    r = zcdp_dp_synrag(
        K=10, sigma_h=1.0, L=2, epsilon_theta_s=1.0,
        sigma_mu=0.5, T=30, c=1.0, tau=0.5, delta=1e-3,
    )
    assert isinstance(r, ZCDPCompositionResult)
    assert r.method == "zcdp"
    assert isinstance(r.components, dict)
    assert "rho_h" in r.components
    assert "rho_mu" in r.components
    assert "rho_theta_s" in r.components


def test_zcdp_dp_synrag_larger_noise_smaller_epsilon():
    """Increasing noise σ_h and σ_μ must decrease ε (better privacy)."""
    r_low_noise = zcdp_dp_synrag(
        K=10, sigma_h=0.5, L=1, epsilon_theta_s=0.5,
        sigma_mu=0.25, T=10, c=1.0, tau=1.0, delta=1e-3,
    )
    r_high_noise = zcdp_dp_synrag(
        K=10, sigma_h=5.0, L=1, epsilon_theta_s=5.0,
        sigma_mu=2.5, T=10, c=1.0, tau=1.0, delta=1e-3,
    )
    assert r_high_noise.epsilon < r_low_noise.epsilon


# ---------------------------------------------------------------------------
# SQuAD normalization (Option 0) — punctuation/articles no longer penalized
# ---------------------------------------------------------------------------
def test_match_accuracy_ignores_trailing_punctuation():
    assert match_accuracy(["David Seville."], ["David Seville"]) == 1.0


def test_match_accuracy_ignores_articles():
    assert match_accuracy(["The Phantom of the Opera"], ["Phantom of Opera"]) == 1.0


def test_f1_score_perfect_after_squad_norm():
    assert f1_score(["the cat, sat."], ["cat sat"]) == pytest.approx(1.0)


def test_f1_score_partial_multiset():
    # pred tokens {new, york, city}; ref {new, york} -> P=2/3, R=1, F1=0.8
    assert f1_score(["New York City"], ["New York"]) == pytest.approx(0.8, abs=1e-6)


# ---------------------------------------------------------------------------
# Alias / max-over-references (Option 1) — a reference may be a list of golds
# ---------------------------------------------------------------------------
def test_match_accuracy_alias_list_any_match():
    assert match_accuracy(["Twain"], [["Mark Twain", "Twain", "Samuel Clemens"]]) == 1.0


def test_match_accuracy_alias_list_no_match():
    assert match_accuracy(["Dickens"], [["Mark Twain", "Samuel Clemens"]]) == 0.0


def test_f1_score_alias_takes_best():
    # matches the 2nd alias exactly -> 1.0 even though the 1st is unrelated
    assert f1_score(["Samuel Clemens"], [["Mark Twain", "Samuel Clemens"]]) == pytest.approx(1.0)


def test_contains_accuracy_alias_list():
    assert contains_accuracy(["I think it was Samuel Clemens"],
                             [["Mark Twain", "Samuel Clemens"]]) == 1.0


def test_rouge_l_alias_takes_best():
    assert rouge_l(["Samuel Clemens"], [["Mark Twain", "Samuel Clemens"]]) == pytest.approx(1.0)


def test_levenshtein_alias_takes_best():
    assert levenshtein(["paris"], [["london", "paris"]]) == pytest.approx(1.0)


def test_single_string_reference_still_works():
    # backward compatibility: plain string references behave as before
    assert match_accuracy(["paris"], ["paris"]) == 1.0
    assert f1_score(["paris"], ["paris"]) == pytest.approx(1.0)


def test_compute_utility_metrics_uses_alias_sets():
    res = compute_utility_metrics(
        ["Twain"], ["Mark Twain"], ["match_accuracy", "f1_score"],
        alias_sets=[["Mark Twain", "Twain"]],
    )
    assert res["match_accuracy"] == 1.0
    assert res["f1_score"] == pytest.approx(1.0)


def test_compute_utility_metrics_without_alias_sets_unchanged():
    res = compute_utility_metrics(["Twain"], ["Mark Twain"], ["match_accuracy"])
    assert res["match_accuracy"] == 0.0  # only 'Twain' vs 'Mark Twain' -> no exact match
