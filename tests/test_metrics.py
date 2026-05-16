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
    rouge_l,
    compute_utility_metrics,
    bert_score_f1,
    recall_at_k,
    answer_faithfulness,
    answer_relevancy,
)
from metrics.privacy import (
    mia_auc_bleu,
    private_entity_leakage,
    compute_privacy_metrics,
    bleu_membership_score,
    privacy_budget_consumption,
)
from metrics.privacy_accounting import compute_privacy_accounting


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
# private_entity_leakage
# ---------------------------------------------------------------------------

def test_private_entity_leakage_no_leak():
    score = private_entity_leakage(["text without private entities"], ["secret_name"])
    assert score == 0.0


def test_private_entity_leakage_full_leak():
    score = private_entity_leakage(["contains secret_name here"], ["secret_name"])
    assert score == 1.0


def test_private_entity_leakage_empty_entities():
    score = private_entity_leakage(["any text"], [])
    assert score == 0.0


# ---------------------------------------------------------------------------
# compute_privacy_metrics (dispatcher)
# ---------------------------------------------------------------------------

def test_compute_privacy_metrics_runs():
    result = compute_privacy_metrics(
        ["pred"], ["ref"], ["private_entity"],
        ["mia_auc_bleu", "private_entity_leakage"],
    )
    assert "mia_auc_bleu" in result
    assert "private_entity_leakage" in result


# ---------------------------------------------------------------------------
# privacy_accounting
# ---------------------------------------------------------------------------

def test_privacy_accounting_all_methods_return_finite():
    results = compute_privacy_accounting(
        eps_retrieval=1.0,
        eps_generation_per_token=0.1,
        num_tokens=32,
        delta=1e-3,
        methods=("basic", "rdp", "advanced"),
    )
    for name in ("basic", "rdp", "advanced"):
        assert name in results
        assert math.isfinite(results[name].epsilon)
        assert results[name].epsilon > 0


def test_privacy_accounting_delta_preserved():
    # All methods must return the same δ that was passed in.
    delta = 1e-3
    results = compute_privacy_accounting(
        eps_retrieval=1.0,
        eps_generation_per_token=0.1,
        num_tokens=32,
        delta=delta,
        methods=("basic", "rdp", "advanced"),
    )
    for name in ("basic", "rdp", "advanced"):
        assert results[name].delta == pytest.approx(delta)


def test_privacy_accounting_large_epsilon_large_noise():
    results = compute_privacy_accounting(
        eps_retrieval=1000.0,
        eps_generation_per_token=1000.0,
        num_tokens=1,
        delta=1e-3,
        methods=("basic",),
    )
    assert results["basic"].epsilon > 100


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
        [],
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
        [],
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


def test_zcdp_dp_synrag_greedy_fallback():
    """tau=0 must not raise and must return a finite ε."""
    import math

    r = zcdp_dp_synrag(
        K=10, sigma_h=1.0, L=2, epsilon_theta_s=1.0,
        sigma_mu=0.5, T=30, c=1.0, tau=0.0, delta=1e-3,
    )
    assert math.isfinite(r.epsilon)
    assert r.epsilon > 0


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
