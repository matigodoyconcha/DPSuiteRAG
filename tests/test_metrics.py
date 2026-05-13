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
    assert match_accuracy(["hola"], ["hola"]) == 1.0


def test_match_accuracy_zero():
    assert match_accuracy(["hola"], ["chau"]) == 0.0


def test_match_accuracy_case_insensitive():
    assert match_accuracy(["Hola Mundo"], ["hola mundo"]) == 1.0


def test_match_accuracy_empty_predictions():
    assert match_accuracy([], ["ref"]) == 0.0


def test_match_accuracy_partial():
    score = match_accuracy(["bien", "mal"], ["bien", "bien"])
    assert score == 0.5


# ---------------------------------------------------------------------------
# rouge_l
# ---------------------------------------------------------------------------

def test_rouge_l_identical():
    score = rouge_l(["el gato come pescado"], ["el gato come pescado"])
    assert score == pytest.approx(1.0, abs=1e-4)


def test_rouge_l_empty_prediction():
    score = rouge_l([], ["referencia"])
    assert score == 0.0


def test_rouge_l_returns_float_in_range():
    score = rouge_l(["algo diferente aquí"], ["referencia distinta"])
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_utility_metrics (dispatcher)
# ---------------------------------------------------------------------------

def test_compute_utility_metrics_runs():
    result = compute_utility_metrics(
        ["respuesta"], ["respuesta"], ["match_accuracy", "rouge_l"]
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
    preds = ["el perro corre", "la casa es grande", "llueve en la ciudad"]
    refs  = ["el perro corre", "la casa es pequeña", "hace sol"]
    auc = mia_auc_bleu(preds, refs)
    assert 0.0 <= auc <= 1.0


def test_mia_auc_bleu_empty():
    assert mia_auc_bleu([], []) == 0.5


# ---------------------------------------------------------------------------
# private_entity_leakage
# ---------------------------------------------------------------------------

def test_private_entity_leakage_no_leak():
    score = private_entity_leakage(["texto sin entidades privadas"], ["nombre_secreto"])
    assert score == 0.0


def test_private_entity_leakage_full_leak():
    score = private_entity_leakage(["contiene nombre_secreto aquí"], ["nombre_secreto"])
    assert score == 1.0


def test_private_entity_leakage_empty_entities():
    score = private_entity_leakage(["cualquier texto"], [])
    assert score == 0.0


# ---------------------------------------------------------------------------
# compute_privacy_metrics (dispatcher)
# ---------------------------------------------------------------------------

def test_compute_privacy_metrics_runs():
    result = compute_privacy_metrics(
        ["pred"], ["ref"], ["entidad_privada"],
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
