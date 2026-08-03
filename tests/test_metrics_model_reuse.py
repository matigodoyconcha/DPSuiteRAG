"""Scoring models are held per process; the numbers must not notice.

``bert_score_f1`` and ``perplexity`` keep their models resident between calls.
Both are read-only users of those models, so a call served by a resident copy has
to produce exactly the value a freshly loaded copy produces — these tests compare
the two, bit for bit.
"""

from __future__ import annotations

import pytest

from metrics import utility
from metrics.utility import bert_score_f1, perplexity

PREDS = ["the cat sat on the mat", "paris is in france"]
REFS = ["a cat was sitting on a mat", "the capital of france is paris"]


def test_bert_score_f1_matches_a_fresh_scorer() -> None:
    utility._BERTSCORERS.clear()
    try:
        cold = bert_score_f1(PREDS, REFS)
    except Exception as exc:  # pragma: no cover - model availability
        pytest.skip(f"bert_score unavailable: {exc}")
    if not utility._BERTSCORERS:
        pytest.skip("bert_score degraded to 0.0; nothing was made resident")

    resident = bert_score_f1(PREDS, REFS)
    assert resident == cold

    scorer = utility._BERTSCORERS["en"]
    # Scoring an unrelated file in between must leave the next value untouched.
    bert_score_f1(["a completely different sentence"], ["another unrelated line"])
    assert utility._BERTSCORERS["en"] is scorer
    assert bert_score_f1(PREDS, REFS) == cold

    utility._BERTSCORERS.clear()
    reloaded = bert_score_f1(PREDS, REFS)
    assert utility._BERTSCORERS["en"] is not scorer
    assert reloaded == cold


def test_bert_score_f1_degrades_without_the_library(monkeypatch) -> None:
    """A scorer that cannot be built still yields 0.0 rather than raising."""
    import builtins

    utility._BERTSCORERS.clear()
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "bert_score":
            raise ImportError("no bert_score")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    monkeypatch.setattr(utility, "_BERTSCORE_WARNED", True)
    assert bert_score_f1(PREDS, REFS) == 0.0
    utility._BERTSCORERS.clear()


def test_perplexity_matches_a_fresh_load() -> None:
    texts = ["the cat sat on the mat", "paris is the capital of france"]
    model_name = "sshleifer/tiny-gpt2"

    utility._PPL_MODELS.clear()
    try:
        cold = perplexity(texts, model_name=model_name, device="cpu")
    except Exception as exc:  # pragma: no cover - model availability
        pytest.skip(f"Could not load {model_name}: {exc}")

    entry = utility._PPL_MODELS[(model_name, "cpu")]
    resident = perplexity(texts, model_name=model_name, device="cpu")
    assert resident == cold

    # Scoring other texts in between leaves the model in the same state.
    perplexity(["an unrelated sentence entirely"], model_name=model_name, device="cpu")
    assert utility._PPL_MODELS[(model_name, "cpu")] is entry
    assert perplexity(texts, model_name=model_name, device="cpu") == cold

    utility._PPL_MODELS.clear()
    reloaded = perplexity(texts, model_name=model_name, device="cpu")
    assert utility._PPL_MODELS[(model_name, "cpu")] is not entry
    assert reloaded == cold
