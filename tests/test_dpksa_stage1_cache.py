"""DP-KSA stage-1 responses must not depend on the privacy budget.

The sample-and-aggregate stage decodes one short answer per retrieved document
before any noise is drawn, so a generator may be handed a memo of those answers
that was filled under a different epsilon. These tests pin that property: the
memo must return exactly what a fresh decode would have returned.
"""

from __future__ import annotations

import pytest

from dp_methods.per_query.dpksa import generator as ksa_generator
from dp_methods.per_query.dpksa.generator import DPKSAGenerator

QUERY = "What is the capital of France?"
DOCS = [f"Paris is the capital of France. Note {i}." for i in range(3)]


@pytest.fixture(scope="module")
def gpt2_model():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    except Exception as exc:  # pragma: no cover - network/model availability
        pytest.skip(f"Could not load gpt2: {exc}")
    return model, tokenizer


def _make(model, tokenizer, epsilon: float, cache=None) -> DPKSAGenerator:
    return DPKSAGenerator(
        model=model,
        tokenizer=tokenizer,
        target_epsilon=epsilon,
        target_delta=1e-4,
        n_docs=len(DOCS),
        max_answer_tokens=4,
        max_final_tokens=4,
        k_max=2,
        random_seed=42,
        stage1_cache=cache,
    )


def _capture_responses(monkeypatch) -> list[list[str]]:
    """Record the response list each mechanism run histograms."""
    seen: list[list[str]] = []
    real = ksa_generator.build_token_histogram

    def spy(responses, **kwargs):
        seen.append(list(responses))
        return real(responses, **kwargs)

    monkeypatch.setattr(ksa_generator, "build_token_histogram", spy)
    return seen


def test_stage1_responses_are_budget_independent(gpt2_model, monkeypatch) -> None:
    model, tokenizer = gpt2_model
    seen = _capture_responses(monkeypatch)

    tight = _make(model, tokenizer, 0.5)
    loose = _make(model, tokenizer, 16.0)
    # The two budgets really are different mechanisms, not a distinction without
    # a difference.
    assert tight.gumbel_scale != loose.gumbel_scale

    tight.generate_ksa(QUERY, DOCS)
    loose.generate_ksa(QUERY, DOCS)

    assert len(seen) == 2
    assert seen[0] == seen[1]


def test_shared_memo_reproduces_a_fresh_decode(gpt2_model, monkeypatch) -> None:
    model, tokenizer = gpt2_model
    seen = _capture_responses(monkeypatch)

    uncached = _make(model, tokenizer, 0.5)
    uncached.generate_ksa(QUERY, DOCS)

    cache: dict = {}
    warming = _make(model, tokenizer, 16.0, cache=cache)
    warming.generate_ksa(QUERY, DOCS)
    assert len(cache) == len(DOCS)

    # A generator that shares the warm memo must never reach the model for
    # stage 1; only the final generation may decode.
    reusing = _make(model, tokenizer, 2.0, cache=cache)
    prompts: list[str] = []
    real_generate = reusing._generate_answer

    def spy(prompt: str, max_new_tokens: int) -> str:
        prompts.append(prompt)
        return real_generate(prompt, max_new_tokens)

    monkeypatch.setattr(reusing, "_generate_answer", spy)
    reusing.generate_ksa(QUERY, DOCS)

    assert len(prompts) == 1
    assert not any(doc in prompts[0] for doc in DOCS)
    assert seen[0] == seen[1] == seen[2]


def test_memo_holds_the_stripped_response(gpt2_model) -> None:
    model, tokenizer = gpt2_model
    cache: dict = {}
    gen = _make(model, tokenizer, 4.0, cache=cache)
    gen.generate_ksa(QUERY, DOCS)

    assert set(cache) == {(QUERY, doc) for doc in DOCS}
    for value in cache.values():
        assert value == gen._strip_scaffold(value)
