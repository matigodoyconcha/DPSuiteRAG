"""DP-RAG's per-document decoding must be unchanged by the incremental KV cache.

DP-RAG forwards one prompt per retrieved document at every token and samples the
next token from an exponential mechanism over the whole vocabulary, then appends
that same token to every prompt. The rows therefore grow in lockstep exactly as
the vote generators' do, so the cache is exact — these tests pin that down:

* the cached last-token logits match the uncached ones step after step, on a
  model whose positional table actually drives the output,
* the padded batch helper agrees with per-document forwards, which only holds
  when it left-pads and passes ``position_ids``,
* the per-query caches are released on every exit path, including an exception,
* ``BENCHDP_DISABLE_KV_CACHE=1`` restores the full-recompute behaviour.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from dp_methods.per_query.dp_rag.generator import DPICLGenerator

# The generator only carries the name around; the weights come from the
# ``position_sensitive_lm`` fixture in conftest.
TINY_MODEL = "sshleifer/tiny-gpt2"


def _make_generator(position_sensitive_lm, **overrides):
    model, tokenizer = position_sensitive_lm
    kwargs = dict(
        model_name=TINY_MODEL,
        epsilon=5.0,
        max_new_tokens=4,
        device="cpu",
        model=model,
        tokenizer=tokenizer,
    )
    kwargs.update(overrides)
    return DPICLGenerator(**kwargs)


PROMPTS = [
    "Context: Paris is the capital of France. Question: capital? Answer:",
    "Context: a considerably longer document that talks at length about "
    "something else entirely. Question: capital? Answer:",
    "Context: short. Question: capital? Answer:",
    "Answer the question in a few words.\nQuestion: capital?\nAnswer:",
]


def _prompt_ids(tokenizer, texts: list[str]) -> list[torch.Tensor]:
    return [
        torch.tensor(tokenizer(t, add_special_tokens=False)["input_ids"], dtype=torch.long)
        for t in texts
    ]


def _replay(gen, prompt_ids_list, suffixes: list[list[int]], sub_batch: int):
    """Per-step logits for a lockstep-growing suffix, with the caches open."""
    gen._open_steppers(sub_batch)
    try:
        return [gen._get_logits(prompt_ids_list, s) for s in suffixes]
    finally:
        gen._close_steppers()


def _replay_uncached(gen, prompt_ids_list, suffixes: list[list[int]]):
    """The same steps with no caches at all — the reference path."""
    return [gen._get_logits(prompt_ids_list, s) for s in suffixes]


# The suffix grows by exactly one token per step, as the mechanism appends it.
SUFFIXES = [[], [11], [11, 22], [11, 22, 33], [11, 22, 33, 44]]


# ----------------------------------------------------------------------
# The cache must not change the logits the mechanism scores
# ----------------------------------------------------------------------


@pytest.mark.parametrize("sub_batch", [1, 16])
def test_cached_logits_match_uncached(position_sensitive_lm, sub_batch):
    gen = _make_generator(position_sensitive_lm)
    ids = _prompt_ids(gen.tokenizer, PROMPTS)
    cached = _replay(gen, ids, SUFFIXES, sub_batch=sub_batch)
    uncached = _replay_uncached(gen, ids, SUFFIXES)
    for step, (c, u) in enumerate(zip(cached, uncached)):
        assert c.shape == (len(PROMPTS), gen.model.config.vocab_size)
        assert np.allclose(c, u, atol=1e-4), f"step {step} diverged"
    # The logits must actually move with the appended tokens, or the comparison
    # would pass on a model that ignores its input.
    assert not np.allclose(cached[0], cached[-1], atol=1e-4)


def test_the_cache_is_really_used(position_sensitive_lm):
    gen = _make_generator(position_sensitive_lm)
    ids = _prompt_ids(gen.tokenizer, PROMPTS)
    gen._open_steppers(16)
    try:
        for suffix in SUFFIXES:
            gen._get_logits(ids, suffix)
        steppers = gen._steppers._steppers
        assert len(steppers) == 1
        # One prefill, then one cached step per appended token.
        assert steppers[0].n_full_steps == 1
        assert steppers[0].n_cached_steps == len(SUFFIXES) - 1
    finally:
        gen._close_steppers()


def test_disable_env_var_takes_the_recompute_path(position_sensitive_lm, monkeypatch):
    monkeypatch.setenv("BENCHDP_DISABLE_KV_CACHE", "1")
    gen = _make_generator(position_sensitive_lm)
    ids = _prompt_ids(gen.tokenizer, PROMPTS)
    cached = _replay(gen, ids, SUFFIXES, sub_batch=16)
    monkeypatch.delenv("BENCHDP_DISABLE_KV_CACHE")
    uncached = _replay_uncached(gen, ids, SUFFIXES)
    for c, u in zip(cached, uncached):
        assert np.allclose(c, u, atol=1e-6)


# ----------------------------------------------------------------------
# The padded batch must agree with per-document forwards
# ----------------------------------------------------------------------


def test_batch_helper_matches_sequential_with_unequal_lengths(position_sensitive_lm):
    """Left padding + explicit position_ids, checked where right padding would fail.

    The prompts have three different lengths, so a right-padded batch would read
    its last token from a pad slot and hand the model positions shifted by the
    padding — on this position-sensitive model that shows up as different logits.
    """
    gen = _make_generator(position_sensitive_lm)
    ids = _prompt_ids(gen.tokenizer, PROMPTS)
    assert len({int(t.numel()) for t in ids}) > 2, "this case must exercise padding"
    for suffix in SUFFIXES:
        batched = gen._get_logits_batch(ids, suffix)
        sequential = gen._get_logits_sequential(ids, suffix)
        assert np.allclose(batched, sequential, atol=1e-4)


def test_batch_helper_matches_sequential_with_equal_lengths(position_sensitive_lm):
    """The pad-free control: agreement here cannot come from the padding logic."""
    gen = _make_generator(position_sensitive_lm)
    ids = _prompt_ids(gen.tokenizer, ["A B C D E", "F G H I J", "K L M N O"])
    assert len({int(t.numel()) for t in ids}) == 1
    assert np.allclose(
        gen._get_logits_batch(ids, [7]), gen._get_logits_sequential(ids, [7]), atol=1e-5
    )


# ----------------------------------------------------------------------
# Per-query lifetime of the caches
# ----------------------------------------------------------------------


def test_steppers_are_closed_after_generation(position_sensitive_lm):
    gen = _make_generator(position_sensitive_lm)
    answer = gen.generate_with_dp(json.dumps(PROMPTS))
    assert isinstance(answer, str)
    assert gen._steppers is None


def test_steppers_are_closed_when_generation_raises(position_sensitive_lm, monkeypatch):
    gen = _make_generator(position_sensitive_lm)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("mechanism failed")

    monkeypatch.setattr(gen, "_sample_token", _boom)
    with pytest.raises(RuntimeError):
        gen.generate_with_dp(json.dumps(PROMPTS))
    assert gen._steppers is None


def test_steppers_are_closed_when_eos_stops_generation(position_sensitive_lm):
    gen = _make_generator(position_sensitive_lm)
    eos = int(gen.tokenizer.eos_token_id)
    monkey = {"n": 0}

    def _always_eos(_utilities, seed):  # noqa: ARG001
        monkey["n"] += 1
        return eos

    gen._sample_token = _always_eos
    assert gen.generate_with_dp(json.dumps(PROMPTS)) == ""
    assert monkey["n"] == 1  # stopped at the first token
    assert gen._steppers is None


def test_generation_is_unchanged_by_the_cache(position_sensitive_lm, monkeypatch):
    """End to end: the emitted token sequence must not depend on the cache."""
    gen = _make_generator(position_sensitive_lm)
    with_cache = gen.generate_with_dp(json.dumps(PROMPTS))

    monkeypatch.setenv("BENCHDP_DISABLE_KV_CACHE", "1")
    without_cache = _make_generator(position_sensitive_lm).generate_with_dp(json.dumps(PROMPTS))
    assert with_cache == without_cache
