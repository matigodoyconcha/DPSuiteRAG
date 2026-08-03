"""DP-SynRAG's per-document decoding must be unchanged by the incremental KV cache.

Stage 2d forwards one prompt per document of a cluster at every token, aggregates
the clipped logit rows into z_n and samples a single token that is appended to
*every* prompt. The rows therefore grow in lockstep exactly as DP-RAG's do, so the
cache is exact — these tests pin that down:

* the cached last-token logits match the uncached ones step after step, on a model
  whose positional table actually drives the output,
* the padded batch helper agrees with per-document forwards, which only holds when
  it left-pads and passes ``position_ids``, and keeps agreeing when a real token
  happens to equal ``pad_token_id`` (a mask built from token values would drop it
  and read the row's logits from the wrong position),
* the per-cluster caches are released on every exit path, including an exception,
* ``BENCHDP_DISABLE_KV_CACHE=1`` restores the full-recompute behaviour,
* the synthetic text itself is byte-identical with and without the cache.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dp_methods.offline.dp_synrag.synthetic_generator import DPSyntheticGenerator

# The generator only carries the name around; the weights come from the
# ``position_sensitive_lm`` fixture in conftest.
TINY_MODEL = "sshleifer/tiny-gpt2"


def _make_generator(position_sensitive_lm, **overrides):
    model, tokenizer = position_sensitive_lm
    kwargs = dict(model_name=TINY_MODEL, T=4, c=1.0, tau=1.0, device="cpu", seed=7)
    kwargs.update(overrides)
    gen = DPSyntheticGenerator(**kwargs)
    # Inject the fixture's weights instead of letting _ensure_loaded download any.
    gen.model = model
    gen.tokenizer = tokenizer
    return gen


DOCUMENTS = [
    "Paris is the capital of France and sits on the river Seine.",
    "A considerably longer document that talks at length about something else "
    "entirely, going on for a while so its token count clearly differs.",
    "Short one.",
    "A third document of yet another length, medium sized.",
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
# The cache must not change the logits the mechanism aggregates
# ----------------------------------------------------------------------


@pytest.mark.parametrize("sub_batch", [1, 16])
def test_cached_logits_match_uncached(position_sensitive_lm, sub_batch):
    gen = _make_generator(position_sensitive_lm)
    ids = _prompt_ids(gen.tokenizer, DOCUMENTS)
    cached = _replay(gen, ids, SUFFIXES, sub_batch=sub_batch)
    uncached = _replay_uncached(gen, ids, SUFFIXES)
    for step, (c, u) in enumerate(zip(cached, uncached)):
        assert c.shape == (len(DOCUMENTS), gen.model.config.vocab_size)
        assert np.allclose(c, u, atol=1e-4), f"step {step} diverged"
    # The logits must actually move with the appended tokens, or the comparison
    # would pass on a model that ignores its input.
    assert not np.allclose(cached[0], cached[-1], atol=1e-4)


def test_the_cache_is_really_used(position_sensitive_lm):
    gen = _make_generator(position_sensitive_lm)
    ids = _prompt_ids(gen.tokenizer, DOCUMENTS)
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
    ids = _prompt_ids(gen.tokenizer, DOCUMENTS)
    cached = _replay(gen, ids, SUFFIXES, sub_batch=16)
    monkeypatch.delenv("BENCHDP_DISABLE_KV_CACHE")
    uncached = _replay_uncached(gen, ids, SUFFIXES)
    for c, u in zip(cached, uncached):
        assert np.allclose(c, u, atol=1e-6)


# ----------------------------------------------------------------------
# The padded batch must agree with per-document forwards
# ----------------------------------------------------------------------


def test_batch_helper_matches_sequential_with_unequal_lengths(position_sensitive_lm):
    """Left padding + explicit position_ids, checked where padding is exercised.

    The documents have four different lengths, so every row but the longest is
    padded; on this position-sensitive model a batch whose ``position_ids`` do not
    follow each row's own tokens gives different logits from a batch-of-one.
    """
    gen = _make_generator(position_sensitive_lm)
    ids = _prompt_ids(gen.tokenizer, DOCUMENTS)
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


def test_batch_helper_survives_a_real_token_equal_to_the_pad_id(position_sensitive_lm):
    """A pad-valued token inside a document must stay a real token.

    Deriving the attention mask from the token *values* (``ids != pad_token_id``)
    both masks such a token out and undercounts the row, so the last-token logits
    are read one position too early. The mask has to come from the row lengths.
    """
    gen = _make_generator(position_sensitive_lm)
    pad_id = gen._pad_token_id
    ids = _prompt_ids(gen.tokenizer, DOCUMENTS[:2])
    ids = [torch.cat([row[:3], torch.tensor([pad_id]), row[3:]]) for row in ids]
    assert any(int(pad_id) in row.tolist() for row in ids)
    assert np.allclose(
        gen._get_logits_batch(ids, [7]), gen._get_logits_sequential(ids, [7]), atol=1e-4
    )


# ----------------------------------------------------------------------
# Per-cluster lifetime of the caches
# ----------------------------------------------------------------------


def _clusters() -> list[list[int]]:
    """Two non-empty clusters and one empty one, as Stage 1c can produce."""
    return [[0, 1, 2, 3], [], [1, 3]]


def test_steppers_are_closed_after_generation(position_sensitive_lm):
    gen = _make_generator(position_sensitive_lm)
    texts = gen.generate_for_clusters(DOCUMENTS, _clusters())
    assert len(texts) == 3
    assert texts[1] == ""  # the empty cluster keeps its slot
    assert gen._steppers is None


def test_steppers_are_closed_when_generation_raises(position_sensitive_lm, monkeypatch):
    gen = _make_generator(position_sensitive_lm)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("mechanism failed")

    monkeypatch.setattr(gen, "_sample_token", _boom)
    with pytest.raises(RuntimeError):
        gen.generate_for_clusters(DOCUMENTS, _clusters())
    assert gen._steppers is None


def test_steppers_are_closed_when_eos_stops_generation(position_sensitive_lm):
    gen = _make_generator(position_sensitive_lm)
    eos = int(gen.tokenizer.eos_token_id)
    calls = {"n": 0}

    def _always_eos(_z_n, seed):  # noqa: ARG001
        calls["n"] += 1
        return eos

    gen._sample_token = _always_eos
    assert gen.generate_for_clusters(DOCUMENTS, [[0, 1]]) == [""]
    assert calls["n"] == 1  # stopped at the first token
    assert gen._steppers is None


# ----------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------


@pytest.mark.parametrize("batch_threshold", [0, 8])
def test_synthetic_text_is_unchanged_by_the_cache(
    position_sensitive_lm, monkeypatch, batch_threshold
):
    """Both shipped paths — sequential (paper_faithful) and batched (full)."""
    gen = _make_generator(position_sensitive_lm, batch_threshold=batch_threshold)
    with_cache = gen.generate_for_clusters(DOCUMENTS, _clusters())

    monkeypatch.setenv("BENCHDP_DISABLE_KV_CACHE", "1")
    without_cache = _make_generator(
        position_sensitive_lm, batch_threshold=batch_threshold
    ).generate_for_clusters(DOCUMENTS, _clusters())
    assert with_cache == without_cache
