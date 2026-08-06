"""The incremental KV cache must emit exactly the tokens the full recompute does.

The vote generators decode one token at a time and append the same winning token
to every prompt, which is what makes an incremental cache exact. These tests
replay that pattern and compare the cached path against the uncached reference
token by token, including the cases that break naive implementations:

* rows of unequal length (left padding + position_ids),
* an absolute-position architecture (GPT-2), where a wrong position offset would
  silently change the logits rather than raise,
* rows that stop growing in lockstep (must fall back, not corrupt),
* a memory budget too small to cache (must fall back, not OOM or diverge).

Cached-versus-uncached comparisons cannot see a fault in the padding itself: both
paths pad through the same helper, so a wrong padding side moves them together.
The one-row reference below is what pins that down — a single row is never padded,
so it is the ground truth the padded batch has to reproduce.

Every test decodes with the ``position_sensitive_lm`` fixture (see conftest),
whose 64-dimensional hidden state makes a wrong position or a last token read out
of a pad slot show up as different logits instead of vanishing in the noise.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.incremental_lm import (
    IncrementalLMStepper,
    SubBatchSteppers,
    greedy_next_logits,
    greedy_next_tokens,
    kv_cache_bytes_per_token,
)

DISTIL_MODEL = "distilgpt2"

_MODEL_CACHE: dict[str, tuple] = {}


def _load(name: str):
    if name not in _MODEL_CACHE:
        model = AutoModelForCausalLM.from_pretrained(name)
        model.eval()
        _MODEL_CACHE[name] = (model, AutoTokenizer.from_pretrained(name))
    return _MODEL_CACHE[name]


def _decode_both(model, rows: list[list[int]], n_steps: int, pad_token_id: int = 0):
    """Run ``n_steps`` lockstep decoding steps twice: cached and uncached.

    Returns ``(cached_tokens, uncached_tokens, stepper)`` where each entry of the
    token lists is the per-row argmax at that step. The token appended to every
    row is the first row's argmax, so the batch grows by a shared suffix exactly
    as the vote generators grow it.
    """
    stepper = IncrementalLMStepper(model, pad_token_id=pad_token_id)
    cached, uncached = [], []
    cached_rows = [list(r) for r in rows]
    uncached_rows = [list(r) for r in rows]
    for _ in range(n_steps):
        c = stepper.next_tokens(cached_rows)
        u = greedy_next_tokens(model, uncached_rows, pad_token_id=pad_token_id)
        cached.append(c)
        uncached.append(u)
        cached_rows = [r + [c[0]] for r in cached_rows]
        uncached_rows = [r + [u[0]] for r in uncached_rows]
    return cached, uncached, stepper


def _prompt_ids(tokenizer, texts: list[str]) -> list[list[int]]:
    return [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]


# ----------------------------------------------------------------------
# The padded batch is only correct if it reproduces one-row forwards
# ----------------------------------------------------------------------


def test_padded_batch_matches_one_row_at_a_time(position_sensitive_lm):
    """A batch of unequal rows must give each row what it gets on its own.

    A one-row batch has no padding at all, so it fixes both the padding side and
    the positions independently of the batching helper: right padding would read
    the last logits out of a pad slot, and positions taken from a plain ``arange``
    would offset every row by its own pad count.
    """
    model, tokenizer = position_sensitive_lm
    rows = _prompt_ids(
        tokenizer,
        [
            "Q: who wrote it? A:",
            "Context: a much longer document about something else entirely. "
            "Q: who wrote it? A:",
            "Short. A:",
        ],
    )
    assert len({len(r) for r in rows}) == 3, "this case must exercise padding"
    batched = greedy_next_logits(model, rows, pad_token_id=0)
    for i, row in enumerate(rows):
        alone = greedy_next_logits(model, [row], pad_token_id=0)
        assert torch.allclose(batched[i], alone[0], atol=1e-4), f"row {i} differs padded"


def test_stepper_prefill_matches_one_row_at_a_time(position_sensitive_lm):
    """The same ground truth for the cache-building prefill, which pads separately."""
    model, tokenizer = position_sensitive_lm
    rows = _prompt_ids(
        tokenizer,
        ["Tiny. A:", "A rather longer prompt with a good few more tokens in it. A:"],
    )
    stepper = IncrementalLMStepper(model, pad_token_id=0)
    batched = stepper.next_logits(rows)
    for i, row in enumerate(rows):
        alone = greedy_next_logits(model, [row], pad_token_id=0)
        assert torch.allclose(batched[i], alone[0], atol=1e-4), f"row {i} differs padded"


def test_cached_steps_match_one_row_at_a_time(position_sensitive_lm):
    """And so must every cached step, row by row, against an unpadded recompute."""
    model, tokenizer = position_sensitive_lm
    rows = _prompt_ids(
        tokenizer,
        ["Tiny. A:", "A rather longer prompt with a good few more tokens in it. A:"],
    )
    stepper = IncrementalLMStepper(model, pad_token_id=0)
    stepper.next_logits(rows)
    for token in (11, 22, 33):
        rows = [r + [token] for r in rows]
        cached = stepper.next_logits(rows)
        for i, row in enumerate(rows):
            alone = greedy_next_logits(model, [row], pad_token_id=0)
            assert torch.allclose(cached[i], alone[0], atol=1e-4), f"row {i} diverged"
    assert stepper.n_cached_steps == 3


# ----------------------------------------------------------------------
# Equal and unequal prompt lengths on a GPT-2-style (absolute position) model
# ----------------------------------------------------------------------


def test_cached_matches_uncached_equal_lengths(position_sensitive_lm):
    model, tokenizer = position_sensitive_lm
    rows = _prompt_ids(
        tokenizer,
        [
            "The capital of France is",
            "The capital of Spain is",
            "The capital of Italy is",
        ],
    )
    assert len({len(r) for r in rows}) == 1, "this case is meant to be pad-free"
    cached, uncached, stepper = _decode_both(model, rows, n_steps=6)
    assert cached == uncached
    assert stepper.n_cached_steps == 5  # step 0 prefills, the rest reuse the cache


def test_cached_matches_uncached_unequal_lengths_left_padding(position_sensitive_lm):
    model, tokenizer = position_sensitive_lm
    rows = _prompt_ids(
        tokenizer,
        [
            "Q: who wrote it? A:",
            "Context: a much longer document about something entirely different. "
            "Q: who wrote it? A:",
            "Short. A:",
        ],
    )
    assert len({len(r) for r in rows}) == 3, "this case must exercise left padding"
    cached, uncached, stepper = _decode_both(model, rows, n_steps=8)
    assert cached == uncached
    assert stepper.n_cached_steps == 7
    # The positions really do drive the output, so a wrong offset would show.
    assert len({tuple(step) for step in uncached}) > 1


def test_cached_matches_uncached_distilgpt2_absolute_positions():
    """Real GPT-2 weights: a wrong position offset shows up as different tokens."""
    model, tokenizer = _load(DISTIL_MODEL)
    rows = _prompt_ids(
        tokenizer,
        [
            "The capital city of France is called",
            "Instruction: answer briefly.\nContext: Paris is the capital of France.\n"
            "Question: what is the capital of France?\nAnswer:",
            "Once upon a time",
        ],
    )
    cached, uncached, stepper = _decode_both(
        model, rows, n_steps=8, pad_token_id=tokenizer.eos_token_id
    )
    assert cached == uncached
    assert stepper.n_cached_steps == 7
    # A non-degenerate check: the model must actually be saying something.
    assert len({tuple(step) for step in uncached}) > 1


# ----------------------------------------------------------------------
# Defensive paths
# ----------------------------------------------------------------------


def test_diverging_row_falls_back_to_full_recompute(position_sensitive_lm):
    """A row that is not its previous self plus a suffix must not use the cache."""
    model, tokenizer = position_sensitive_lm
    rows = _prompt_ids(tokenizer, ["The capital of France is", "A different prompt here"])
    stepper = IncrementalLMStepper(model, pad_token_id=0)

    first = stepper.next_tokens(rows)
    assert first == greedy_next_tokens(model, rows, pad_token_id=0)

    # Rewrite one row's history instead of appending to it.
    diverged = [rows[0] + [42], [999] + rows[1][1:] + [42]]
    got = stepper.next_tokens(diverged)
    assert got == greedy_next_tokens(model, diverged, pad_token_id=0)
    assert stepper.n_full_steps == 2  # the divergence forced a second prefill

    # And the rebuilt cache is usable again from there.
    grown = [r + [7] for r in diverged]
    assert stepper.next_tokens(grown) == greedy_next_tokens(model, grown, pad_token_id=0)
    assert stepper.n_cached_steps == 1


def test_rows_growing_by_different_amounts_fall_back(position_sensitive_lm):
    model, tokenizer = position_sensitive_lm
    rows = _prompt_ids(tokenizer, ["The capital of France is", "The capital of Spain is"])
    stepper = IncrementalLMStepper(model, pad_token_id=0)
    stepper.next_tokens(rows)
    ragged = [rows[0] + [11], rows[1] + [11, 12]]
    assert stepper.next_tokens(ragged) == greedy_next_tokens(model, ragged, pad_token_id=0)
    assert stepper.n_cached_steps == 0


def test_memory_guard_refuses_to_cache_but_stays_correct(position_sensitive_lm):
    model, tokenizer = position_sensitive_lm
    rows = _prompt_ids(tokenizer, ["The capital of France is", "Short one"])
    stepper = IncrementalLMStepper(model, pad_token_id=0, max_cache_bytes=1)
    for _ in range(3):
        got = stepper.next_tokens(rows)
        assert got == greedy_next_tokens(model, rows, pad_token_id=0)
        rows = [r + [got[0]] for r in rows]
    assert stepper.caching_enabled is False
    assert stepper.n_cached_steps == 0


def test_env_var_disables_the_cache(position_sensitive_lm, monkeypatch):
    """BENCHDP_DISABLE_KV_CACHE=1 must take the old recompute path, read per stepper."""
    model, tokenizer = position_sensitive_lm
    rows = _prompt_ids(tokenizer, ["The capital of France is", "Short one"])

    monkeypatch.setenv("BENCHDP_DISABLE_KV_CACHE", "1")
    disabled = IncrementalLMStepper(model, pad_token_id=0)
    assert disabled.caching_enabled is False
    steppers = SubBatchSteppers(model, pad_token_id=0, sub_batch=1)
    try:
        for _ in range(3):
            got = steppers.next_tokens(rows)
            assert got == greedy_next_tokens(model, rows, pad_token_id=0)
            rows = [r + [got[0]] for r in rows]
        assert all(s.n_cached_steps == 0 for s in steppers._steppers)
    finally:
        steppers.close()

    # Read at construction time, so clearing it re-enables caching without a reimport.
    monkeypatch.delenv("BENCHDP_DISABLE_KV_CACHE")
    assert IncrementalLMStepper(model, pad_token_id=0).caching_enabled is True


def test_reset_clears_the_cache(position_sensitive_lm):
    model, tokenizer = position_sensitive_lm
    rows = _prompt_ids(tokenizer, ["The capital of France is", "Short one"])
    stepper = IncrementalLMStepper(model, pad_token_id=0)
    stepper.next_tokens(rows)
    stepper.reset()
    grown = [r + [5] for r in rows]
    assert stepper.next_tokens(grown) == greedy_next_tokens(model, grown, pad_token_id=0)
    assert stepper.n_cached_steps == 0  # the reset forced a prefill, not a cached step


# ----------------------------------------------------------------------
# Sub-batching and the size estimate
# ----------------------------------------------------------------------


def test_sub_batch_steppers_match_uncached_across_sub_batches(position_sensitive_lm):
    model, tokenizer = position_sensitive_lm
    rows = _prompt_ids(
        tokenizer,
        [f"Context number {i}. Question: what is it? Answer:" for i in range(9)],
    )
    # Uneven last sub-batch, so the slicing is exercised.
    steppers = SubBatchSteppers(model, pad_token_id=0, sub_batch=4, max_new_tokens=4)
    try:
        for _ in range(4):
            got = steppers.next_tokens(rows)
            assert got == greedy_next_tokens(model, rows, pad_token_id=0)
            rows = [r + [got[0]] for r in rows]
        assert len(steppers._steppers) == 3
        assert all(s.n_cached_steps == 3 for s in steppers._steppers)
    finally:
        steppers.close()
    assert steppers._steppers is None


# ----------------------------------------------------------------------
# Sorting the rows by length
# ----------------------------------------------------------------------

# Six rows, each about something else and each cut to its own length. The lengths
# are scrambled, so the length-sorted order is neither the input order nor its
# reverse; the subjects differ, so a row that came back in the wrong place is a
# different logit vector rather than a rounding difference.
_RAGGED = [
    (12, "The capital of France is Paris, a city on the river Seine in Europe"),
    (4, "Photosynthesis turns sunlight into sugar inside the leaves of green plants"),
    (8, "The Pacific ocean is the largest and deepest of the world's oceans"),
    (2, "Beethoven wrote nine symphonies before he died in Vienna in 1827"),
    (17, "A compiler translates source code into machine instructions that a "
         "processor can execute directly without any further interpretation"),
    (6, "Antarctica is a desert, because almost no precipitation falls there"),
]


def _ragged_rows(tokenizer):
    rows = [
        _prompt_ids(tokenizer, [text])[0][:length] for length, text in _RAGGED
    ]
    lengths = [len(r) for r in rows]
    assert lengths == [length for length, _ in _RAGGED], "a row was shorter than its cut"
    assert len(set(lengths)) == len(lengths), "the rows must have distinct lengths"
    assert lengths != sorted(lengths), "sorting must actually permute the rows"
    return rows


def test_sorting_by_length_returns_the_logits_in_the_input_order(position_sensitive_lm):
    """Row ``i`` of the result must be row ``i`` of the input, however it was batched.

    The caller reads row ``i`` as its own item ``i`` — DP-RAG's exponential mechanism
    scores it as document ``i`` — so a permutation left un-inverted would raise nothing
    and fail no shape check; it would just score the wrong document at the reported
    epsilon. Comparing against the unsorted steppers is what makes that visible: an
    inverse missing from *both* paths would let a sorted-versus-sorted comparison agree.
    """
    model, tokenizer = position_sensitive_lm
    rows = _ragged_rows(tokenizer)
    plain = SubBatchSteppers(model, pad_token_id=0, sub_batch=2, max_new_tokens=4)
    sorted_ = SubBatchSteppers(
        model, pad_token_id=0, sub_batch=2, max_new_tokens=4, sort_by_length=True
    )
    try:
        for _ in range(4):
            reference = plain.next_logits(rows)
            got = sorted_.next_logits(rows)
            assert got.shape == reference.shape
            assert torch.allclose(got, reference, atol=1e-4)
            # A model that ignored its input would make any permutation pass.
            assert not torch.allclose(reference[0], reference[1], atol=1e-4)
            rows = [r + [int(reference[0].argmax())] for r in rows]
    finally:
        plain.close()
        sorted_.close()


def test_sorting_by_length_groups_rows_of_similar_length(position_sensitive_lm):
    """The point of sorting: each forward pads to its own longest row, not the batch's.

    Padding is charged per row and attention costs O(rows * seq^2), so a sub-batch that
    mixes the shortest and longest rows pays for the longest one six times over.
    """
    model, tokenizer = position_sensitive_lm
    rows = _ragged_rows(tokenizer)
    steppers = SubBatchSteppers(
        model, pad_token_id=0, sub_batch=2, max_new_tokens=4, sort_by_length=True
    )
    order = steppers._row_order(rows)
    lengths = [len(rows[i]) for i in order]
    assert lengths == sorted(lengths)

    def padded_tokens(seq: list[int]) -> int:
        return sum(2 * max(seq[i : i + 2]) for i in range(0, len(seq), 2))

    assert padded_tokens(lengths) < padded_tokens([len(r) for r in rows])


def test_the_sort_order_is_the_same_at_every_step(position_sensitive_lm):
    """A permutation that moved between steps would break every sub-batch's cache.

    Each sub-batch caches the rows it saw last step; a row that lands in a different
    sub-batch is no longer its previous self plus a suffix, so the stepper would drop
    the cache and recompute at every step. Every row grows by the same token, so the
    length order is invariant — this pins that the cached path is actually taken.
    """
    model, tokenizer = position_sensitive_lm
    rows = _ragged_rows(tokenizer)
    steppers = SubBatchSteppers(
        model, pad_token_id=0, sub_batch=2, max_new_tokens=4, sort_by_length=True
    )
    try:
        first = steppers._row_order(rows)
        for _ in range(4):
            got = steppers.next_tokens(rows)
            assert steppers._row_order(rows) == first
            rows = [r + [got[0]] for r in rows]
        assert all(s.n_cached_steps == 3 for s in steppers._steppers)
    finally:
        steppers.close()


def test_kv_cache_bytes_per_token_matches_the_config(position_sensitive_lm):
    model, _ = position_sensitive_lm
    config = model.config
    head_dim = config.n_embd // config.n_head
    expected = 2 * config.n_layer * config.n_head * head_dim * 4  # fp32
    assert model.dtype == torch.float32
    assert kv_cache_bytes_per_token(model) == expected


def test_kv_cache_bytes_per_token_unknown_config_is_zero():
    class _Bare:
        config = object()

        def parameters(self):
            return iter(())

    assert kv_cache_bytes_per_token(_Bare()) == 0
