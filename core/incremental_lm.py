"""Greedy last-token decoding for a batch of prompts that grow by a shared suffix.

The vote-based generators (DPVoteRAG, DPSparseVoteRAG, Private-RAG) decode one
token at a time: every step forwards ``m`` voter prompts, votes on their argmax
tokens, and appends the *same* winning token to every prompt. The prompts are
~2000 tokens and generation runs for up to ~50 steps, so a plain re-forward of
the whole sequence at every step recomputes ~98% of key/value vectors that are
provably unchanged since the previous step.

Because every row grows by the identical suffix, the batch stays in lockstep and
an incremental KV cache is exact: prefill once, then forward only the new tokens
with ``past_key_values``. :class:`IncrementalLMStepper` does that for one fixed
batch; :class:`SubBatchSteppers` holds one stepper per sub-batch for a whole
query and is what the generators use.

DP-RAG decodes in the same shape — one prompt per retrieved document, all growing
by the shared sampled token — but its exponential mechanism scores the whole
vocabulary rather than the argmax, so it reads ``next_logits`` where the vote
generators read ``next_tokens``.

Two things make the cached path subtle:

* **Left padding.** Rows of unequal length are left-padded so the real last
  token sits at index ``-1`` in every row. The padding then makes the model's
  default position derivation wrong, so ``position_ids`` is always passed
  explicitly: ``cumsum(attention_mask) - 1`` (clamped at 0) at prefill, and the
  previous last position plus 1..s on each incremental step. Absolute-position
  models (GPT-2, OPT) silently produce garbage otherwise.
* **The attention mask must cover the cached prefix**, not just the new tokens,
  so it is kept on the stepper and extended by one column per new token.

Whenever an assumption breaks (a row that is not its previous self plus a
suffix, rows growing by different amounts, a model that rejects the cache) the
stepper silently recomputes the full batch without a cache and rebuilds its
state, so the returned tokens are correct on every path. Setting
``BENCHDP_DISABLE_KV_CACHE=1`` takes that recompute path for every step, and
``BENCHDP_DISABLE_LENGTH_SORT=1`` forwards the rows in the order they arrive.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from core.device import free_memory

# Max prompts per forward pass in the m(+1)-voter vote. Sub-batching bounds the
# peak GPU memory (dominated by the O(batch * seq^2) attention scores); each
# prompt is an independent causal-forward row, so the result is identical to one
# big batch.
VOTER_SUB_BATCH = 16

# Share of the *currently free* VRAM that a query's KV caches may claim, summed
# over all sub-batches alive at once. The caches are not the only thing that has
# to fit: the forward itself needs activations, the attention scores and the
# logits on top of them, and the allocator fragments. Half the free memory for
# state that lives across the whole generation loop leaves the other half for
# the transient per-step peak; when the estimate does not fit, the stepper
# refuses to cache and falls back to the (slower, memory-light) recompute path
# rather than risking an OOM mid-sweep.
CACHE_VRAM_FRACTION = 0.5

#: Set to "1" to force the pre-cache behaviour — every step re-encodes the whole
#: growing prompt with ``use_cache=False``. The escape hatch for A/B-ing the
#: optimisation and for bisecting a suspect run without editing code. Read at
#: generation time (each stepper's construction), not at import, so a parent
#: process that sets it before spawning a method subprocess is honoured.
DISABLE_ENV_VAR = "BENCHDP_DISABLE_KV_CACHE"

#: Set to "1" to take the rows in the order the caller passed them, whatever a
#: generator asked for. Sorting decides which rows share a forward and therefore
#: the order the floating-point reductions run in, so it is one of the knobs an
#: A/B has to be able to hold still (scripts/ab_dp_rag_batching.py). Read when
#: the steppers for a query are built, not at import, so a parent process that
#: sets it before spawning a method subprocess is honoured.
SORT_DISABLE_ENV_VAR = "BENCHDP_DISABLE_LENGTH_SORT"


def kv_cache_disabled() -> bool:
    """Whether ``BENCHDP_DISABLE_KV_CACHE=1`` asks for the uncached path."""
    return os.environ.get(DISABLE_ENV_VAR, "").strip() == "1"


def length_sort_disabled() -> bool:
    """Whether ``BENCHDP_DISABLE_LENGTH_SORT=1`` asks for the rows as they come."""
    return os.environ.get(SORT_DISABLE_ENV_VAR, "").strip() == "1"


def _config_value(config: Any, names: tuple[str, ...], default: int | None = None) -> int | None:
    """First present, non-None attribute of ``config`` among ``names``."""
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return default


def kv_cache_bytes_per_token(model: Any) -> int:
    """Bytes one cached token occupies for one row, or 0 if it can't be derived.

    Keys and values, over every layer and key/value head:
    ``2 * layers * kv_heads * head_dim * itemsize``. Grouped-query attention
    (``num_key_value_heads`` < ``num_attention_heads``) is respected, which is
    the difference between a 4 GiB and a 16 GiB estimate on a 7B model.
    """
    config = getattr(model, "config", None)
    if config is None:
        return 0
    layers = _config_value(config, ("num_hidden_layers", "n_layer", "num_layers"))
    heads = _config_value(config, ("num_attention_heads", "n_head", "num_heads"))
    kv_heads = _config_value(config, ("num_key_value_heads", "num_kv_heads"), heads)
    hidden = _config_value(config, ("hidden_size", "n_embd", "d_model"))
    head_dim = _config_value(config, ("head_dim",))
    if head_dim is None and hidden is not None and heads:
        head_dim = hidden // heads
    if not layers or not kv_heads or not head_dim:
        return 0
    try:
        itemsize = next(model.parameters()).dtype.itemsize
    except StopIteration:
        itemsize = 2
    return 2 * layers * kv_heads * head_dim * itemsize


def total_cache_budget_bytes() -> int | None:
    """KV-cache budget for a whole query, or None when there is nothing to guard.

    Only CUDA is guarded: that is where the 24 GB ceiling bites. On CPU/MPS the
    cache lands in host RAM (125 GB on this fleet) and no budget is imposed.
    """
    if not torch.cuda.is_available():
        return None
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        return None
    return int(free * CACHE_VRAM_FRACTION)


def kv_cache_need_bytes(
    model: Any, batch: int, prompt_len: int, max_new_tokens: int
) -> int | None:
    """Bytes the cache of one sub-batch will hold at the end of generation.

    The projection is to the *end*: ``prompt_len`` grows by one token per step, so
    budgeting for the prompt alone would admit a cache that outgrows its allowance
    halfway through the query. None when the architecture does not disclose enough
    to size a cached token.
    """
    per_token = kv_cache_bytes_per_token(model)
    if per_token <= 0:
        return None
    return int(batch) * (int(prompt_len) + max(0, int(max_new_tokens))) * per_token


def allocate_cache_budgets(needs: list[int | None], total: int | None) -> list[int | None]:
    """Split ``total`` over sub-batches in proportion to what each one needs.

    A query's sub-batches are not interchangeable: DP-RAG sorts its rows by length,
    so the slice of short documents needs a third of what the slice of long ones
    does. Dividing the budget equally therefore starves the long slice while the
    short slices leave most of their allowance unspent — at the shipped fraction,
    eight of nine sub-batches were refused a cache even though all nine caches
    together fit inside the budget.

    Each sub-batch is granted exactly what it asked for, and whatever is left over
    is shared out in proportion, so a later re-prefill on a longer prompt still has
    headroom. The sum of the grants never exceeds ``total`` — the fraction that
    produced it also has to cover the forward's activations, attention scores and
    logits, so overshooting it trades a refusal for an OOM.

    When the needs genuinely do not fit, the cheapest sub-batches are admitted
    first, until the next one no longer fits. That maximises how many sub-batches
    decode incrementally, which is what the wall clock responds to: an admitted
    sub-batch replaces a full re-encode with a one-token step at every one of the
    remaining steps, and a refused one falls back to exactly the path it took
    before. Admitting the expensive slices instead — or refusing the whole query
    because one slice is too big — would leave more of the query on the slow path.

    A need of None (unsizeable architecture) is refused a budget rather than
    guessed at.
    """
    if total is None:
        return [None] * len(needs)
    granted = [0] * len(needs)
    order = sorted(
        (i for i, need in enumerate(needs) if need is not None),
        key=lambda i: (needs[i], i),
    )
    spent = 0
    admitted: list[int] = []
    for i in order:
        need = int(needs[i] or 0)
        if spent + need > total:
            break
        spent += need
        granted[i] = need
        admitted.append(i)
    if admitted and spent > 0:
        # Hand the unclaimed remainder to the admitted sub-batches, proportionally.
        leftover = total - spent
        for i in admitted:
            granted[i] += leftover * granted[i] // spent
    return list(granted)


def _pad_and_mask(
    ids_list: list[list[int]], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad ``ids_list`` to a rectangle and return (input_ids, attention_mask)."""
    max_len = max(len(ids) for ids in ids_list)
    padded, masks = [], []
    for ids in ids_list:
        pad_len = max_len - len(ids)
        padded.append([pad_token_id] * pad_len + list(ids))
        masks.append([0] * pad_len + [1] * len(ids))
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    attn_mask = torch.tensor(masks, dtype=torch.long, device=device)
    return input_ids, attn_mask


def _positions_from_mask(attn_mask: torch.Tensor) -> torch.Tensor:
    """Absolute positions for a left-padded batch: ``cumsum(mask) - 1``, clamped at 0.

    Pad positions are meaningless (they are masked out anyway) but must stay in
    range for the embedding lookup, hence the clamp.
    """
    return (attn_mask.cumsum(-1) - 1).clamp(min=0)


def _call_model(model: Any, kwargs: dict[str, Any]) -> Any:
    """Forward ``model``, dropping kwargs it does not accept.

    ``logits_to_keep=1`` runs the LM head on only the last position instead of
    every position: the vote reads just the last-token logits, so the
    ``[batch, seq, vocab]`` tensor (several GiB for a long context) is never
    materialised. Older/exotic architectures reject it — and some reject
    ``cache_position`` or even ``use_cache`` — so each is peeled off in turn.

    A ``TypeError`` raised from inside the model looks exactly like a rejected
    kwarg here, so the first one is chained onto the final failure: without it a
    genuine bug surfaces as "unsupported signature" after four silent retries.
    """
    first_error: TypeError | None = None
    for drop in ((), ("logits_to_keep",), ("logits_to_keep", "cache_position"),
                 ("logits_to_keep", "cache_position", "use_cache")):
        call_kwargs = {k: v for k, v in kwargs.items() if k not in drop}
        try:
            return model(**call_kwargs)
        except TypeError as exc:
            if first_error is None:
                first_error = exc
            continue
    raise TypeError("model rejected every supported forward signature") from first_error


def greedy_next_logits(
    model: Any, ids_list: list[list[int]], *, pad_token_id: int
) -> torch.Tensor:
    """Uncached last-position logits ``[batch, vocab]`` for pre-tokenised prompts.

    The reference path: left-pad, forward the whole batch with ``use_cache=False``
    and read the last position. Correct for any batch — no assumption about how
    the rows relate to each other.

    The vote generators only need the argmax of this (:func:`greedy_next_tokens`);
    DP-RAG's exponential mechanism scores the whole vocabulary, so the logit
    vectors themselves are what it consumes.
    """
    if not ids_list:
        return torch.empty(0)
    device = next(model.parameters()).device
    input_ids, attn_mask = _pad_and_mask(ids_list, pad_token_id, device)
    with torch.no_grad():
        outputs = _call_model(
            model,
            {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "position_ids": _positions_from_mask(attn_mask),
                "use_cache": False,
                "logits_to_keep": 1,
            },
        )
        return outputs.logits[:, -1, :]


def greedy_next_tokens(
    model: Any, ids_list: list[list[int]], *, pad_token_id: int
) -> list[int]:
    """Uncached greedy next-token (argmax) for one batch of pre-tokenised prompts."""
    if not ids_list:
        return []
    logits = greedy_next_logits(model, ids_list, pad_token_id=pad_token_id)
    return [int(x) for x in torch.argmax(logits, dim=-1).tolist()]


class IncrementalLMStepper:
    """Last-token logits for a fixed batch of prompts that grow by a shared suffix.

    Call :meth:`next_tokens` once per decoding step with the *full* prompt of
    every row. The first call prefills and stores the KV cache; later calls
    forward only the tokens appended since the previous call. The batch size and
    the row order must not change between calls.
    """

    def __init__(
        self,
        model: Any,
        *,
        pad_token_id: int,
        max_new_tokens: int = 0,
        max_cache_bytes: int | None = None,
    ) -> None:
        self.model = model
        self.pad_token_id = int(pad_token_id)
        self.max_new_tokens = max(0, int(max_new_tokens))
        self.max_cache_bytes = max_cache_bytes
        self._cache: Any = None
        self._rows: list[list[int]] | None = None
        self._attn_mask: torch.Tensor | None = None
        self._last_pos: torch.Tensor | None = None
        #: False once caching has been refused (env opt-out, memory guard) or failed.
        self.caching_enabled = not kv_cache_disabled()
        #: Diagnostics: how many steps took the cached path / the recompute path.
        self.n_cached_steps = 0
        self.n_full_steps = 0

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop the cache and all per-batch state (keeps the memory guard verdict)."""
        self._cache = None
        self._rows = None
        self._attn_mask = None
        self._last_pos = None

    def _fits_in_budget(self, batch: int, prompt_len: int) -> bool:
        """Whether the projected KV cache for this batch fits the memory budget."""
        if self.max_cache_bytes is None:
            return True
        need = kv_cache_need_bytes(self.model, batch, prompt_len, self.max_new_tokens)
        if need is None:
            # Unknown architecture: refuse rather than guess when a budget was set.
            return False
        return need <= self.max_cache_bytes

    def next_logits(self, ids_list: list[list[int]]) -> torch.Tensor:
        """Last-position logits ``[rows, vocab]``, using the KV cache when it is valid."""
        if not ids_list:
            return torch.empty(0)
        if self.caching_enabled and self._cache is not None:
            suffixes = self._shared_suffixes(ids_list)
            if suffixes is not None:
                logits = self._step_cached(suffixes)
                if logits is not None:
                    self.n_cached_steps += 1
                    return logits
        return self._full_forward(ids_list)

    def next_tokens(self, ids_list: list[list[int]]) -> list[int]:
        """Greedy next token for every row, using the KV cache when it is valid."""
        if not ids_list:
            return []
        logits = self.next_logits(ids_list)
        return [int(x) for x in torch.argmax(logits, dim=-1).tolist()]

    # ------------------------------------------------------------------

    def _shared_suffixes(self, ids_list: list[list[int]]) -> list[list[int]] | None:
        """Per-row tokens appended since the last call, or None if that isn't what happened.

        Requires: same number of rows, every row is its previous self plus a
        suffix, and every suffix has the same (non-zero) length — the lockstep
        the cache relies on. Anything else returns None and the caller recomputes.
        """
        rows = self._rows
        if rows is None or len(rows) != len(ids_list):
            return None
        suffixes: list[list[int]] = []
        grow = len(ids_list[0]) - len(rows[0])
        if grow <= 0:
            return None
        for prev, new in zip(rows, ids_list):
            if len(new) - len(prev) != grow or new[: len(prev)] != prev:
                return None
            suffixes.append(list(new[len(prev):]))
        return suffixes

    def _step_cached(self, suffixes: list[list[int]]) -> torch.Tensor | None:
        """Forward only the new tokens against the stored cache; None if the model refuses."""
        assert self._attn_mask is not None and self._last_pos is not None
        device = self._attn_mask.device
        n_new = len(suffixes[0])
        new_ids = torch.tensor(suffixes, dtype=torch.long, device=device)
        # Full mask = cached prefix + the new tokens (all real, none padded).
        attn_mask = torch.cat(
            [self._attn_mask, torch.ones((len(suffixes), n_new), dtype=torch.long, device=device)],
            dim=-1,
        )
        # Continue each row's own position counter, which is offset by that row's
        # left padding — never restart from the padded sequence length.
        offsets = torch.arange(1, n_new + 1, dtype=torch.long, device=device).unsqueeze(0)
        position_ids = self._last_pos + offsets
        past_len = int(self._attn_mask.shape[1])
        cache_position = torch.arange(past_len, past_len + n_new, dtype=torch.long, device=device)
        try:
            with torch.no_grad():
                outputs = _call_model(
                    self.model,
                    {
                        "input_ids": new_ids,
                        "attention_mask": attn_mask,
                        "position_ids": position_ids,
                        "past_key_values": self._cache,
                        "cache_position": cache_position,
                        "use_cache": True,
                        "logits_to_keep": 1,
                    },
                )
                last_logits = outputs.logits[:, -1, :]
        except Exception:
            # A model that cannot take the cache back must not break generation.
            self.caching_enabled = False
            self.reset()
            return None
        self._cache = getattr(outputs, "past_key_values", None)
        if self._cache is None:
            self.caching_enabled = False
            self.reset()
            return None
        self._attn_mask = attn_mask
        self._last_pos = position_ids[:, -1:]
        self._rows = [prev + suf for prev, suf in zip(self._rows or [], suffixes)]
        return last_logits

    def _full_forward(self, ids_list: list[list[int]]) -> torch.Tensor:
        """Recompute the whole batch, (re)building the cache when it is allowed."""
        self.reset()
        self.n_full_steps += 1
        max_len = max(len(ids) for ids in ids_list)
        if self.caching_enabled and not self._fits_in_budget(len(ids_list), max_len):
            # The estimate already projects to the end of generation, so a refusal
            # here holds for the rest of the query: stop re-checking it.
            self.caching_enabled = False
        if not self.caching_enabled:
            return greedy_next_logits(self.model, ids_list, pad_token_id=self.pad_token_id)

        device = next(self.model.parameters()).device
        input_ids, attn_mask = _pad_and_mask(ids_list, self.pad_token_id, device)
        position_ids = _positions_from_mask(attn_mask)
        cache_position = torch.arange(max_len, dtype=torch.long, device=device)
        try:
            with torch.no_grad():
                outputs = _call_model(
                    self.model,
                    {
                        "input_ids": input_ids,
                        "attention_mask": attn_mask,
                        "position_ids": position_ids,
                        "cache_position": cache_position,
                        "use_cache": True,
                        "logits_to_keep": 1,
                    },
                )
                last_logits = outputs.logits[:, -1, :]
        except Exception:
            self.caching_enabled = False
            self.reset()
            return greedy_next_logits(self.model, ids_list, pad_token_id=self.pad_token_id)
        cache = getattr(outputs, "past_key_values", None)
        if cache is not None:
            self._cache = cache
            self._rows = [list(ids) for ids in ids_list]
            self._attn_mask = attn_mask
            self._last_pos = position_ids[:, -1:]
        return last_logits


class SubBatchSteppers:
    """One :class:`IncrementalLMStepper` per sub-batch, for the lifetime of a query.

    The generators forward their voters in slices of at most ``sub_batch`` rows.
    The slicing is the same at every decoding step (a fixed-length list sliced at
    fixed offsets), so each slice is itself a lockstep batch and gets its own
    cache. All of those caches are alive at once, so the VRAM budget is split
    between them — in proportion to what each sub-batch's rows actually need, see
    :func:`allocate_cache_budgets`.

    Create one at the start of a query's generation loop and :meth:`close` it on
    every exit path — the caches are per-query state and must never be reused
    across queries.

    ``sort_by_length`` groups rows of similar length into the same sub-batch.
    Every forward pads to its longest row, and attention costs O(rows * seq^2),
    so a slice of a length-heterogeneous batch pays for its longest member on
    every row. Sorting first makes each slice nearly rectangular. It is off by
    default because it changes which rows share a forward, and therefore the
    order the floating-point reductions run in, exactly as the sub-batching
    itself does. ``BENCHDP_DISABLE_LENGTH_SORT=1`` forces it off for a generator
    that asked for it.
    """

    def __init__(
        self,
        model: Any,
        *,
        pad_token_id: int,
        sub_batch: int = VOTER_SUB_BATCH,
        max_new_tokens: int = 0,
        sort_by_length: bool = False,
    ) -> None:
        self.model = model
        self.pad_token_id = int(pad_token_id)
        self.sub_batch = max(1, int(sub_batch))
        self.max_new_tokens = max(0, int(max_new_tokens))
        self.sort_by_length = bool(sort_by_length) and not length_sort_disabled()
        self._steppers: list[IncrementalLMStepper] | None = None

    def _slices(self, rows: list[list[int]]) -> list[list[list[int]]]:
        """``rows`` cut into the fixed sub-batches, in forwarding order."""
        return [rows[i : i + self.sub_batch] for i in range(0, len(rows), self.sub_batch)]

    def _ensure(self, rows: list[list[int]]) -> list[IncrementalLMStepper]:
        """The steppers for these rows, built with a budget sized to each one's slice.

        The rows arrive already in forwarding order, which is what makes a
        need-proportional split possible: only here is it known how long the rows
        of each sub-batch are, and after sorting those lengths differ by a factor
        of several. The budgets are fixed when the steppers are built and not
        revised per step — the need is already projected to the last token, and a
        budget that moved under a stepper would un-stick its refusal.
        """
        slices = self._slices(rows)
        if self._steppers is None or len(self._steppers) != len(slices):
            self.close()
            needs = [
                kv_cache_need_bytes(
                    self.model, len(s), max(len(r) for r in s), self.max_new_tokens
                )
                for s in slices
            ]
            budgets = allocate_cache_budgets(needs, total_cache_budget_bytes())
            self._steppers = [
                IncrementalLMStepper(
                    self.model,
                    pad_token_id=self.pad_token_id,
                    max_new_tokens=self.max_new_tokens,
                    max_cache_bytes=budget,
                )
                for budget in budgets
            ]
        return self._steppers

    def _row_order(self, ids_list: list[list[int]]) -> list[int] | None:
        """Row indices shortest-first, or None when the rows are taken as they come.

        Python's sort is stable and every row grows by the same number of tokens per
        step, so within a query this returns the identical permutation at every step:
        each sub-batch keeps its rows, and therefore its KV cache, across the whole
        generation loop. (If it ever did not, the stepper would see rows that are not
        its previous rows plus a suffix and silently recompute — slower, still correct.)
        """
        if not self.sort_by_length:
            return None
        return sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))

    def next_logits(self, ids_list: list[list[int]]) -> torch.Tensor:
        """Last-position logits ``[len(ids_list), vocab]``, sub-batched and cached.

        Row ``i`` of the result is always the last-position logits of ``ids_list[i]``,
        whatever order the rows were forwarded in. Callers score row ``i`` as their own
        item ``i`` — DP-RAG's exponential mechanism scores it as document ``i`` — so a
        reordering that was not undone would score the wrong document with no error.
        """
        if not ids_list:
            return torch.empty(0)
        order = self._row_order(ids_list)
        rows = ids_list if order is None else [ids_list[i] for i in order]
        steppers = self._ensure(rows)
        chunks = [
            stepper.next_logits(chunk)
            for stepper, chunk in zip(steppers, self._slices(rows))
        ]
        logits = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        if order is None:
            return logits
        # Undo the permutation: forwarded row j carries the logits of input row
        # order[j], so it belongs at output position order[j].
        restored = torch.empty_like(logits)
        restored[torch.tensor(order, dtype=torch.long, device=logits.device)] = logits
        return restored

    def next_tokens(self, ids_list: list[list[int]]) -> list[int]:
        """Greedy next token for every prompt, sub-batched and cached."""
        if not ids_list:
            return []
        logits = self.next_logits(ids_list)
        return [int(x) for x in torch.argmax(logits, dim=-1).tolist()]

    @property
    def n_forwards(self) -> int:
        """Model calls made so far, over every sub-batch and both paths.

        Cached and full steps cost wildly different amounts, but each is one call,
        and the count is what a wider sub-batch buys: it falls with the number of
        sub-batches, not with the number of prompts.
        """
        if not self._steppers:
            return 0
        return sum(s.n_full_steps + s.n_cached_steps for s in self._steppers)

    def close(self) -> None:
        """Free every cache and return the memory to the allocator."""
        if self._steppers:
            for stepper in self._steppers:
                stepper.reset()
        self._steppers = None
        free_memory()
