from __future__ import annotations

import hashlib
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.generator import BaseGenerator
from core.incremental_lm import (
    VOTER_SUB_BATCH,
    SubBatchSteppers,
    greedy_next_tokens,
)
from core.text_gen import to_model_prompt, trim_with_stop_strings
from metrics.privacy_accounting_autodp import AutodpLDAccountant


class NotFoundLDTop1(Exception):
    """Raised when the Limited Domain Gumbel mechanism fails to find a valid top-1 candidate."""
    pass


_VALID_FAIL_MODES = {"stop"}


class VoteRAGGenerator(BaseGenerator):
    """Voting-based differentially private generator (Koga et al., arXiv:2412.04697).

    Implements the Limited Domain Gumbel mechanism (Durfee & Rogers, NeurIPS 2019)
    with paper-faithful prompt construction. Accounting is selectable via
    compose_method: 'rdp' (default; EM-peeling RDP + BBGHS, matching the reference
    runtime) or 'simple' (advanced+sequential composition, the method as literally
    written in the paper — the value set by the shipped benchmark configs).

    fail_mode controls behavior when LD-Gumbel raises NotFoundLDTop1:
      - 'stop' (paper default, arXiv:2412.04697v3 §2.1): "we halt the algorithm, or
        equivalently, regard that it outputs the end of sequence token."
        Generation terminates immediately (exit_status=1).
    """

    #: Cut each context document at its first ``//``, as the reference implementation
    #: does. True is paper-faithful; the shared protocol turns it off so this method
    #: reads the same document text as the six that never inherited the behaviour.
    truncate_at_double_slash: bool = True

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        m: int = 30,
        k: int = 1,
        epsilon_token: float = 2.0,
        delta_token: float = 1e-5,
        epsilon_total: float = 10.0,
        delta_total: float = 1e-4,
        k_bar: int | None = None,
        random_seed: int = 42,
        sensitivity: float = 2.0,
        tmax_absolute: int = 50,
        max_length: int = 50,
        n_peels: int = 1,
        fail_mode: str = "stop",
        verbose: bool = False,
        exclude_stopwords: bool = False,
        exclude_question_words: bool = False,
        stopwords_path: str | None = None,
        task_instruction: str | None = None,
        use_chat_template: bool | str = "auto",
        stop_strings: list[str] | None = None,
        compose_method: str = "rdp",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.m = int(m)
        self.k = int(k)
        self.epsilon_token = float(epsilon_token)
        self.delta_token = float(delta_token)
        self.epsilon_total = float(epsilon_total)
        self.delta_total = float(delta_total)
        self.k_bar = int(k_bar) if k_bar is not None else int(m)
        self.random_seed = int(random_seed)
        self.sensitivity = float(sensitivity)
        # None => paper-faithful: the token count is derived purely from the
        # composition budget (accountant). An int acts only as a safety ceiling.
        self.tmax_absolute = int(tmax_absolute) if tmax_absolute is not None else None
        # DPVOTE-09: fixed generation length (ref `--max_length`, default 50). The loop
        # iterates up to this many steps and the ONLINE budget check stops generation
        # when composition is exhausted, instead of pre-capping the loop at the
        # accountant's (possibly conservative) t_max.
        self.max_length = int(max_length)
        self.n_peels = int(n_peels)
        if fail_mode not in _VALID_FAIL_MODES:
            raise ValueError(
                f"fail_mode must be one of {sorted(_VALID_FAIL_MODES)}, got {fail_mode!r}"
            )
        self.fail_mode = fail_mode
        # Mechanistic step-by-step logging (set per query by the runner). Logging
        # only — never alters the DP math or consumes RNG state.
        self.verbose = bool(verbose)
        self.verbose_label = ""

        self._py_rng = random.Random(self.random_seed)
        self._np_rng = np.random.default_rng(self.random_seed)

        # Per-query KV caches for the vote forwards; only ever non-None inside a
        # generation loop (see _open_steppers / _close_steppers).
        self._steppers: SubBatchSteppers | None = None

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        self.eos_token_id = int(eos_id) if eos_id is not None else 0

        self.vocab_size = int(self.tokenizer.vocab_size)

        # Stopword / question-word exclusion (privacy-neutral: the excluded set is
        # public / data-independent, so removing it from the LD-Gumbel candidate
        # domain is post-processing on public information). Mirrors the sparse
        # variant; stopword token ids are precomputed once.
        self.exclude_stopwords = bool(exclude_stopwords)
        self.exclude_question_words = bool(exclude_question_words)
        self.stopwords_path = stopwords_path
        # Per-dataset task instruction (None => paper-faithful default). Same
        # instruction across methods for a given dataset -> fair comparison.
        self.task_instruction = task_instruction
        # C2 — chat template (auto by default); C3 — optional stop strings (OFF by default).
        self.use_chat_template = use_chat_template
        self.stop_strings = stop_strings
        # DPVOTE-01 — composition method for the accountant. 'simple' = max(advanced,
        # sequential), the method WRITTEN in the paper (footnote 4, Alg.2); 'rdp' = the
        # reference RUNTIME default (EM-peeling RDP), tighter-looking but it drives Tmax
        # to 0 at low epsilon -> empty answers. Paper-faithful headline uses 'simple'.
        self.compose_method = compose_method
        self._stopword_token_ids: set[int] = (
            self._build_stopword_token_ids() if self.exclude_stopwords else set()
        )

        self.accountant = AutodpLDAccountant(
            target_epsilon=self.epsilon_total,
            target_delta=self.delta_total,
            epsilon_token=self.epsilon_token,
            delta_token=self.delta_token,
            m=self.m,
            k=self.k,
            k_bar=self.k_bar,
            sensitivity=self.sensitivity,
            n_peels=self.n_peels,
            compose_method=self.compose_method,
        )

    def _load_stopwords(self) -> list[str]:
        """Load stopwords from data/stopword_words.pt (paper source)."""
        # Resolve relative paths against the project root (parents[3]); the file
        # lives at <root>/data/stopword_words.pt. Using parents[2] (= dp_methods/)
        # silently missed the file and emptied the exclusion set.
        project_root = Path(__file__).resolve().parents[3]
        if self.stopwords_path:
            path = Path(self.stopwords_path)
            if not path.is_absolute():
                path = project_root / path
        else:
            path = project_root / "data" / "stopword_words.pt"
        if path.exists():
            return list(torch.load(path))
        warnings.warn(
            f"stopword_words.pt not found at {path}; stopword exclusion will be empty.",
            RuntimeWarning,
        )
        return []

    def _build_stopword_token_ids(self) -> set[int]:
        """Tokenize stopwords using paper convention: token at index 1 (skipping <bos>)."""
        ids: set[int] = set()
        for word in self._load_stopwords():
            encoded = self.tokenizer(word)["input_ids"]
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            if encoded and isinstance(encoded[0], list):
                encoded = encoded[0]
            if len(encoded) > 1:
                ids.add(int(encoded[1]))
        return ids

    def _tokenize_to_ids(self, text: str) -> list[int]:
        """Tokenize a string to a flat list of int token IDs (no BOS, no return_tensors)."""
        raw = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        return [int(x) for x in raw]

    def _limited_domain_or_fail(
        self,
        histogram: dict[int, int],
        eps: float,
        delta: float,
        k_bar: int | None = None,
        sens: float | None = None,
    ) -> int:
        """Durfee-Rogers Limited Domain Gumbel mechanism (NeurIPS 2019).

        Faithful port of tacchan7412/DPRAG/utils/dp.py::LDGumbelMechanism.get_top1
        (itself a port of DP-OPT). dim = full vocab; the m-D_obs unobserved slots
        in the limited domain act as a competing threshold v_perp_0, not as a
        competing max. On success the output is always a REAL observed token; on
        the output_rand_if_fail failure branch a random UNOBSERVED token is
        returned. Otherwise raises NotFoundLDTop1 (formal privacy failure).
        """
        if k_bar is None:
            k_bar = self.k_bar
        if sens is None:
            sens = self.sensitivity
        if not histogram:
            raise NotFoundLDTop1("Empty histogram: no candidates to vote on.")
        if eps <= 0.0 or delta <= 0.0 or delta >= 1.0:
            raise ValueError(f"Invalid privacy parameters: eps={eps}, delta={delta}")

        sorted_items = sorted(histogram.items(), key=lambda item: item[1], reverse=True)
        real_len = len(sorted_items)
        sorted_counts = np.array([float(c) for _, c in sorted_items], dtype=np.float64)
        sorted_token_ids = [int(t) for t, _ in sorted_items]

        dim = int(self.vocab_size)  # full vocabulary (paper: ld_extend_to_vocab=True)
        gumbel_scale = 1.0 / eps
        output_rand_if_fail = False
        k_bar_local = int(k_bar)

        if k_bar_local < dim:
            # h_perp baseline: count at index k_bar, or 0 if fewer distinct tokens observed
            if k_bar_local + 1 > real_len:
                h_perp = 0.0
            else:
                h_perp = float(sorted_counts[k_bar_local])
            log_arg = min(float(sens), float(k_bar_local), float(dim - k_bar_local)) / delta
            log_term = float(np.log(log_arg)) if log_arg > 0 else 0.0
            h_perp = h_perp + 1.0 + log_term / eps
            v_perp = h_perp + float(self._np_rng.gumbel(0.0, gumbel_scale))

            if k_bar_local > real_len:
                v_perp_0 = float(
                    np.max(self._np_rng.gumbel(0.0, gumbel_scale, size=k_bar_local - real_len))
                )
                if v_perp_0 >= v_perp:
                    v_perp = v_perp_0
                    output_rand_if_fail = True
                k_bar_local = real_len
        else:
            # k_bar == dim → PATE reduction
            if k_bar_local > real_len:
                v_perp = float(
                    np.max(self._np_rng.gumbel(0.0, gumbel_scale, size=k_bar_local - real_len))
                )
                k_bar_local = real_len
                output_rand_if_fail = True
            else:
                v_perp = float("-inf")  # never fails

        if k_bar_local < 1:
            raise ValueError(f"k_bar_real must be >= 1, got {k_bar_local}")

        v_cnts = sorted_counts[:k_bar_local] + self._np_rng.gumbel(
            0.0, gumbel_scale, size=k_bar_local
        )
        v_max_idx = int(np.argmax(v_cnts))
        v_max = float(v_cnts[v_max_idx])

        if v_max >= v_perp:
            return int(sorted_token_ids[v_max_idx])
        if output_rand_if_fail:
            observed = set(sorted_token_ids)
            candidates = [t for t in range(dim) if t not in observed]
            if candidates:
                return int(self._np_rng.choice(candidates))
            return int(self._np_rng.integers(0, dim))
        raise NotFoundLDTop1(f"v_max={v_max:.6f} < v_perp={v_perp:.6f}; mechanism fails.")

    @property
    def _pad_token_id(self) -> int:
        return int(getattr(self.tokenizer, "pad_token_id", None) or 0)

    def _open_steppers(self, max_new_tokens: int) -> None:
        """Start the per-query KV caches used by :meth:`_generate_next_tokens_from_ids`."""
        self._steppers = SubBatchSteppers(
            self.model,
            pad_token_id=self._pad_token_id,
            sub_batch=VOTER_SUB_BATCH,
            max_new_tokens=int(max_new_tokens),
        )

    def _close_steppers(self) -> None:
        """Free the per-query KV caches. Safe to call when none were opened."""
        steppers = getattr(self, "_steppers", None)
        if steppers is not None:
            steppers.close()
        self._steppers = None

    def _generate_next_tokens_from_ids(self, ids_list: list[list[int]]) -> list[int]:
        """Generate one next-token per prompt from pre-tokenized ID lists.

        Prompts are forwarded in sub-batches of at most ``VOTER_SUB_BATCH`` so the
        peak memory of the m+1-voter vote — dominated by the O(batch * seq^2)
        attention scores — stays bounded on one GPU. Each prompt is an independent
        row of a causal forward, so sub-batching is numerically identical to a
        single large batch.

        Inside a generation loop the per-query steppers keep a KV cache per
        sub-batch, so each step forwards only the token appended since the last
        one instead of the whole ~2000-token prompt. Called outside such a loop
        (no steppers open) it falls back to a plain uncached forward.
        """
        if not ids_list:
            return []
        steppers = getattr(self, "_steppers", None)
        if steppers is None:
            return greedy_next_tokens(
                self.model, ids_list, pad_token_id=self._pad_token_id
            )
        return steppers.next_tokens(ids_list)

    def _partition_docs(self, docs: list[str]) -> list[list[str]]:
        docs_copy = list(docs)
        self._py_rng.shuffle(docs_copy)
        return [docs_copy[i * self.k : (i + 1) * self.k] for i in range(self.m)]

    def _context_text(self, doc: str) -> str:
        """The document as it enters the prompt.

        The reference implementation (Koga et al., eval_rag_generator.py) cuts each
        context document at its first ``//``. On the corpora it was written for that
        strips a trailing source marker; on SearchQA snippets and ChatDoctor entries,
        which contain URLs, it truncates the document mid-content. Only this method
        and MuRAG inherit the behaviour, so under the shared protocol it would feed
        two of the eight methods a shorter context than the other six read from the
        same corpus.

        Kept for the paper-faithful configuration and off for the common one.
        """
        return doc.split("//")[0] if self.truncate_at_double_slash else doc

    def _build_prompt(self, query: str, docs: list[str], generated_so_far: str) -> str:
        """Paper-faithful prompt template (Koga et al., eval_rag_generator.py)."""
        if docs:
            ctx_list = [f"{i+1}. {self._context_text(d)}" for i, d in enumerate(docs)]
            instr = self.task_instruction or "Give a simple short answer for the question based on the context"
            instruction = f"Instruction: {instr}\n"
            context = "Context: " + "; ".join(ctx_list) + ".\n"
        else:
            instr = self.task_instruction or "Give a simple short answer for the question"
            instruction = f"Instruction: {instr}\n"
            context = ""
        return instruction + context + f"Question: {query}\nAnswer:{generated_so_far}"

    # ------------------------------------------------------------------
    # Mechanistic verbose logging (logging only — no RNG / no math changes)
    # ------------------------------------------------------------------

    def _decode_token(self, token_id: int) -> str:
        """Decode a single token id to its surface string (whitespace visible)."""
        return self.tokenizer.decode([int(token_id)])

    def _log_vote_step_pre(self, step: int, voter_tokens: list[int], histogram: dict[int, int]) -> None:
        """Print the voters' proposals and the pre-noise top-1 (max raw votes)."""
        print(f"  --- token step {step} ---")
        props = ", ".join(repr(self._decode_token(t)) for t in voter_tokens)
        print(f"    voter proposals (m={len(voter_tokens)} argmax votes): {props}")
        top = sorted(histogram.items(), key=lambda kv: kv[1], reverse=True)[:8]
        if not top:
            print("    vote histogram: (empty — LD-Gumbel will fail and halt)")
            return
        hist_str = ", ".join(f"{repr(self._decode_token(t))}:{c}" for t, c in top)
        print(f"    vote histogram (top): {hist_str}")
        best_id, best_c = top[0]
        print(
            f"    pre-noise winner (max votes): "
            f"{repr(self._decode_token(best_id))} [id={best_id}, votes={best_c}]"
        )

    def _log_vote_step_post(self, histogram: dict[int, int], winning_token: int) -> None:
        """Print the token actually selected by the LD-Gumbel mechanism (post-noise)."""
        top = sorted(histogram.items(), key=lambda kv: kv[1], reverse=True)
        best_id = top[0][0] if top else None
        changed = best_id is None or int(winning_token) != int(best_id)
        note = "   <-- LD-Gumbel noise changed the choice" if changed else ""
        print(
            f"    SELECTED after LD-Gumbel: "
            f"{repr(self._decode_token(winning_token))} [id={int(winning_token)}]{note}"
        )

    def generate(self, prompt: str) -> str:
        _ = prompt
        raise NotImplementedError(
            "VoteRAGGenerator requires calling generate_vote(query, docs) directly."
        )

    def generate_vote(self, query: str, docs: list[str]) -> str:
        query_offset = int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:8], 16)
        query_seed = self.random_seed + query_offset
        self._py_rng = random.Random(query_seed)
        self._np_rng = np.random.default_rng(query_seed)

        self.accountant.reset()
        self.last_run_stats = {
            "exit_status": 0,
            "tokens_generated": 0,
            "n_private_tokens": 0,
            "n_nonprivate_tokens": 0,
            "dp_spent_epsilon": 0.0,
            "dp_spent_delta": 0.0,
            "dp_summary": {},
        }

        if not docs:
            self.last_run_stats["exit_status"] = 1
            return ""

        required_docs = self.m * self.k
        if len(docs) < required_docs:
            raise ValueError(
                f"DPVoteRAG requires len(docs) >= m*k = {required_docs}, got {len(docs)}. "
                "Increase k_retrieved in the runner config or reduce m/k."
            )

        partitions = self._partition_docs(list(docs[:required_docs]))

        generated_tokens: list[int] = []
        exit_status: int = 0
        # Logging-only noise-impact counters: how often LD-Gumbel selected a token
        # other than the pre-noise vote leader (histogram argmax). Read off values
        # the mechanism already computed — no extra RNG, no effect on the output.
        n_noise_flipped = 0

        # Public, data-independent token ids removed from the LD-Gumbel candidate
        # domain (privacy-neutral). Stopwords are precomputed; question words are
        # derived from the (public) query.
        exclude_ids: set[int] = set(self._stopword_token_ids)
        if self.exclude_question_words:
            q_ids = self.tokenizer(query, return_tensors="pt")["input_ids"][0].tolist()
            exclude_ids.update(int(t) for t in q_ids)

        # Tokenize each voter's prompt once; extend with winning token each step (paper mode).
        # C2 — wrap in the chat template for instruct models so each voter answers
        # instead of echoing the scaffold; only generated tokens are decoded below, so
        # the per-token voting is unchanged. No-op for base models.
        prompt_token_ids_list: list[list[int]] = [
            self._tokenize_to_ids(
                to_model_prompt(
                    self.tokenizer,
                    self._build_prompt(query, voter_docs, ""),
                    use_chat_template=self.use_chat_template,
                )
            )
            for voter_docs in partitions
        ]

        if self.verbose:
            print("=" * 30)
            print(f"[DPVoteRAG mechanism] Question: {self.verbose_label}")
            print(
                f"  m={self.m} voters, k={self.k} docs/voter, "
                f"epsilon_token={self.epsilon_token}, k_bar={self.k_bar}"
            )

        # DPVOTE-09: iterate up to a fixed generation length (max_length, ref default
        # 50) and let the ONLINE budget check below (`check_budget()['ok']` /
        # `account_token()`) terminate when composition is exhausted — rather than
        # pre-deriving the loop bound from the accountant's t_max. tmax_absolute, if
        # set, is only an additional safety ceiling on top of max_length.
        loop_cap = self.max_length if self.tmax_absolute is None else min(self.max_length, self.tmax_absolute)
        # One KV cache per voter sub-batch, alive for this query only: every step
        # appends the same winning token to every prompt, so the batch grows in
        # lockstep and only the new token has to be forwarded. Closed on every exit
        # path (EOS, LD-Gumbel failure, budget exhaustion, error).
        self._open_steppers(loop_cap)
        try:
            for step in range(loop_cap):
                if not self.accountant.check_budget()["ok"]:
                    exit_status = 2
                    if self.verbose:
                        print(f"  --- token step {step}: budget exhausted, stopping (exit_status=2) ---")
                    break

                voter_tokens = self._generate_next_tokens_from_ids(prompt_token_ids_list)
                histogram = dict(Counter(voter_tokens))
                if exclude_ids:
                    # Drop excluded (stopword / question) tokens from the candidate
                    # domain before LD-Gumbel. Public set -> privacy-neutral.
                    histogram = {t: c for t, c in histogram.items() if t not in exclude_ids}

                if self.verbose:
                    self._log_vote_step_pre(step, voter_tokens, histogram)

                try:
                    winning_token = self._limited_domain_or_fail(
                        histogram, self.epsilon_token, self.delta_token, k_bar=self.k_bar,
                    )
                except NotFoundLDTop1:
                    # DPVOTE-07: a failed get_top1 still inspected the private histogram, so
                    # we charge it before halting (otherwise epsilon is under-reported on the
                    # failure path). NOTE: this is slightly more conservative than the
                    # reference, which on the fail branch increments only the delta counter
                    # (total_queries), not the epsilon counter (total_k). Our single-counter
                    # accountant folds the failed step into both, over-counting epsilon by
                    # ~one token on the terminal failure — sound (epsilon stays a valid upper
                    # bound) and bounded to a single token (a failure always ends generation).
                    self.accountant.account_token()
                    exit_status = 1
                    if self.verbose:
                        print("    LD-Gumbel FAILED (null/limited-domain miss) -> halt (exit_status=1)")
                    break

                if histogram:
                    pre_noise_id = max(histogram.items(), key=lambda kv: kv[1])[0]
                    n_noise_flipped += int(int(winning_token) != int(pre_noise_id))

                if self.verbose:
                    self._log_vote_step_post(histogram, winning_token)

                if not self.accountant.account_token():
                    exit_status = 2
                    if self.verbose:
                        print(f"  --- token step {step}: budget exhausted on accounting (exit_status=2) ---")
                    break

                if int(winning_token) == self.eos_token_id:
                    if self.verbose:
                        print(f"  --- token step {step}: SELECTED EOS, stopping ---")
                    break

                generated_tokens.append(int(winning_token))
                prompt_token_ids_list = [ids + [int(winning_token)] for ids in prompt_token_ids_list]
        finally:
            self._close_steppers()

        self.last_run_stats = {
            "exit_status": int(exit_status),
            "tokens_generated": len(generated_tokens),
            "n_private_tokens": len(generated_tokens),
            "n_nonprivate_tokens": 0,
            "n_noise_flipped": int(n_noise_flipped),
            "dp_spent_epsilon": float(self.accountant.spent_epsilon),
            "dp_spent_delta": float(self.accountant.spent_delta),
            "dp_summary": self.accountant.get_summary(),
        }

        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # C3 — optional truncation at configured stop strings (OFF by default).
        answer = trim_with_stop_strings(answer, self.stop_strings)
        if self.verbose:
            print(f"  FINAL ANSWER (decoded, {len(generated_tokens)} tokens): {repr(answer)}")
            print("=" * 30)
        return answer

    def generate_with_dp(self, prompts: list[str], epsilon: float, delta: float) -> str:
        _ = (prompts, epsilon, delta)
        raise NotImplementedError(
            "VoteRAGGenerator does not use generate_with_dp(); use generate_vote(query, docs)."
        )
