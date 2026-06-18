from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Any

import numpy as np
import torch

from core.generator import BaseGenerator
from metrics.privacy_accounting_autodp import AutodpLDAccountant


class NotFoundLDTop1(Exception):
    """Raised when the Limited Domain Gumbel mechanism fails to find a valid top-1 candidate."""
    pass


_VALID_FAIL_MODES = {"stop"}


class VoteRAGGenerator(BaseGenerator):
    """Voting-based differentially private generator (Koga et al., arXiv:2412.04697).

    Implements the Limited Domain Gumbel mechanism (Durfee & Rogers, NeurIPS 2019)
    with advanced+sequential composition accounting and paper-faithful prompt construction.

    fail_mode controls behavior when LD-Gumbel raises NotFoundLDTop1:
      - 'stop' (paper default, arXiv:2412.04697v3 §2.1): "we halt the algorithm, or
        equivalently, regard that it outputs the end of sequence token."
        Generation terminates immediately (exit_status=1).
    """

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
        n_peels: int = 1,
        fail_mode: str = "stop",
        verbose: bool = False,
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
        self.tmax_absolute = int(tmax_absolute)
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

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        self.eos_token_id = int(eos_id) if eos_id is not None else 0

        self.vocab_size = int(self.tokenizer.vocab_size)

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
        )

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

    def _generate_next_tokens_from_ids(self, ids_list: list[list[int]]) -> list[int]:
        """Generate one next-token per prompt from pre-tokenized ID lists (batched)."""
        if not ids_list:
            return []
        model_device = next(self.model.parameters()).device

        max_len = max(len(ids) for ids in ids_list)
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0

        padded = []
        attention_masks = []
        for ids in ids_list:
            pad_len = max_len - len(ids)
            padded.append([pad_id] * pad_len + list(ids))
            attention_masks.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(padded, dtype=torch.long).to(model_device)
        attn_mask = torch.tensor(attention_masks, dtype=torch.long).to(model_device)

        with torch.no_grad():
            try:
                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
            except TypeError:
                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits[:, -1, :]
            next_ids = torch.argmax(logits, dim=-1)
        return [int(x) for x in next_ids.tolist()]

    def _partition_docs(self, docs: list[str]) -> list[list[str]]:
        docs_copy = list(docs)
        self._py_rng.shuffle(docs_copy)
        return [docs_copy[i * self.k : (i + 1) * self.k] for i in range(self.m)]

    def _build_prompt(self, query: str, docs: list[str], generated_so_far: str) -> str:
        """Paper-faithful prompt template (Koga et al., eval_rag_generator.py)."""
        if docs:
            ctx_list = [f"{i+1}. {d.split('//')[0]}" for i, d in enumerate(docs)]
            instruction = "Instruction: Give a simple short answer for the question based on the context\n"
            context = "Context: " + "; ".join(ctx_list) + ".\n"
        else:
            instruction = "Instruction: Give a simple short answer for the question\n"
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

        # Tokenize each voter's prompt once; extend with winning token each step (paper mode)
        prompt_token_ids_list: list[list[int]] = [
            self._tokenize_to_ids(self._build_prompt(query, voter_docs, ""))
            for voter_docs in partitions
        ]

        if self.verbose:
            print("=" * 30)
            print(f"[DPVoteRAG mechanism] Question: {self.verbose_label}")
            print(
                f"  m={self.m} voters, k={self.k} docs/voter, "
                f"epsilon_token={self.epsilon_token}, k_bar={self.k_bar}"
            )

        for step in range(self.tmax_absolute):
            if not self.accountant.check_budget()["ok"]:
                exit_status = 2
                if self.verbose:
                    print(f"  --- token step {step}: budget exhausted, stopping (exit_status=2) ---")
                break

            voter_tokens = self._generate_next_tokens_from_ids(prompt_token_ids_list)
            histogram = dict(Counter(voter_tokens))

            if self.verbose:
                self._log_vote_step_pre(step, voter_tokens, histogram)

            try:
                winning_token = self._limited_domain_or_fail(
                    histogram, self.epsilon_token, self.delta_token, k_bar=self.k_bar,
                )
            except NotFoundLDTop1:
                exit_status = 1
                if self.verbose:
                    print("    LD-Gumbel FAILED (null/limited-domain miss) -> halt (exit_status=1)")
                break

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

        self.last_run_stats = {
            "exit_status": int(exit_status),
            "tokens_generated": len(generated_tokens),
            "n_private_tokens": len(generated_tokens),
            "n_nonprivate_tokens": 0,
            "dp_spent_epsilon": float(self.accountant.spent_epsilon),
            "dp_spent_delta": float(self.accountant.spent_delta),
            "dp_summary": self.accountant.get_summary(),
        }

        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if self.verbose:
            print(f"  FINAL ANSWER (decoded, {len(generated_tokens)} tokens): {repr(answer)}")
            print("=" * 30)
        return answer

    def generate_with_dp(self, prompts: list[str], epsilon: float, delta: float) -> str:
        _ = (prompts, epsilon, delta)
        raise NotImplementedError(
            "VoteRAGGenerator does not use generate_with_dp(); use generate_vote(query, docs)."
        )
