"""DP-KSA generator (Tang et al., arXiv:2602.14374).

Implements the keyword-selection-and-aggregation mechanism ``M_KSA(x, D)``:

1. **Sample-and-aggregate** — the top-``N`` retrieved documents are each paired
   with the query and independently generate a short response ``y_i = F(D_i, x)``
   (disjoint partition: 1 document -> 1 response).
2. **FindBestK** — privately picks the number of keywords ``k_hat`` from the count
   gaps of the token-frequency histogram (exponential mechanism / Gumbel-max).
3. **TopKWithPTR** — a Gaussian Propose-Test-Release test decides whether the
   exact top-``k_hat`` keywords may be released; on failure no keyword is released.
4. **Final generation** — the answer ``y = F(x, {w_i})`` is produced from the
   query and the privately released keywords only; the database ``D`` is never
   re-accessed.

``epsilon`` is an input budget: the FindBestK Gumbel scale and the PTR noise are
calibrated to the target ``(epsilon, delta)`` via
:mod:`metrics.privacy_accounting_dpksa`.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

from core.generator import BaseGenerator
from dp_methods.per_query.dpksa.keyword_selector import (
    build_token_histogram,
    find_best_k,
    sorted_counts_and_tokens,
    top_k_with_ptr,
)
from metrics.privacy_accounting_dpksa import calibrate_dpksa


class DPKSAGenerator(BaseGenerator):
    """Keyword-selection DP generator (DP-KSA)."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        target_epsilon: float = 3.0,
        target_delta: float = 1e-4,
        n_docs: int = 80,
        max_answer_tokens: int = 16,
        max_final_tokens: int = 32,
        gap_sensitivity: float = 2.0,
        ptr_delta_fraction: float = 0.5,
        ptr_noise_coupling: float = 1.0,
        k_min: int = 1,
        k_max: int | None = None,
        lowercase: bool = True,
        exclude_stopwords: bool = False,
        random_seed: int = 42,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.target_epsilon = float(target_epsilon)
        self.target_delta = float(target_delta)
        self.n_docs = int(n_docs)
        self.max_answer_tokens = int(max_answer_tokens)
        self.max_final_tokens = int(max_final_tokens)
        self.gap_sensitivity = float(gap_sensitivity)
        self.ptr_delta_fraction = float(ptr_delta_fraction)
        self.ptr_noise_coupling = float(ptr_noise_coupling)
        self.k_min = int(k_min)
        self.k_max = int(k_max) if k_max is not None else int(n_docs)
        self.lowercase = bool(lowercase)
        self.exclude_stopwords = bool(exclude_stopwords)
        self.random_seed = int(random_seed)

        self.calibration = calibrate_dpksa(
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            gap_sensitivity=self.gap_sensitivity,
            ptr_delta_fraction=self.ptr_delta_fraction,
            ptr_noise_coupling=self.ptr_noise_coupling,
        )
        self.gumbel_scale = float(self.calibration["gumbel_scale"])
        self.sigma = float(self.calibration["sigma"])
        self.delta_ptr = float(self.calibration["delta_ptr"])

        self._np_rng = np.random.default_rng(self.random_seed)
        self.last_run_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Text generation helpers
    # ------------------------------------------------------------------

    def _generate_answer(self, prompt: str, max_new_tokens: int) -> str:
        """Greedy short-answer generation, returning only the continuation."""
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full[len(prompt):].strip()

    @staticmethod
    def _per_doc_prompt(query: str, doc: str) -> str:
        return (
            "Instruction: Give a simple short answer for the question based on the context\n"
            f"Context: {doc}\n"
            f"Question: {query}\nAnswer:"
        )

    @staticmethod
    def _keyword_prompt(query: str, keywords: list[str]) -> str:
        if keywords:
            return (
                "Instruction: Give a simple short answer for the question using these keywords\n"
                f"Keywords: {', '.join(keywords)}\n"
                f"Question: {query}\nAnswer:"
            )
        return (
            "Instruction: Give a simple short answer for the question\n"
            f"Question: {query}\nAnswer:"
        )

    # ------------------------------------------------------------------
    # DP-KSA mechanism
    # ------------------------------------------------------------------

    def generate_ksa(self, query: str, docs: list[str]) -> str:
        """Run M_KSA(x, D) for one query and return the final answer."""
        query_offset = int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:8], 16)
        self._np_rng = np.random.default_rng(self.random_seed + query_offset)

        usable_docs = [d for d in docs[: self.n_docs] if str(d).strip()]
        n = len(usable_docs)

        # Stage 1: sample-and-aggregate (one short response per document).
        responses = [
            self._generate_answer(self._per_doc_prompt(query, doc), self.max_answer_tokens)
            for doc in usable_docs
        ]

        histogram = build_token_histogram(
            responses, lowercase=self.lowercase, exclude_stopwords=self.exclude_stopwords
        )
        counts, tokens = sorted_counts_and_tokens(histogram)

        # Stage 2: FindBestK + TopKWithPTR.
        k_hat, d_khat = find_best_k(
            counts,
            gumbel_scale=self.gumbel_scale,
            k_min=self.k_min,
            k_max=self.k_max,
            rng=self._np_rng,
        )
        ptr_passed = bool(k_hat >= 1) and top_k_with_ptr(
            d_khat, sigma=self.sigma, delta_ptr=self.delta_ptr, rng=self._np_rng
        )
        keywords = tokens[:k_hat] if ptr_passed else []

        # Stage 3: final generation from query + released keywords only.
        answer = self._generate_answer(
            self._keyword_prompt(query, keywords), self.max_final_tokens
        )

        self.last_run_stats = {
            "n_docs": n,
            "k_hat": int(k_hat),
            "gap": int(d_khat),
            "ptr_passed": ptr_passed,
            "n_keywords": len(keywords),
            "keywords": list(keywords),
            "target_epsilon": self.target_epsilon,
            "achieved_epsilon": float(self.calibration["achieved_epsilon"]),
            "delta": self.target_delta,
            "gumbel_scale": self.gumbel_scale,
            "sigma": self.sigma,
        }
        return answer

    # ------------------------------------------------------------------
    # BaseGenerator interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        _ = prompt
        raise NotImplementedError(
            "DPKSAGenerator requires calling generate_ksa(query, docs) directly."
        )

    def generate_with_dp(self, prompts: list[str], epsilon: float, delta: float) -> str:
        _ = (prompts, epsilon, delta)
        raise NotImplementedError(
            "DPKSAGenerator does not use generate_with_dp(); use generate_ksa(query, docs)."
        )
