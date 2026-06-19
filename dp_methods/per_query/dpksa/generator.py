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

Privacy unit & gap sensitivity (Lemma KSA-1, thesis). Neighboring datastores
``D ~ D'`` differ in exactly one document ``d*`` (the document-level neighborhood
used throughout the thesis). Under the disjoint sample-and-aggregate partition
(each retrieved document feeds exactly one prompt), changing ``d*`` alters at most
one of the ``N`` retrieved documents; since top-``N`` retrieval refills the freed
slot, this **replaces** one of the ``N`` responses, so for every token the count
moves by at most 1 *in either direction* (``||Delta H||_inf <= 1``) and each gap
by at most 2 (``Delta d_k <= 2``). The both-directions move is exactly what makes
the sensitivity 2 (vs 1 in the add-only source algorithm) — see the observation
after Lemma KSA-2 and the factor-of-2 note below; the accountant uses this
``Delta d_k = 2`` (``eps_em = 4/b``). In this benchmark the indexed documents are
the chunks produced by the chunker (``FixedSizeChunker``, with overlap), so the
privacy unit is one chunk: a source document spanning several chunks contributes
several units. Retrieval returns distinct entries, so the 1-document -> 1-response
partition holds.

.. warning::
   **Reproducibility vs. deployment (deterministic noise).** For reproducible
   benchmarking, the per-query RNG is seeded deterministically from
   ``random_seed + hash(query)``, so the FindBestK Gumbel draws and the PTR
   Gaussian draw are a fixed function of the (public) query text and seed. The
   formal (epsilon, delta)-DP guarantee assumes *fresh secret* randomness; with a
   fixed, public seed an adversary who knows the query could in principle predict
   the noise and invert it. This is intentional for the benchmark (deterministic
   re-runs) but is **not** deployment-grade — for any real release, seed the
   mechanism from a cryptographically secure RNG instead.
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
    padded_gap,
    ptr_threshold,
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
        verbose: bool = False,
        task_instruction: str | None = None,
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
        # Fixed, public FindBestK window; default to the paper's k = 1..N-1.
        self.k_max = int(k_max) if k_max is not None else max(1, int(n_docs) - 1)
        self.lowercase = bool(lowercase)
        self.exclude_stopwords = bool(exclude_stopwords)
        # Per-dataset task instruction (None => paper-faithful default). Same
        # instruction across methods for a given dataset -> fair comparison.
        self.task_instruction = task_instruction
        self.random_seed = int(random_seed)
        # Mechanistic step-by-step logging (set per query by the runner). Logging
        # only — never alters the DP math or consumes RNG state.
        self.verbose = bool(verbose)
        self.verbose_label = ""

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
        input_len = int(inputs["input_ids"].shape[1])
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
        # Slice on token ids (the decoded prompt is not always a verbatim prefix).
        gen_ids = outputs[0][input_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def _per_doc_prompt(self, query: str, doc: str) -> str:
        instr = self.task_instruction or "Give a simple short answer for the question based on the context"
        return (
            f"Instruction: {instr}\n"
            f"Context: {doc}\n"
            f"Question: {query}\nAnswer:"
        )

    def _keyword_prompt(self, query: str, keywords: list[str]) -> str:
        instr = self.task_instruction or "Give a simple short answer for the question"
        if keywords:
            return (
                f"Instruction: {instr} using these keywords\n"
                f"Keywords: {', '.join(keywords)}\n"
                f"Question: {query}\nAnswer:"
            )
        return (
            f"Instruction: {instr}\n"
            f"Question: {query}\nAnswer:"
        )

    # ------------------------------------------------------------------
    # DP-KSA mechanism
    # ------------------------------------------------------------------

    def generate_ksa(self, query: str, docs: list[str]) -> str:
        """Run M_KSA(x, D) for one query and return the final answer."""
        # Deterministic per-query seeding for reproducible benchmark runs. NOTE:
        # this makes the DP noise predictable from (query, random_seed); it is
        # benchmark-only and NOT deployment-grade — see the module-level warning.
        query_offset = int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:8], 16)
        self._np_rng = np.random.default_rng(self.random_seed + query_offset)

        usable_docs = [d for d in docs[: self.n_docs] if str(d).strip()]
        n = len(usable_docs)

        if self.verbose:
            print("=" * 30)
            print(f"[DP-KSA mechanism] Question: {self.verbose_label}")
            print(
                f"  n_docs={n}, target_eps={self.target_epsilon}, "
                f"gumbel_scale={self.gumbel_scale:.4f}, sigma={self.sigma:.4f}, "
                f"k window=[{self.k_min},{self.k_max}]"
            )

        # Stage 1: sample-and-aggregate (one short response per document).
        responses = [
            self._generate_answer(self._per_doc_prompt(query, doc), self.max_answer_tokens)
            for doc in usable_docs
        ]

        histogram = build_token_histogram(
            responses, lowercase=self.lowercase, exclude_stopwords=self.exclude_stopwords
        )
        counts, tokens = sorted_counts_and_tokens(histogram)

        if self.verbose:
            print("  Stage 1 — per-doc LLM responses (sample-and-aggregate):")
            for i, resp in enumerate(responses, start=1):
                print(f"    doc {i:>3}: {repr(resp)}")
            top_hist = list(zip(tokens, counts))[:12]
            hist_str = ", ".join(f"{repr(t)}:{c}" for t, c in top_hist)
            print(f"  Stage 2 — token histogram (top): {hist_str}")
            # Noiseless FindBestK reference (argmax gap over the public window), for
            # comparison with the Gumbel-noised choice below. No RNG consumed.
            window = range(max(1, self.k_min), max(max(1, self.k_min), self.k_max) + 1)
            noiseless_k = max(window, key=lambda kk: padded_gap(counts, kk))
            print(
                f"    FindBestK noiseless argmax-k: k={noiseless_k} "
                f"(gap d_k={padded_gap(counts, noiseless_k)})"
            )

        # Stage 2: FindBestK + TopKWithPTR.
        k_hat, d_khat = find_best_k(
            counts,
            gumbel_scale=self.gumbel_scale,
            k_min=self.k_min,
            k_max=self.k_max,
            rng=self._np_rng,
        )

        if self.verbose:
            changed = int(k_hat) != int(noiseless_k)
            note = "   <-- Gumbel noise changed k" if changed else ""
            print(f"    FindBestK noisy choice (Gumbel-max): k_hat={k_hat} (gap d_khat={d_khat}){note}")
        # TopKWithPTR always runs (paper Alg. 1); the only branch is the noisy test.
        # FindBestK ranges over a fixed public window, so degenerate histograms
        # (few/zero distinct tokens) flow through here with gap 0 and fail the test
        # with probability >= 1 - delta_ptr, rather than being special-cased.
        ptr_passed = top_k_with_ptr(
            d_khat, sigma=self.sigma, delta_ptr=self.delta_ptr, rng=self._np_rng
        )
        keywords = tokens[:k_hat] if ptr_passed else []

        if self.verbose:
            thresh = ptr_threshold(self.sigma, self.delta_ptr)
            # The real test (keyword_selector.top_k_with_ptr) is:
            #   release iff  max(2, d_khat) + N(0, (2*sigma)^2) - thresh  >  2.
            # We print the NOISELESS margin; the realized Gaussian draw (not shown,
            # already consumed by the mechanism) decides the printed PASS/FAIL, so a
            # tight margin can legitimately flip the outcome.
            noiseless = max(2.0, float(d_khat)) - thresh
            print(
                f"    TopKWithPTR: noiseless max(2,d_khat)-thresh={noiseless:.3f} "
                f"(d_khat={d_khat}, thresh≈{thresh:.3f}); test = noiseless + N(0,(2σ)²) > 2 "
                f"-> {'PASS (release top-k)' if ptr_passed else 'FAIL (release NO keywords)'}"
            )
            print(f"  Released keywords: {keywords if keywords else '(none — zero-shot fallback)'}")

        # Stage 3: final generation from query + released keywords only.
        answer = self._generate_answer(
            self._keyword_prompt(query, keywords), self.max_final_tokens
        )

        if self.verbose:
            print(f"  FINAL ANSWER (from query + keywords): {repr(answer)}")
            print("=" * 30)

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
