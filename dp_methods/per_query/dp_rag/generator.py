"""
DP-ICL Generator: Privacy-preserving generation with DP In-Context Learning.

Reference: Grislain et al., "RAG with Differential Privacy", arXiv:2412.19291v2 (2024)

epsilon is the TOTAL generation budget for one query (not per-token).
The per-token epsilon is computed internally via binary search using PLD composition,
so that composing max_new_tokens exponential mechanisms equals epsilon exactly.
This makes epsilon directly comparable across methods.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.device import model_dtype, warn_if_cuda_oom
from core.generator import BaseGenerator
from core.incremental_lm import (
    VOTER_SUB_BATCH,
    SubBatchSteppers,
    greedy_next_logits,
)
from core.text_gen import chat_template_active, to_model_prompt, trim_with_stop_strings

#: Prompts per forward on the batched path. DP-RAG's decoding has the same shape as
#: the vote generators' — one prompt per retrieved document, every row growing by the
#: same sampled token — so it takes the same sub-batch, and peak memory is bounded by
#: the same two mechanisms: a fixed number of rows per forward, and the KV-cache VRAM
#: guard in :func:`core.incremental_lm.allocate_cache_budgets`, which refuses to cache
#: (and falls back to the memory-light recompute) the sub-batches that do not fit.
#:
#: Retrieved documents are not the vote generators' near-equal prompts: on ChatDoctor a
#: query's 128 selected documents span 114-883 tokens around a mean of ~320. Padding
#: each sub-batch to its longest row therefore costs about half again the tokens the
#: rows actually hold at this width, which is why the steppers are asked to sort the
#: rows by length first — see :class:`core.incremental_lm.SubBatchSteppers`. Sorted, the
#: same width pads by ~8%, and widening it to 32 would cost ~16% for parallelism the
#: forwards (a full re-encode of every prompt, already compute-bound at 16 rows) do not
#: convert into speed.
DOC_SUB_BATCH = VOTER_SUB_BATCH

#: Set to "1" to forward one document per model call, the shape DP-RAG had before the
#: prompts were grouped into sub-batches. The escape hatch for A/B-ing the grouping
#: (scripts/ab_dp_rag_batching.py) and for a machine whose peak memory cannot take
#: :data:`DOC_SUB_BATCH` rows at once. Read at generation time, not at import, so a
#: parent process that sets it before spawning a method subprocess is honoured.
DISABLE_BATCH_ENV_VAR = "BENCHDP_DISABLE_DOC_BATCHING"


def doc_batching_disabled() -> bool:
    """Whether ``BENCHDP_DISABLE_DOC_BATCHING=1`` asks for one forward per document."""
    return os.environ.get(DISABLE_BATCH_ENV_VAR, "").strip() == "1"


class DPICLGenerator(BaseGenerator):
    """
    DP-RAG Generator with DP-ICL: Implements privacy-preserving token generation.

    Privacy mechanism per token:
    - Raw logits transformed: l_norm → center → clip (sensitivity bounded by C)
    - Utility: U_ICL(r) = θ·ln(L_pub(r)) + Σⱼ l_clipped_{j}(r)
    - Token sampled from: r ∝ exp(token_epsilon · U_ICL(r) / (2C))

    The token_epsilon is calibrated via PLD binary search so that
    composing max_new_tokens such mechanisms gives exactly (epsilon, delta)-DP.
    """

    def __init__(
        self,
        model_name: str,
        epsilon: float,
        delta: float = 1e-3,
        max_new_tokens: int = 64,
        theta: float = 0.01,
        temperature: float = 1.0,
        alpha: float = 1.0,
        device: str = "auto",
        seed: int = 42,
        use_batch: bool = True,
        verbose: bool = False,
        use_chat_template: bool | str = "auto",
        stop_strings: list[str] | None = None,
        model: Any = None,
        tokenizer: Any = None,
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            epsilon: TOTAL privacy budget for generation of one query.  The
                per-token epsilon is calibrated internally via PLD binary
                search so that composing max_new_tokens mechanisms gives
                exactly (epsilon, delta)-DP.
            delta: Failure probability for (epsilon, delta)-DP conversion.
                Default 1e-3 matches the paper (Grislain et al., arXiv:2412.19291v2).
                Override via YAML config when a stricter regime is needed.
            max_new_tokens: Maximum tokens to generate.
            theta: Weight for the public document prior in the DP-ICL utility:
                U_ICL(r) = theta·ln(L_pub(r)) + Σ l_clipped_j(r).
                theta=0 ignores the public prior; larger theta pulls the
                output towards the LLM's unconditional prediction.
                Default 0.01 matches the reference (sarus-tech/dp-rag, omega=0.01).
            temperature: Sampling temperature T, matching the reference
                implementation (sarus-tech/dp-rag, dp_model.py).  The L∞
                clipping bound is *derived* from it as
                    C = token_epsilon · T / 2
                (computed after token_epsilon is calibrated).  This is the
                value used both to clip each per-document logit vector (after
                l_norm and centering) and as the sensitivity Δ=C in the
                exponential mechanism exp(token_epsilon·U/(2C)).  Substituting
                C back yields exp(U/T): token_epsilon cancels, so the effective
                sampling distribution is softmax(U/T) — exactly the reference's
                do_sample=True with temperature T.  T=1.0 reproduces the paper
                (arXiv:2412.19291v2); the resulting C is exposed as ``self.C``.
            alpha: Exponent for the l_norm transformation applied to private
                logits: l_norm(r) = (exp(alpha·Δlog_prob) − 1) / alpha.
                alpha=1 → normalised probability; alpha→0 → log-probability
                differences; alpha→∞ → argmax indicator.
            device: "auto", "cuda", or "cpu".
            seed: Random seed for reproducibility.
            use_batch: Forward the per-document prompts in padded sub-batches of
                :data:`DOC_SUB_BATCH` instead of one document per forward. The
                sub-batch — not a cap on how many documents a query may retrieve —
                is what bounds the peak memory of a single forward, so this holds at
                every k_docs. Under the shipped top-p retriever (p=0.02 over a 10-20k
                document corpus, capped at max_docs=128) a query selects 43-129
                documents, which one forward each turns into thousands of sequential
                model calls per query. False forwards one document at a time, as does
                ``BENCHDP_DISABLE_DOC_BATCHING=1``.
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0 for DP generation.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be >= 1.")
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        if alpha <= 0:
            raise ValueError("alpha must be > 0.")

        self.model_name = model_name
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.max_new_tokens = int(max_new_tokens)
        self.theta = float(theta)
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.device = self._resolve_device(device)
        self.seed = int(seed)
        self.use_batch = bool(use_batch)
        # C2 — chat template (auto by default); C3 — optional stop strings (OFF by default).
        self.use_chat_template = use_chat_template
        self.stop_strings = stop_strings
        # Mechanistic step-by-step logging (set per query by the runner). Logging
        # only — never alters the DP math or consumes RNG state.
        self.verbose = bool(verbose)
        self.verbose_label = ""
        # Per-query mechanism stats (populated by generate_with_dp). Logging-only:
        # read off values the mechanism already computed; never alters RNG or math.
        self.last_run_stats: dict[str, Any] = {}
        # KV caches for the current query's decoding loop, held only for the span of
        # one generate_with_dp call (see _open_steppers / _close_steppers).
        self._steppers: SubBatchSteppers | None = None
        # Model calls the last query's decoding loop made, read off the steppers before
        # they are freed. Logging only — it is what the batching is bought with.
        self._last_n_forwards = 0

        torch.set_grad_enabled(False)

        # The model and tokenizer may be supplied by the caller so a single loaded
        # copy of the LLM is reused across many mechanism settings instead of being
        # re-read from disk for each one; otherwise they are loaded here.
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left", truncation_side="left"
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        if model is not None:
            self.model = model
            self.device = str(next(model.parameters()).device)
        else:
            self.model = self._load_model(model_name)
        self.model.eval()

        # Per-token epsilon: calibrated so that T compositions = (epsilon, delta)-DP.
        self.token_epsilon = self._compute_token_epsilon()

        # Clipping bound derived from token_epsilon and temperature, matching the
        # reference (C = token_epsilon·T/2).  Substituting into the exponential
        # mechanism exp(token_epsilon·U/(2C)) gives exp(U/T): token_epsilon
        # cancels, so sampling reduces to softmax(U/T) as in sarus-tech/dp-rag.
        self.C = self.token_epsilon * self.temperature / 2.0

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _load_model(self, model_name: str):
        try:
            dtype = model_dtype()
            kwargs: dict[str, Any] = {"torch_dtype": dtype}
            if self.device == "auto":
                kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            if not hasattr(model, "hf_device_map"):
                model = model.to(self.device)
            return model
        except Exception as exc:
            warn_if_cuda_oom(exc, model_name=model_name, device=self.device)
            self.device = "cpu"
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model = model.to(self.device)
            return model

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _compute_token_epsilon(self, tol: float = 1e-4, max_iter: int = 200) -> float:
        """
        Binary search for per-token epsilon using PLD composition.

        Finds token_eps such that composing max_new_tokens exponential mechanisms,
        each with token_eps, equals (self.epsilon, self.delta)-DP exactly.
        """
        from dp_accounting.pld.privacy_loss_distribution import (
            from_privacy_parameters,
            identity,
        )
        from dp_accounting.pld.common import DifferentialPrivacyParameters

        def composed_epsilon(token_eps: float) -> float:
            pld = identity()
            single = from_privacy_parameters(
                DifferentialPrivacyParameters(epsilon=token_eps)
            )
            for _ in range(self.max_new_tokens):
                pld = pld.compose(single)
            return pld.get_epsilon_for_delta(self.delta)

        # `lo` is kept on the side that composes within the budget and `hi` on the side
        # that overruns it, and `lo` is what comes back. Accepting whichever endpoint
        # happened to land within a tolerance of the target returns a per-token budget
        # whose composition can sit *above* it: the run then reports a guarantee it did
        # not honour, and on a shared axis that is a cell labelled 50 delivering 50.02.
        # Undershooting only leaves budget unspent, which is the safe direction.
        # The tolerance is relative to the budget being solved for, not absolute: a
        # tight cell's per-token budget is a few thousandths, and an absolute 1e-3 there
        # is an 18% error that leaves most of the allowance unspent.
        lo, hi = 0.0, self.epsilon
        for _ in range(max_iter):
            if hi - lo < tol * hi:
                break
            mid = (lo + hi) / 2
            if composed_epsilon(mid) <= self.epsilon:
                lo = mid
            else:
                hi = mid
        return lo

    # ------------------------------------------------------------------
    # Logit computation
    # ------------------------------------------------------------------

    @property
    def _pad_token_id(self) -> int:
        return int(getattr(self.tokenizer, "pad_token_id", None) or 0)

    def _sub_batch(self) -> int:
        """Prompts per forward for this query: :data:`DOC_SUB_BATCH`, or 1 sequentially.

        The environment is read here rather than at construction so a parent process
        that sets the switch before spawning a method subprocess is honoured.
        """
        if not self.use_batch or doc_batching_disabled():
            return 1
        return DOC_SUB_BATCH

    def _open_steppers(self, sub_batch: int) -> None:
        """Start the per-query KV caches used by :meth:`_get_logits`.

        ``sub_batch`` is how many prompts share one forward: :data:`DOC_SUB_BATCH` on
        the batched path, one document per forward on the sequential one.
        """
        self._last_n_forwards = 0
        self._steppers = SubBatchSteppers(
            self.model,
            pad_token_id=self._pad_token_id,
            sub_batch=int(sub_batch),
            max_new_tokens=self.max_new_tokens,
            sort_by_length=True,
        )

    def _close_steppers(self) -> None:
        """Free the per-query KV caches. Safe to call when none were opened."""
        steppers = getattr(self, "_steppers", None)
        if steppers is not None:
            self._last_n_forwards = steppers.n_forwards
            steppers.close()
        self._steppers = None

    @staticmethod
    def _current_ids(
        prompt_ids_list: list[torch.Tensor], generated_suffix: list[int]
    ) -> list[list[int]]:
        """Each document's prompt followed by the tokens sampled so far."""
        suffix = [int(t) for t in generated_suffix]
        return [
            [int(t) for t in base_ids.tolist()] + suffix for base_ids in prompt_ids_list
        ]

    def _get_logits(
        self, prompt_ids_list: list[torch.Tensor], generated_suffix: list[int]
    ) -> np.ndarray:
        """Last-token logits of shape (k_docs, vocab_size), one row per prompt.

        Every row is its previous self plus the single token the exponential
        mechanism sampled, so inside a generation loop the per-query steppers keep
        a KV cache and each step forwards only that one token instead of the whole
        prompt again. Called outside such a loop (no steppers open) it falls back
        to a plain uncached forward of the full prompts.
        """
        ids_list = self._current_ids(prompt_ids_list, generated_suffix)
        steppers = getattr(self, "_steppers", None)
        if steppers is None:
            logits = greedy_next_logits(
                self.model, ids_list, pad_token_id=self._pad_token_id
            )
        else:
            logits = steppers.next_logits(ids_list)
        return logits.float().detach().cpu().numpy()

    def _get_logits_batch(
        self, prompt_ids_list: list[torch.Tensor], generated_suffix: list[int]
    ) -> np.ndarray:
        """Uncached (k_docs, vocab_size) logits from one padded batch.

        Left padding puts every row's real last token at index -1, which is what
        makes ``logits_to_keep=1`` usable and keeps the explicit ``position_ids``
        (``cumsum(mask) - 1``) aligned with each row's own token positions —
        absolute-position models (GPT-2, OPT) read padding as real positions
        otherwise.
        """
        ids_list = self._current_ids(prompt_ids_list, generated_suffix)
        logits = greedy_next_logits(
            self.model, ids_list, pad_token_id=self._pad_token_id
        )
        return logits.float().detach().cpu().numpy()

    def _get_logits_sequential(
        self, prompt_ids_list: list[torch.Tensor], generated_suffix: list[int]
    ) -> np.ndarray:
        """Uncached (k_docs, vocab_size) logits, one document per forward.

        Batch-1 rows need no padding, so this trades the batched path's parallelism
        for a peak memory that does not grow with the number of documents.
        """
        ids_list = self._current_ids(prompt_ids_list, generated_suffix)
        rows = [
            greedy_next_logits(self.model, [ids], pad_token_id=self._pad_token_id)
            .float()
            .detach()
            .cpu()
            .numpy()[0]
            for ids in ids_list
        ]
        return np.stack(rows, axis=0)

    # ------------------------------------------------------------------
    # DP-ICL logit transformations
    # ------------------------------------------------------------------

    def _transform_l_norm(self, logits: np.ndarray) -> np.ndarray:
        """l_norm(r) = (exp(α·(ln L − ln max L)) − 1) / α"""
        logits_shifted = logits - np.max(logits)
        if self.alpha < 1e-6:
            return logits_shifted.astype(np.float32)
        return ((np.exp(self.alpha * logits_shifted) - 1.0) / self.alpha).astype(
            np.float32
        )

    def _center_logits(self, l_norm_logits: np.ndarray) -> np.ndarray:
        """l_centered(r) = l_norm(r) − (max + min) / 2"""
        center = (np.max(l_norm_logits) + np.min(l_norm_logits)) / 2.0
        return (l_norm_logits - center).astype(np.float32)

    def _clip_logits(self, l_centered: np.ndarray) -> np.ndarray:
        """L∞ global clipping: l_clipped = l_centered · min(1, C / max|l_centered|)"""
        max_abs = np.max(np.abs(l_centered))
        if max_abs <= 1e-8:
            return l_centered.astype(np.float32)
        return (l_centered * min(1.0, self.C / max_abs)).astype(np.float32)

    def _compute_utility_icl(
        self, logits_pub: np.ndarray, logits_per_doc: list[np.ndarray]
    ) -> np.ndarray:
        """U_ICL(r) = θ · ln(L_pub(r)) + Σⱼ l_clipped_{j}(r)"""
        private_utility = np.zeros_like(logits_pub, dtype=np.float32)
        for l_clipped_j in logits_per_doc:
            private_utility += l_clipped_j
        return (self.theta * logits_pub + private_utility).astype(np.float32)

    def _sample_token(self, utilities: np.ndarray, seed: int) -> int:
        """
        Sample token using the exponential mechanism.

        r ∝ exp(token_epsilon · U_ICL(r) / (2C))
        Each call is token_epsilon-DP (sensitivity Δ = C from clipping).
        """
        utilities_shifted = utilities - np.max(utilities)
        exp_weights = np.exp(
            (self.token_epsilon * utilities_shifted) / (2.0 * self.C)
        )
        probabilities = exp_weights / np.sum(exp_weights)
        probabilities = np.clip(probabilities, 0.0, 1.0)
        probabilities /= np.sum(probabilities)
        return int(np.random.default_rng(seed).choice(len(utilities), p=probabilities))

    # ------------------------------------------------------------------
    # Mechanistic verbose logging (logging only — no RNG / no math changes)
    # ------------------------------------------------------------------

    def _decode_token(self, token_id: int) -> str:
        """Decode a single token id to its surface string (whitespace visible)."""
        return self.tokenizer.decode([int(token_id)])

    def _topk_exp_mech_probs(self, utilities: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """Top-k of the exponential-mechanism sampling distribution (for display).

        Recomputes the same softmax(token_epsilon*U/(2C)) used by ``_sample_token``
        but draws NO randomness, so it cannot perturb the actual sampled token.
        """
        utilities_shifted = utilities - np.max(utilities)
        exp_weights = np.exp((self.token_epsilon * utilities_shifted) / (2.0 * self.C))
        probs = exp_weights / np.sum(exp_weights)
        top_idx = np.argsort(probs)[::-1][:k]
        return [(int(i), float(probs[i])) for i in top_idx]

    def _log_step(
        self,
        step: int,
        all_logits: np.ndarray,
        logits_pub: np.ndarray,
        utility_icl: np.ndarray,
        next_token_id: int,
        k_docs: int,
    ) -> None:
        """Print the per-token mechanism: proposals -> pre-noise winner -> sampled."""
        pre_noise_id = int(np.argmax(utility_icl))
        print(f"  --- token step {step} ---")
        if k_docs > 1:
            proposals = [int(np.argmax(all_logits[j])) for j in range(k_docs - 1)]
            prop_str = ", ".join(repr(self._decode_token(t)) for t in proposals)
            print(f"    LLM proposals (argmax next-token per private doc): {prop_str}")
            print(f"    public-prior top: {repr(self._decode_token(int(np.argmax(logits_pub))))}")
        else:
            print(f"    single-prompt top: {repr(self._decode_token(int(np.argmax(logits_pub))))}")
        print(
            f"    pre-noise winner (argmax U_ICL): "
            f"{repr(self._decode_token(pre_noise_id))} [id={pre_noise_id}]"
        )
        topk = self._topk_exp_mech_probs(utility_icl, k=5)
        dist_str = ", ".join(f"{repr(self._decode_token(t))}:{p:.3f}" for t, p in topk)
        print(f"    exp-mech top-5 probs: {dist_str}")
        note = "" if int(next_token_id) == pre_noise_id else "   <-- DP noise changed the choice"
        print(
            f"    SELECTED after noise: "
            f"{repr(self._decode_token(next_token_id))} [id={int(next_token_id)}]{note}"
        )

    # ------------------------------------------------------------------
    # Generation interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Entry point: if prompt is a JSON list, delegates to generate_with_dp."""
        try:
            parsed = json.loads(prompt)
            if isinstance(parsed, list):
                return self.generate_with_dp(prompt)
        except (json.JSONDecodeError, TypeError):
            pass

        templated = chat_template_active(self.tokenizer, self.use_chat_template)
        model_prompt = to_model_prompt(self.tokenizer, prompt, use_chat_template=self.use_chat_template)
        inputs = self.tokenizer(
            model_prompt, return_tensors="pt", add_special_tokens=not templated
        ).to(self._model_device())
        input_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = outputs[0][input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return trim_with_stop_strings(text, self.stop_strings)

    def generate_with_dp(self, prompts_json: Any) -> str:
        """
        DP-ICL generation from k+1 prompts (k private + 1 public).

        Private prompts: indices 0..k-1 (one per retrieved document).
        Public prompt: index k (used as L_pub prior, weighted by θ).

        For each token:
        1. Get raw logits from all k+1 prompts
        2. Private logits: l_norm → center → clip
        3. Utility: θ·logits_pub + Σ l_clipped
        4. Sample: r ∝ exp(token_epsilon · U / (2C))
        """
        prompts = json.loads(prompts_json) if isinstance(prompts_json, str) else prompts_json
        prompts = [str(p) for p in prompts if str(p).strip()]
        k_docs = len(prompts)

        if k_docs == 0:
            return ""

        # Query-level seed derived from the run seed together with the PUBLIC prompt
        # (prompts[-1] is the public, non-private prompt = PUBLIC_TEMPLATE.format(query)).
        # Keeps the noise reproducible per (seed, query, step) WITHOUT seeding from the
        # private document text.
        #
        # The seed goes through the hash rather than being added to the offset. Under
        # `seed + offset + step` two runs seeded 1 and 2 walk the same integers one step
        # apart, so at 70 decoding steps they share 68 of their 70 draws: a sweep meant
        # to measure the mechanism's variability would report almost none of it. The
        # same arithmetic also let two queries whose offsets differ by less than the
        # step count share draws, correlating releases the composition assumes are
        # independent.
        public_prompt = prompts[-1] if prompts else ""
        query_key = f"{self.seed}|{public_prompt}".encode("utf-8")
        query_offset = int(hashlib.sha256(query_key).hexdigest()[:8], 16)

        # Truncate prompts to fit within the model's context window.
        # Without this, retrieved documents can exceed max_position_embeddings,
        # causing quadratic attention buffers that are physically impossible to
        # allocate (e.g. 234 GB for a 2746-token sequence on TinyLlama-2048).
        # We truncate from the LEFT (truncation_side="left" on the tokenizer)
        # so the query at the end of each prompt is always preserved; only the
        # beginning of the document context is dropped when necessary.
        # We reserve max_new_tokens slots so the generation loop can always
        # append tokens without going out of bounds.
        model_max_len = getattr(self.model.config, "max_position_embeddings", 2048)
        max_prompt_len = model_max_len - self.max_new_tokens
        # C2 — wrap each per-doc / public prompt in the model's chat template so an
        # instruct model produces answer logits instead of echoing the scaffold. The
        # DP-ICL math (logits -> l_norm -> center -> clip -> sample) is unchanged; only
        # the input encoding differs. No-op for base models (gpt2/opt/pythia).
        templated = chat_template_active(self.tokenizer, self.use_chat_template)
        model_prompts = [
            to_model_prompt(self.tokenizer, p, use_chat_template=self.use_chat_template)
            for p in prompts
        ]
        encoded = self.tokenizer(
            model_prompts,
            add_special_tokens=not templated,
            truncation=True,
            max_length=max_prompt_len,
        )
        prompt_ids_list = [
            torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]
        ]

        generated_ids: list[int] = []
        eos_token_id = self.tokenizer.eos_token_id
        # Logging-only noise-impact counters: how often the exponential mechanism
        # selected a token other than the pre-noise argmax(U_ICL). Read off values
        # already computed below — no extra RNG, no effect on the sampled token.
        n_steps = 0
        n_noise_flipped = 0

        if self.verbose:
            print("=" * 30)
            print(f"[DP-RAG mechanism] Question: {self.verbose_label}")
            print(
                f"  k_docs={k_docs} (k-1 private + 1 public), "
                f"token_epsilon={self.token_epsilon:.4f}, C={self.C:.4f}, theta={self.theta}"
            )

        # Every prompt grows by the same sampled token, which is what lets the steppers
        # cache the unchanged prefix and what makes the k+1 prompts one lockstep batch.
        # Grouping them ceil(k_docs / sub_batch) to a forward is the difference between
        # a handful of model calls per token and one per retrieved document.
        self._open_steppers(self._sub_batch())
        try:
            for step in range(self.max_new_tokens):
                all_logits = self._get_logits(prompt_ids_list, generated_ids)

                if k_docs == 1:
                    logits_pub = all_logits[0]
                    clipped_logits_list = []
                else:
                    # Private: first k-1 prompts; public: last prompt
                    logits_pub = all_logits[-1]
                    clipped_logits_list = [
                        self._clip_logits(
                            self._center_logits(self._transform_l_norm(lg))
                        )
                        for lg in all_logits[:-1]
                    ]

                utility_icl = self._compute_utility_icl(logits_pub, clipped_logits_list)
                next_token_id = self._sample_token(utility_icl, seed=query_offset + step)

                n_steps += 1
                n_noise_flipped += int(next_token_id != int(np.argmax(utility_icl)))

                if self.verbose:
                    self._log_step(
                        step, all_logits, logits_pub, utility_icl, next_token_id, k_docs
                    )

                if eos_token_id is not None and next_token_id == eos_token_id:
                    if self.verbose:
                        print(f"  --- token step {step}: SELECTED EOS, stopping ---")
                    break

                generated_ids.append(next_token_id)
        finally:
            self._close_steppers()

        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        # C3 — optional truncation at configured stop strings (OFF by default).
        answer = trim_with_stop_strings(answer, self.stop_strings)
        self.last_run_stats = {
            "n_tokens": int(n_steps),
            "n_noise_flipped": int(n_noise_flipped),
            "n_forwards": int(self._last_n_forwards),
            "n_docs": int(k_docs),
            "token_epsilon": float(self.token_epsilon),
            "C": float(self.C),
        }
        if self.verbose:
            print(f"  FINAL ANSWER (decoded): {repr(answer)}")
            print("=" * 30)
        return answer


# Backward-compatible alias used by experiments and older code paths.
DPAggregatedGenerator = DPICLGenerator
