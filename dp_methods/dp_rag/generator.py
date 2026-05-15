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
from typing import Any, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.generator import BaseGenerator


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
        delta: float = 1e-5,
        max_new_tokens: int = 64,
        theta: float = 1.0,
        C: float = 1.0,
        alpha: float = 1.0,
        device: str = "auto",
        seed: int = 42,
        use_batch: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            epsilon: TOTAL privacy budget for generation of one query.  The
                per-token epsilon is calibrated internally via PLD binary
                search so that composing max_new_tokens mechanisms gives
                exactly (epsilon, delta)-DP.
            delta: Failure probability for (epsilon, delta)-DP conversion.
                **Note:** the paper (Grislain et al., arXiv:2412.19291v2)
                uses delta=1e-3 in its evaluation.  This constructor defaults
                to 1e-5 (a stricter but common ML convention).  Always pass
                delta explicitly, or set it in the YAML config, to match the
                desired privacy regime.  The runner will print a warning when
                verbose=True and delta differs from the paper value.
            max_new_tokens: Maximum tokens to generate.
            theta: Weight for the public document prior in the DP-ICL utility:
                U_ICL(r) = theta·ln(L_pub(r)) + Σ l_clipped_j(r).
                theta=0 ignores the public prior; larger theta pulls the
                output towards the LLM's unconditional prediction.
            C: Clipping parameter — L∞ norm bound applied to each per-document
                logit vector after l_norm and centering.  Sensitivity of
                U_ICL w.r.t. one document is Δ=C, which appears in the
                exponential mechanism as exp(ε·U/(2C)).
                **Note on temperature:** the reference implementation
                (sarus-tech/dp-rag, dp_model.py) introduces an extra
                ``temperature`` knob that scales the clipping as
                C_ref = token_epsilon·T/2 and then delegates sampling to
                HuggingFace's do_sample=True with that temperature.  This is
                mathematically equivalent to the paper's formula — temperature
                is absorbed into C and does not appear in arXiv:2412.19291v2.
                We therefore do not expose a separate temperature parameter.
            alpha: Exponent for the l_norm transformation applied to private
                logits: l_norm(r) = (exp(alpha·Δlog_prob) − 1) / alpha.
                alpha=1 → normalised probability; alpha→0 → log-probability
                differences; alpha→∞ → argmax indicator.
            device: "auto", "cuda", or "cpu".
            seed: Random seed for reproducibility.
            use_batch: Use padded batch inference when k_docs <= 8.
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0 for DP generation.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be >= 1.")
        if C <= 0:
            raise ValueError("C (clipping parameter) must be > 0.")
        if alpha <= 0:
            raise ValueError("alpha must be > 0.")

        self.model_name = model_name
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.max_new_tokens = int(max_new_tokens)
        self.theta = float(theta)
        self.C = float(C)
        self.alpha = float(alpha)
        self.device = self._resolve_device(device)
        self.seed = int(seed)
        self.use_batch = bool(use_batch)

        torch.set_grad_enabled(False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", truncation_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._load_model(model_name)
        self.model.eval()

        # Per-token epsilon: calibrated so that T compositions = (epsilon, delta)-DP.
        self.token_epsilon = self._compute_token_epsilon()

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
            dtype = torch.float32 if self.device == "mps" else torch.float16
            kwargs: dict[str, Any] = {"torch_dtype": dtype}
            if self.device == "auto":
                kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            if not hasattr(model, "hf_device_map"):
                model = model.to(self.device)
            return model
        except Exception:
            self.device = "cpu"
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model = model.to(self.device)
            return model

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _compute_token_epsilon(self, tol: float = 1e-3, max_iter: int = 100) -> float:
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

        lo, hi = 0.0, self.epsilon
        for _ in range(max_iter):
            if (hi - lo) / 2 < tol:
                break
            mid = (lo + hi) / 2
            eps_mid = composed_epsilon(mid)
            if abs(eps_mid - self.epsilon) < tol:
                return mid
            if (composed_epsilon(lo) - self.epsilon) * (eps_mid - self.epsilon) < 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    # ------------------------------------------------------------------
    # Logit computation
    # ------------------------------------------------------------------

    def _get_logits_batch(
        self, prompt_ids_list: list[torch.Tensor], generated_suffix: list[int]
    ) -> np.ndarray:
        """Returns logits of shape (k_docs, vocab_size) via padded batch."""
        suffix = (
            torch.tensor(generated_suffix, dtype=torch.long)
            if generated_suffix
            else None
        )
        current_ids: list[torch.Tensor] = []
        for base_ids in prompt_ids_list:
            if suffix is not None and suffix.numel() > 0:
                current_ids.append(torch.cat([base_ids, suffix], dim=0))
            else:
                current_ids.append(base_ids)

        padded = torch.nn.utils.rnn.pad_sequence(
            current_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = (padded != self.tokenizer.pad_token_id).long()

        device = self._model_device()
        padded = padded.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=padded, attention_mask=attention_mask)
            logits = outputs.logits

        last_positions = attention_mask.sum(dim=1) - 1
        row_ids = torch.arange(logits.size(0), device=logits.device)
        last_logits = logits[row_ids, last_positions, :]
        return last_logits.float().detach().cpu().numpy()

    def _get_logits_sequential(
        self, prompt_ids_list: list[torch.Tensor], generated_suffix: list[int]
    ) -> np.ndarray:
        """Returns logits of shape (k_docs, vocab_size) sequentially."""
        suffix = (
            torch.tensor(generated_suffix, dtype=torch.long)
            if generated_suffix
            else None
        )
        all_logits = []
        device = self._model_device()

        for base_ids in prompt_ids_list:
            current_ids = (
                torch.cat([base_ids, suffix], dim=0)
                if suffix is not None and suffix.numel() > 0
                else base_ids
            )
            input_ids = current_ids.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :]
            all_logits.append(logits.float().detach().cpu().numpy())

        return np.stack(all_logits, axis=0)

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

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._model_device())
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_output[len(prompt):].strip()

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

        # Query-level seed: same prompts → same noise; different prompts → different noise
        query_key = "\n||\n".join(prompts).encode("utf-8")
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
        encoded = self.tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_prompt_len,
        )
        prompt_ids_list = [
            torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]
        ]

        generated_ids: list[int] = []
        eos_token_id = self.tokenizer.eos_token_id

        for step in range(self.max_new_tokens):
            if self.use_batch and k_docs <= 8:
                all_logits = self._get_logits_batch(prompt_ids_list, generated_ids)
            else:
                all_logits = self._get_logits_sequential(prompt_ids_list, generated_ids)

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
            next_token_id = self._sample_token(
                utility_icl, seed=self.seed + query_offset + step
            )

            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            generated_ids.append(next_token_id)

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


# Backward-compatible alias used by experiments and older code paths.
DPAggregatedGenerator = DPICLGenerator
