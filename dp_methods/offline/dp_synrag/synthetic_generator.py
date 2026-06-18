"""DP-SynRAG Stage 2d: private token prediction by clipped-logit aggregation.

Reference: arXiv:2510.06719v2, Section "Stage 2: Synthetic Text Generation",
sub-step "Private Prediction".

For each retrieved subset S_r, the generator builds one prompt per document
(p_i = concat(prompt_template(d_i))), runs all prompts through the same LLM,
and at every token step:

    1. Collects the next-token logit vector from each per-document prompt.
    2. Transforms each row via clip_c (paper formula):
           l_exp(t)  = exp(l(t) - max_s l(s))             # numerical stability
           l_cent(t) = l_exp(t) - (max l_exp + min l_exp) / 2
           clip_c(l)(t) = l_cent(t) * min(1, c / ||l_cent||_inf)
    3. Aggregates: z_n = sum_i clip_c(L(p_i, y_<n)).
    4. Samples the next token:
           - tau > 0: y_n ~ softmax(z_n / tau)
           - tau ~ 0: y_n = argmax z_n   (greedy mode)
    5. Repeats for T tokens.

Output: one synthetic text per non-empty cluster S_r.

Performance: O(R * |S_r| * T) forward passes. Padded batching mirrors the
DP-RAG generator (see dp_methods/dp_rag/generator.py:200-234).
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.device import free_memory


class DPSyntheticGenerator:
    """Stage 2d: token-by-token clipped-logit aggregation under DP."""

    DEFAULT_PROMPT_TEMPLATE = (
        "{task_prefix}\n"
        "Example document:\n{document}\n\n"
        "Write a new, similar document:\n"
    )

    def __init__(
        self,
        model_name: str,
        T: int,
        c: float,
        tau: float,
        prompt_template: str | None = None,
        batch_threshold: int = 8,
        device: str = "auto",
        seed: int = 42,
        verbose: bool = False,
        verbose_max_examples: int | None = None,
    ):
        if T <= 0:
            raise ValueError("T must be >= 1.")
        if c <= 0:
            raise ValueError("c must be > 0.")
        if tau <= 1e-6:
            raise ValueError(
                f"tau={tau} <= 1e-6: greedy/argmax sampling does not satisfy the "
                "privacy proof in arXiv:2510.06719v2 Theorem 1 (proof assumes tau > 0). "
                "Set tau > 0 to enable proper privacy accounting.  "
                "If you intend to run in greedy mode, override _sample_token in a subclass "
                "with a separate privacy analysis."
            )

        self.model_name = model_name
        self.T = int(T)
        self.c = float(c)
        self.tau = float(tau)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.batch_threshold = int(batch_threshold)
        self.seed = int(seed)
        self.device = self._resolve_device(device)
        # Mechanistic step-by-step logging for the first `verbose_max_examples`
        # non-empty clusters. Logging only — never alters the DP math or RNG state.
        self.verbose = bool(verbose)
        self.verbose_max_examples = (
            int(verbose_max_examples) if verbose_max_examples is not None else None
        )
        self._verbose_logged = 0

        torch.set_grad_enabled(False)

        # Lazy load: the model is not placed on the device until the first
        # generate_for_clusters() call, so it does not occupy the GPU during the
        # earlier keyword-extraction / embedding stages. unload() frees it again
        # before the self-filter and inference stages load their own model.
        self.tokenizer = None
        self.model = None

        self.last_stats: dict = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate_for_clusters(
        self,
        documents: list[str],
        subsets: list[list[int]],
        task_prefix: str = "",
    ) -> list[str]:
        """Generate one synthetic text per non-empty subset S_r.

        Returns a list of length len(subsets); clusters with empty S_r get
        an empty string so the caller can preserve cluster ordering.
        """
        self._ensure_loaded()
        synthetic: list[str] = []
        n_empty = 0
        for r, members in enumerate(subsets):
            if not members:
                synthetic.append("")
                n_empty += 1
                continue
            prompts = [
                self.prompt_template.format(
                    task_prefix=task_prefix, document=documents[i]
                )
                for i in members
            ]
            log_this = self.verbose and (
                self.verbose_max_examples is None
                or self._verbose_logged < self.verbose_max_examples
            )
            text = self._generate_one(prompts, cluster_index=r, verbose=log_this)
            if log_this:
                self._verbose_logged += 1
            synthetic.append(text)

        self.last_stats = {
            "T": int(self.T),
            "c": float(self.c),
            "tau": float(self.tau),
            "n_clusters": len(subsets),
            "n_empty_clusters": int(n_empty),
            "n_generated": sum(1 for s in synthetic if s),
        }
        return synthetic

    # ------------------------------------------------------------------
    # Per-cluster decoding
    # ------------------------------------------------------------------

    def _decode_token(self, token_id: int) -> str:
        """Decode a single token id to its surface string (whitespace visible)."""
        return self.tokenizer.decode([int(token_id)])

    def _topk_softmax(self, z_n: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """Top-k of softmax(z_n / tau) for display. Draws NO randomness, so it
        cannot perturb the token actually sampled by ``_sample_token``."""
        shifted = z_n / self.tau
        shifted = shifted - float(np.max(shifted))
        probs = np.exp(shifted)
        probs = probs / float(probs.sum())
        top_idx = np.argsort(probs)[::-1][:k]
        return [(int(i), float(probs[i])) for i in top_idx]

    def _log_synth_step(
        self, step: int, logits: np.ndarray, z_n: np.ndarray, next_id: int
    ) -> None:
        """Print the per-token mechanism: per-doc proposals -> pre-noise argmax -> sampled."""
        pre_noise_id = int(np.argmax(z_n))
        proposals = [int(np.argmax(logits[j])) for j in range(logits.shape[0])]
        # Cap the proposal print so large clusters stay readable.
        shown = proposals[:20]
        prop_str = ", ".join(repr(self._decode_token(t)) for t in shown)
        more = f" (+{len(proposals) - len(shown)} more)" if len(proposals) > len(shown) else ""
        print(f"  --- token step {step} ---")
        print(f"    per-doc proposals (argmax next-token, n_docs={logits.shape[0]}): {prop_str}{more}")
        print(
            f"    pre-noise winner (argmax aggregated z_n): "
            f"{repr(self._decode_token(pre_noise_id))} [id={pre_noise_id}]"
        )
        topk = self._topk_softmax(z_n, k=5)
        dist_str = ", ".join(f"{repr(self._decode_token(t))}:{p:.3f}" for t, p in topk)
        print(f"    softmax(z_n/tau) top-5: {dist_str}")
        note = "" if int(next_id) == pre_noise_id else "   <-- DP noise changed the choice"
        print(
            f"    SELECTED after noise: "
            f"{repr(self._decode_token(next_id))} [id={int(next_id)}]{note}"
        )

    def _generate_one(self, prompts: list[str], cluster_index: int, verbose: bool = False) -> str:
        # Truncate from the left so the trailing instruction is preserved
        # (the LLM is conditioned on the most recent context).
        model_max_len = getattr(self.model.config, "max_position_embeddings", 2048)
        max_prompt_len = max(8, model_max_len - self.T)
        encoded = self.tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_prompt_len,
        )
        prompt_ids_list = [
            torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]
        ]

        # Deterministic per-cluster RNG offset.
        cluster_key = f"{self.seed}|{cluster_index}".encode("utf-8")
        cluster_offset = int(hashlib.sha256(cluster_key).hexdigest()[:8], 16)

        eos_token_id = self.tokenizer.eos_token_id
        generated_ids: list[int] = []
        use_batch = len(prompts) <= self.batch_threshold

        if verbose:
            print("=" * 30)
            print(f"[DP-SynRAG mechanism] synthetic doc for cluster {cluster_index}")
            print(f"  n_docs={len(prompts)}, T={self.T}, c={self.c}, tau={self.tau}")

        for step in range(self.T):
            if use_batch:
                logits = self._get_logits_batch(prompt_ids_list, generated_ids)
            else:
                logits = self._get_logits_sequential(prompt_ids_list, generated_ids)

            z_n = self._aggregate_clipped_logits(logits)
            next_id = self._sample_token(
                z_n, seed=self.seed + cluster_offset + step
            )

            if verbose:
                self._log_synth_step(step, logits, z_n, next_id)

            if eos_token_id is not None and next_id == eos_token_id:
                if verbose:
                    print(f"  --- token step {step}: SELECTED EOS, stopping ---")
                break
            generated_ids.append(next_id)

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if verbose:
            print(f"  SYNTHETIC DOC (decoded, {len(generated_ids)} tokens): {repr(text)}")
            print("=" * 30)
        return text

    # ------------------------------------------------------------------
    # Logits
    # ------------------------------------------------------------------

    def _get_logits_batch(
        self, prompt_ids_list: list[torch.Tensor], generated_suffix: list[int]
    ) -> np.ndarray:
        """Returns logits of shape (n_docs, vocab_size) via padded batch."""
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
    # clip_c (paper definition)
    # ------------------------------------------------------------------

    def _clip_c_row(self, logits_row: np.ndarray) -> np.ndarray:
        # Step 1: exponentiate normalized (shift max -> 0 for stability).
        shifted = logits_row - np.max(logits_row)
        l_exp = np.exp(shifted, dtype=np.float64)
        # Step 2: center by midrange.
        center = (l_exp.max() + l_exp.min()) / 2.0
        l_cent = l_exp - center
        # Step 3: L-infinity clip.
        max_abs = float(np.max(np.abs(l_cent)))
        if max_abs <= 1e-12:
            return l_cent.astype(np.float32)
        scale = min(1.0, self.c / max_abs)
        return (l_cent * scale).astype(np.float32)

    def _aggregate_clipped_logits(self, logits: np.ndarray) -> np.ndarray:
        z = np.zeros(logits.shape[1], dtype=np.float64)
        for row in logits:
            z += self._clip_c_row(row)
        return z

    # ------------------------------------------------------------------
    # Token sampling
    # ------------------------------------------------------------------

    def _sample_token(self, z_n: np.ndarray, seed: int) -> int:
        shifted = z_n / self.tau
        shifted -= float(np.max(shifted))
        probs = np.exp(shifted)
        s = float(probs.sum())
        if not np.isfinite(s) or s <= 0:
            return int(np.argmax(z_n))
        probs /= s
        # Final clean-up to dodge tiny FP overflow before np.random.choice.
        probs = np.clip(probs, 0.0, 1.0)
        probs /= probs.sum()
        return int(np.random.default_rng(seed).choice(len(probs), p=probs))

    # ------------------------------------------------------------------
    # Device / model
    # ------------------------------------------------------------------

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _ensure_loaded(self) -> None:
        """Load tokenizer + model on first use (idempotent)."""
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left", truncation_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self._load_model(self.model_name)
        self.model.eval()

    def unload(self) -> None:
        """Release the model from the GPU so the next stage can load its own.

        Idempotent; the model reloads lazily on the next generate_for_clusters().
        """
        self.model = None
        self.tokenizer = None
        free_memory()

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
            return model.to(self.device)
