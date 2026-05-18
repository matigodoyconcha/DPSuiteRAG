"""DP-SynRAG Stage 2e: post-hoc self-filter on synthetic texts.

Reference: arXiv:2510.06719v2, Section "Stage 2: Synthetic Text Generation",
sub-step "Self-filtering".

The LLM is asked a YES/NO question about each synthetic document. Texts whose
answer starts with "YES" are kept; the rest are dropped. The paper uses this
to discard Medical Synth / Movielens samples that lack the task-critical
information (a disease name, a movie title). It is *not* applied to SearchQA.

Privacy:
    This stage operates entirely on the outputs y_r of Stage 2d, which are
    already (epsilon, delta)-DP. Post-processing of a DP output preserves
    privacy, so the filter consumes no additional budget.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SelfFilter:
    """Stage 2e: YES/NO LLM filter applied to each synthetic text."""

    DEFAULT_PROMPT_TEMPLATE = (
        "Read the following document and answer the question with only YES or NO.\n\n"
        "Document:\n{text}\n\n"
        "Question: Does this document contain task-relevant information?\n"
        "Answer:"
    )

    def __init__(
        self,
        model_name: str,
        prompt_template: str | None = None,
        max_new_tokens: int = 4,
        device: str = "auto",
        seed: int = 42,
    ):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be >= 1.")

        self.model_name = model_name
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.max_new_tokens = int(max_new_tokens)
        self.seed = int(seed)
        self.device = self._resolve_device(device)
        torch.set_grad_enabled(False)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self._load_model(model_name)
        self.model.eval()

        self.last_stats: dict = {}

    def filter(self, synthetic_texts: list[str]) -> list[bool]:
        """Return a list of booleans (True = keep) aligned with synthetic_texts.

        Empty strings are automatically discarded (kept=False) without an LLM call.
        """
        kept: list[bool] = []
        n_calls = 0
        for text in synthetic_texts:
            if not text or not text.strip():
                kept.append(False)
                continue
            answer = self._yes_no(text)
            kept.append(answer)
            n_calls += 1
        self.last_stats = {
            "n_inputs": len(synthetic_texts),
            "n_calls": n_calls,
            "n_kept": int(sum(kept)),
        }
        return kept

    def _yes_no(self, text: str) -> bool:
        prompt = self.prompt_template.format(text=text)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(
            self._model_device()
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt) :]
        return decoded.strip().upper().startswith("YES")

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
