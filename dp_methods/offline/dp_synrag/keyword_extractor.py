"""DP-SynRAG Stage 1a: LLM-based per-document keyword extraction.

Reference: arXiv:2510.06719v2, Section "Stage 1: Soft Clustering",
sub-step "Keywords Histogram Generation".

For each document the LLM is prompted to emit K distinct keywords drawn from a
fixed vocabulary V. The output of this stage is a binary matrix
    Ext in {0, 1}^(N x |V|)
where Ext[i, j] = 1 iff keyword w_j was selected for document d_i.

Privacy sensitivity (used by the zCDP accountant in
metrics/privacy_accounting_zcdp.py):
    Each document contributes exactly K ones to the histogram h(D),
    so the L2 sensitivity of h is sqrt(K). This is what justifies the
    rho_h = K / (2 sigma_h^2) term in the paper formula.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.device import free_memory, model_dtype


class KeywordExtractor:
    """Stage 1a: extract K keywords per document via an LLM, restricted to V."""

    # DPSYN-05: paper wording (arXiv:2510.06719v2) — "Extract {K} single words ...
    # that represent key information specific to the content." The output-format
    # instruction is kept so the comma-separated list parses back into vocab tokens.
    DEFAULT_PROMPT_TEMPLATE = (
        "Extract {K} single words that represent key information specific to the "
        "content. Output the words as a comma-separated list, in lowercase, with "
        "no explanation.\n\n"
        "Document: {document}\n\n"
        "Keywords:"
    )

    def __init__(
        self,
        model_name: str,
        K: int,
        vocab: Iterable[str],
        max_new_tokens: int = 50,
        prompt_template: str | None = None,
        device: str = "auto",
        seed: int = 42,
    ):
        if K <= 0:
            raise ValueError("K must be >= 1.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be >= 1.")

        self.model_name = model_name
        self.K = int(K)
        self.max_new_tokens = int(max_new_tokens)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.seed = int(seed)

        # Vocabulary handling: store both the sorted list (for stable indexing)
        # and a lowercase -> index dict for O(1) lookups during parsing.
        vocab_list = sorted({w.strip().lower() for w in vocab if w and w.strip()})
        if not vocab_list:
            raise ValueError("vocab must contain at least one non-empty word.")
        self.vocab = vocab_list
        self._vocab_index = {w: i for i, w in enumerate(self.vocab)}

        self.device = self._resolve_device(device)
        torch.set_grad_enabled(False)

        # Lazy load: the model is not placed on the device until the first
        # extract() call. DP-SynRAG runs several LLM-backed stages (keyword
        # extraction, synthetic generation, self-filter) that all use the same
        # model; loading lazily and calling unload() after each stage keeps only
        # one copy resident on the GPU at a time.
        self.tokenizer = None
        self.model = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_default_vocab(
        min_word_len: int = 3,
        exclude_stopwords: bool = True,
        download_if_missing: bool = True,
    ) -> list[str]:
        """Build the NLTK-derived English vocabulary used in the paper.

        Uses nltk.corpus.words filtered to alphabetic tokens of length
        >= min_word_len, optionally removing English stopwords.
        """
        import nltk

        def _ensure(name: str, lookup: str) -> None:
            try:
                nltk.data.find(lookup)
            except LookupError:
                if not download_if_missing:
                    raise
                nltk.download(name, quiet=True)

        _ensure("words", "corpora/words")
        if exclude_stopwords:
            _ensure("stopwords", "corpora/stopwords")

        from nltk.corpus import words as nltk_words

        stop = set()
        if exclude_stopwords:
            from nltk.corpus import stopwords as nltk_stopwords

            stop = set(nltk_stopwords.words("english"))

        out: set[str] = set()
        for w in nltk_words.words():
            wl = w.lower()
            if len(wl) < min_word_len:
                continue
            if not wl.isalpha():
                continue
            if wl in stop:
                continue
            out.add(wl)
        return sorted(out)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, documents: list[str]) -> np.ndarray:
        """Return a binary matrix of shape (N, |V|) marking the K extracted keywords."""
        self._ensure_loaded()
        N = len(documents)
        V = self.vocab_size
        ext = np.zeros((N, V), dtype=np.float32)
        n_truncated = 0
        for i, doc in enumerate(documents):
            picks, truncated = self._extract_one(doc)
            n_truncated += int(truncated)
            for w in picks:
                idx = self._vocab_index.get(w)
                if idx is not None:
                    ext[i, idx] = 1.0
        if n_truncated:
            print(
                f"[WARN] keyword extractor: {n_truncated}/{N} documents were truncated "
                f"to the model context window; part of their content was omitted."
            )
        return ext

    def _extract_one(self, document: str) -> tuple[list[str], bool]:
        prompt = self.prompt_template.format(K=self.K, document=document)
        # Cap the prompt to the model's context window (minus the generation budget)
        # so long documents — e.g. movielens user profiles with hundreds of titles —
        # cannot overflow the position embeddings and trigger a device-side assert.
        model_max_len = getattr(self.model.config, "max_position_embeddings", 2048)
        max_len = max(8, int(model_max_len) - self.max_new_tokens)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_len
        ).to(self._model_device())
        truncated = int(inputs["input_ids"].shape[-1]) >= max_len
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip the prompt prefix so we only parse the model's continuation.
        if text.startswith(prompt):
            text = text[len(prompt) :]
        picks = self._parse_keywords(text)
        # Fallback: if the LLM produced no valid vocab words, find vocab words
        # that appear as substrings in the document. This keeps DP semantics
        # intact (still ≤ K keywords from V), while allowing smoke tests with
        # tiny/random models to produce non-trivial extraction matrices.
        if not picks:
            picks = self._fallback_extraction(document)
        return picks, truncated

    def _fallback_extraction(self, document: str) -> list[str]:
        doc_lower = document.lower()
        found = [w for w in self.vocab if w in doc_lower]
        return found[: self.K]

    def _parse_keywords(self, text: str) -> list[str]:
        # Stop at the first newline or sentence terminator after the keyword list.
        for stop in ("\n\n", "\n"):
            idx = text.find(stop)
            if idx >= 0:
                text = text[:idx]
                break
        picks: list[str] = []
        for raw in text.split(","):
            w = raw.strip().lower().strip(".;:'\"")
            if not w or not w.isalpha():
                continue
            if w in self._vocab_index and w not in picks:
                picks.append(w)
            if len(picks) >= self.K:
                break
        return picks

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self._load_model(self.model_name)
        self.model.eval()

    def unload(self) -> None:
        """Release the model from the GPU so the next stage can load its own.

        Idempotent; the model reloads lazily on the next extract() call.
        """
        self.model = None
        self.tokenizer = None
        free_memory()

    def _load_model(self, model_name: str):
        # No CPU fallback: on a GPU box a fp32-CPU reload silently runs the
        # token-by-token stages for hours. Let a CUDA OOM propagate so the caller
        # can skip the LLM cleanly instead of hanging.
        dtype = model_dtype()
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if not hasattr(model, "hf_device_map"):
            model = model.to(self.device)
        return model
