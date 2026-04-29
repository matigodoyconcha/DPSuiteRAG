from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np

from core.retriever import BaseRetriever


class DPRetriever(BaseRetriever):
    """
    DP-RAG Retriever: exponential mechanism (continuous version) for top-k selection.

    Reference: Grislain et al., "RAG with Differential Privacy", arXiv:2412.19291v2 (2024)

    The threshold τ is sampled from a continuous distribution over intervals between
    consecutive scores, weighted by both exp(ε·U(τ)/2) and the interval width Δτ.
    This is the exact continuous form described in the paper.
    """

    def __init__(
        self,
        index,
        epsilon: float,
        k: int = 5,
        max_docs: int = 20,
        min_score: float = -0.5,
        max_score: float = 0.8,
        seed: int = 42,
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0 for DP retrieval.")
        if k <= 0:
            raise ValueError("k must be >= 1.")
        if max_docs <= 0:
            raise ValueError("max_docs must be >= 1.")

        self.indexer = index
        self.epsilon = float(epsilon)
        self.k = int(k)
        self.max_docs = int(max_docs)
        self.min_score = float(min_score)
        self.max_score = float(max_score)
        self.seed = int(seed)

    def _get_index(self) -> Any:
        idx = getattr(self.indexer, "index", None)
        if idx is None and hasattr(self.indexer, "get_index"):
            idx = self.indexer.get_index()
        return idx

    def _exp_mechanism_top_k(self, scores: np.ndarray, rng: np.random.Generator) -> float:
        """
        Sample threshold τ using the continuous exponential mechanism for top-k.

        Sorts scores, inserts boundary values at -1 and 1, clips to [min_score, max_score],
        then samples from intervals weighted by exp(ε·U/2) · Δτ (interval width).

        Utility: U(τ) = -|len(sorted_scores) - k - index|
        """
        sorted_scores = np.sort(scores)
        sorted_scores = np.insert(sorted_scores, 0, -1.0)
        sorted_scores = np.insert(sorted_scores, len(sorted_scores), 1.0)
        sorted_scores = np.clip(sorted_scores, self.min_score, self.max_score)

        # Utility at each interval left-endpoint
        sorted_utilities = -np.abs(
            len(sorted_scores) - self.k - np.arange(len(sorted_scores))
        )
        delta_scores = np.diff(sorted_scores)
        # PDF weighted by interval width (continuous mechanism)
        pdf = np.exp(self.epsilon * sorted_utilities[:-1] / 2) * delta_scores
        pdf_sum = np.sum(pdf)
        if pdf_sum <= 0:
            return float(sorted_scores[len(sorted_scores) // 2])
        pdf /= pdf_sum
        return float(rng.choice(sorted_scores[:-1], p=pdf))

    def retrieve(self, query_vector: np.ndarray, k: int = None) -> list[dict]:  # noqa: ARG002
        index = self._get_index()
        if index is None:
            raise RuntimeError("Index has not been built or loaded yet.")

        ntotal = int(getattr(index, "ntotal", 0))
        if ntotal <= 0:
            return []

        query_vec = np.array([query_vector], dtype=np.float32)
        norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        norm[norm == 0.0] = 1.0
        query_vec = query_vec / norm

        search_k = min(ntotal, max(1, self.max_docs * 3))
        scores, indices = index.search(query_vec, search_k)

        candidates: list[dict] = []
        metadata = getattr(self.indexer, "metadata", [])
        for j, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(metadata):
                continue
            chunk_meta = metadata[idx]
            text = str(chunk_meta.get("text", "")) if isinstance(chunk_meta, dict) else ""
            candidates.append(
                {
                    "text": text,
                    "metadata": chunk_meta,
                    "score": float(scores[0][j]),
                }
            )

        if not candidates:
            return []

        candidates.sort(key=lambda x: x["score"], reverse=True)
        for c in candidates:
            c["score"] = float(np.clip(c["score"], -1.0, 1.0))

        candidate_scores = np.array([c["score"] for c in candidates])

        # Per-query seed for reproducibility
        query_offset = int(hashlib.sha256(query_vec.tobytes()).hexdigest()[:8], 16)
        rng = np.random.default_rng(self.seed + query_offset)

        tau_dp = self._exp_mechanism_top_k(candidate_scores, rng)

        selected = [c for c in candidates if c["score"] >= tau_dp]
        if not selected:
            selected = [candidates[0]]

        # Shuffle post-selection (post-processing; doesn't affect DP guarantee).
        # Prevents the generator from systematically attending to highest-scored
        # documents first due to ordering.
        rng_shuffle = random.Random(self.seed + query_offset + 1)
        rng_shuffle.shuffle(selected)

        return selected[: self.max_docs]


class DPTopPRetriever(BaseRetriever):
    """
    DP-RAG Top-P Retriever: continuous exponential mechanism for top-p selection.

    Reference: Grislain et al., "RAG with Differential Privacy", arXiv:2412.19291v2 (2024)

    Documents are weighted by relevance w(s) = exp(alpha·(s−s_max)/(s_max−s_min)).
    The threshold τ is sampled from the continuous exponential mechanism.
    """

    def __init__(
        self,
        index,
        epsilon: float,
        p: float = 0.3,
        alpha: float = 5.0,
        max_docs: int = 20,
        min_score: float = -0.5,
        max_score: float = 0.8,
        seed: int = 42,
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0 for DP retrieval.")
        if not (0.0 < p <= 1.0):
            raise ValueError("p must be in (0, 1] for top-p selection.")
        if alpha <= 0:
            raise ValueError("alpha must be > 0 for weight function.")
        if max_docs <= 0:
            raise ValueError("max_docs must be >= 1.")

        self.indexer = index
        self.epsilon = float(epsilon)
        self.p = float(p)
        self.alpha = float(alpha)
        self.max_docs = int(max_docs)
        self.min_score = float(min_score)
        self.max_score = float(max_score)
        self.seed = int(seed)

    def _get_index(self) -> Any:
        idx = getattr(self.indexer, "index", None)
        if idx is None and hasattr(self.indexer, "get_index"):
            idx = self.indexer.get_index()
        return idx

    def _exp_mechanism_top_p(self, scores: np.ndarray, rng: np.random.Generator) -> float:
        """
        Sample threshold τ using the continuous exponential mechanism for top-p.

        Documents are weighted by w(s) = exp(alpha·(s−max_score)/(max_score−min_score)).
        Utility: U(τ) = -|Σ w(sᵢ)·(1−p) − cumsum of w over sorted thresholds|
        """
        sorted_scores = np.sort(scores)
        sorted_scores = np.insert(sorted_scores, 0, -1.0)
        sorted_scores = np.insert(sorted_scores, len(sorted_scores), 1.0)
        sorted_scores = np.clip(sorted_scores, self.min_score, self.max_score)

        # Weight each threshold position by its relevance score
        sorted_score_probs = np.exp(
            self.alpha * (sorted_scores - self.max_score) / (self.max_score - self.min_score)
        )
        # Utility: how close is the cumulative weight above τ to (1-p)·total_weight
        sorted_utilities = -np.abs(
            np.sum(sorted_score_probs) * (1 - self.p) - np.cumsum(sorted_score_probs)
        )
        delta_scores = np.diff(sorted_scores)
        pdf = np.exp(self.epsilon * sorted_utilities[:-1] / 2) * delta_scores
        pdf_sum = np.sum(pdf)
        if pdf_sum <= 0:
            return float(sorted_scores[len(sorted_scores) // 2])
        pdf /= pdf_sum
        return float(rng.choice(sorted_scores[:-1], p=pdf))

    def retrieve(self, query_vector: np.ndarray, k: int = None) -> list[dict]:  # noqa: ARG002
        index = self._get_index()
        if index is None:
            raise RuntimeError("Index has not been built or loaded yet.")

        ntotal = int(getattr(index, "ntotal", 0))
        if ntotal <= 0:
            return []

        query_vec = np.array([query_vector], dtype=np.float32)
        norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        norm[norm == 0.0] = 1.0
        query_vec = query_vec / norm

        search_k = min(ntotal, max(1, self.max_docs * 3))
        scores, indices = index.search(query_vec, search_k)

        candidates: list[dict] = []
        metadata = getattr(self.indexer, "metadata", [])
        for j, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(metadata):
                continue
            chunk_meta = metadata[idx]
            text = str(chunk_meta.get("text", "")) if isinstance(chunk_meta, dict) else ""
            candidates.append(
                {
                    "text": text,
                    "metadata": chunk_meta,
                    "score": float(scores[0][j]),
                }
            )

        if not candidates:
            return []

        candidates.sort(key=lambda x: x["score"], reverse=True)
        for c in candidates:
            c["score"] = float(np.clip(c["score"], -1.0, 1.0))

        candidate_scores = np.array([c["score"] for c in candidates])

        # Per-query seed for reproducibility
        query_offset = int(hashlib.sha256(query_vec.tobytes()).hexdigest()[:8], 16)
        rng = np.random.default_rng(self.seed + query_offset)

        tau_dp = self._exp_mechanism_top_p(candidate_scores, rng)

        selected = [c for c in candidates if c["score"] >= tau_dp]
        if not selected:
            selected = [candidates[0]]

        # Shuffle post-selection (post-processing; doesn't affect DP guarantee).
        rng_shuffle = random.Random(self.seed + query_offset + 1)
        rng_shuffle.shuffle(selected)

        return selected[: self.max_docs]
