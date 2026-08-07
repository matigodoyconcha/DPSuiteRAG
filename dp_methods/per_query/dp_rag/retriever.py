from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np

from core.retriever import BaseRetriever


# ---------------------------------------------------------------------------
# Why ALL N documents must be scored before the exponential mechanism
# ---------------------------------------------------------------------------
# The DP exponential mechanism for threshold selection uses the utility:
#
#   U_top_k(τ) = -|#{i : s_i ≥ τ} - k|
#
# This sum runs over **all N documents** in the index.  If we restrict the
# scored set to, say, the top-M candidates from a FAISS pre-filter (M < N),
# the utility function is evaluated on a truncated dataset: documents with low
# scores are silently omitted from the count.  For thresholds τ below those
# missing scores the count is wrong, the utility distribution is biased, and
# the ε-DP proof no longer holds.
#
# The reference implementation (sarus-tech/dp-rag, pup_vector_store.py) avoids
# this by computing cosine scores for the entire corpus via a single matrix
# multiply:  scores = query @ embeddings.T
#
# We do the same.  FAISSIndexer now stores the L2-normalised embedding matrix
# so the retriever can issue one BLAS call: O(N·d).  For IndexFlatIP this is
# the same arithmetic as an exhaustive FAISS search, but without the sorting
# overhead when we only need the mechanism to sample a threshold.
# ---------------------------------------------------------------------------


class DPRetriever(BaseRetriever):
    """DP-RAG Retriever: exponential mechanism (continuous) for top-k selection.

    Reference: Grislain et al., "RAG with Differential Privacy",
    arXiv:2412.19291v2 (2024), Section "Privacy Unit Preserving Document
    Retrieval".

    The threshold τ is sampled from the continuous exponential mechanism:

        τ ∝ exp(ε · U_top_k(τ) / 2) · Δτ

    where U_top_k(τ) = -|#{docs with score ≥ τ} - k| has sensitivity 1.
    The interval width Δτ is the Jacobian of the continuous version (see paper
    Fig. 4).  All N indexed documents are scored before the mechanism so that
    the utility function is evaluated on the complete dataset.
    """

    def __init__(
        self,
        index,
        epsilon: float,
        k: int = 5,
        min_score: float = -0.5,
        max_score: float = 0.8,
        seed: int = 42,
        verbose: bool = False,
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0 for DP retrieval.")
        if k <= 0:
            raise ValueError("k must be >= 1.")

        self.indexer = index
        self.epsilon = float(epsilon)
        self.k = int(k)
        self.min_score = float(min_score)
        self.max_score = float(max_score)
        self.seed = int(seed)
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------
    # Internal: score every document in the corpus
    # ------------------------------------------------------------------

    def _score_all_documents(
        self, query_vec_1d: np.ndarray
    ) -> tuple[np.ndarray, list[dict]]:
        """Return cosine scores for ALL N indexed documents.

        Uses ``indexer.embeddings_matrix`` (fast path: one BLAS matrix-vector
        multiply) when available.  Falls back to an exhaustive FAISS search
        returning all ntotal results when the attribute is absent (e.g. an
        index loaded from a cache built before the embeddings matrix was
        introduced).

        Args:
            query_vec_1d: L2-normalised query embedding, shape (d,).

        Returns:
            Tuple of (scores, metadata) where scores[i] is the cosine
            similarity between the query and the i-th indexed document and
            metadata[i] is the corresponding metadata dict.
        """
        metadata: list[dict] = getattr(self.indexer, "metadata", [])
        embeddings: np.ndarray | None = getattr(
            self.indexer, "embeddings_matrix", None
        )

        if embeddings is not None:
            # Fast path — one matrix-vector multiply: O(N·d)
            scores = (embeddings @ query_vec_1d).astype(np.float64)
            return scores, metadata

        # Fallback: FAISS exhaustive search for all ntotal results.
        # IndexFlatIP always does O(N·d) compute regardless of k, so the only
        # extra cost vs the fast path is a partial sort of N values.
        index = getattr(self.indexer, "index", None)
        if index is None and hasattr(self.indexer, "get_index"):
            index = self.indexer.get_index()
        ntotal = int(getattr(index, "ntotal", 0))

        q2d = query_vec_1d[np.newaxis, :].astype(np.float32)
        raw_scores, raw_indices = index.search(q2d, ntotal)

        scores = np.full(len(metadata), -2.0, dtype=np.float64)
        for j, idx in enumerate(raw_indices[0]):
            if 0 <= idx < len(metadata):
                scores[idx] = float(raw_scores[0][j])
        return scores, metadata

    # ------------------------------------------------------------------
    # Exponential mechanism
    # ------------------------------------------------------------------

    def _exp_mechanism_top_k(
        self, scores: np.ndarray, rng: np.random.Generator
    ) -> float:
        """Sample threshold τ via the continuous exponential mechanism (top-k).

        Inserts boundary values at -1 and +1, clips to [min_score, max_score],
        then draws τ from intervals weighted by exp(ε·U/2)·Δτ.

        Utility per interval left-endpoint:
            U(τ) = -|#{docs with score ≥ τ} - k|

        Sensitivity of U: Δ = 1 (adding/removing one document changes the
        count by at most 1, hence |U(D) - U(D')| ≤ 1 for any τ).  With the
        factor ε/2 in the exponent this gives ε-DP (Dwork & Roth, 2014).
        """
        sorted_scores = np.sort(scores)
        sorted_scores = np.insert(sorted_scores, 0, -1.0)
        sorted_scores = np.insert(sorted_scores, len(sorted_scores), 1.0)
        sorted_scores = np.clip(sorted_scores, self.min_score, self.max_score)

        sorted_utilities = -np.abs(
            len(sorted_scores) - self.k - np.arange(len(sorted_scores))
        )
        delta_scores = np.diff(sorted_scores)
        pdf = np.exp(self.epsilon * sorted_utilities[:-1] / 2) * delta_scores
        pdf_sum = float(np.sum(pdf))
        if pdf_sum <= 0:
            if self.verbose:
                print(
                    "[DP-RAG WARNING] _exp_mechanism_top_k: pdf_sum <= 0 "
                    "(degenerate score distribution). Falling back to median threshold."
                )
            return float(sorted_scores[len(sorted_scores) // 2])
        pdf /= pdf_sum
        return float(rng.choice(sorted_scores[:-1], p=pdf))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, query_vector: np.ndarray, k: int = None) -> list[dict]:  # noqa: ARG002
        """Retrieve documents using the DP exponential mechanism (top-k).

        Scores ALL indexed documents against the query, samples a DP threshold
        τ via the exponential mechanism, and returns EVERY document whose score
        exceeds τ — the released set is exactly the selected set.
        """
        query_vec = np.array(query_vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        scores, metadata = self._score_all_documents(query_vec)
        if len(scores) == 0:
            return []

        scores = np.clip(scores, -1.0, 1.0)

        # The run seed goes through the hash rather than being added to the offset:
        # under `seed + offset` two runs seeded s and s+1 land on the same stream
        # whenever their queries' offsets differ by one, which correlates releases a
        # seed sweep is supposed to keep independent.
        query_offset = int(
            hashlib.sha256(f"{self.seed}|".encode("utf-8") + query_vec.tobytes())
            .hexdigest()[:8],
            16,
        )
        rng = np.random.default_rng(query_offset)

        tau_dp = self._exp_mechanism_top_k(scores, rng)

        selected = [
            {
                "text": str(meta.get("text", "")) if isinstance(meta, dict) else "",
                "metadata": meta,
                "score": float(scores[i]),
            }
            for i, meta in enumerate(metadata)
            if float(scores[i]) > tau_dp
        ]

        # Nothing clears the sampled threshold on some queries, and the answer is then
        # the empty set. Substituting argmax(scores) would be a release of the corpus
        # that no budget paid for: the empty set and {argmax(D)} are disjoint, so the
        # two neighbours differ in two documents rather than one and the generation
        # sensitivity doubles to 2C on exactly the tightest cells, where a large tau is
        # sampled most often. Returning the selected set as it stands is post-processing
        # of the threshold that was paid for, so it costs nothing; the generator still
        # holds the public prompt and answers from its prior.

        # Post-processing shuffle — does not affect the DP guarantee because
        # it is applied after the private selection step.
        rng_shuffle = random.Random(query_offset + 1)
        rng_shuffle.shuffle(selected)
        return selected


class DPTopPRetriever(BaseRetriever):
    """DP-RAG Retriever: continuous exponential mechanism for top-p selection.

    Reference: Grislain et al., "RAG with Differential Privacy",
    arXiv:2412.19291v2 (2024), Section "Privacy Unit Preserving Document
    Retrieval" (top-p variant, Fig. 5).

    Documents are weighted by relevance:
        w(s) = exp(alpha · (s − s_max) / (s_max − s_min))  ∈ [0, 1]

    The threshold τ is sampled from:
        τ ∝ exp(ε · U_top_p(τ) / 2) · Δτ

    where U_top_p(τ) = -|Σᵢ w(sᵢ)·1[sᵢ ≥ τ] − p·Σᵢ w(sᵢ)|.

    All N indexed documents are scored before the mechanism — same DP
    correctness requirement as DPRetriever (see module docstring above).

    Note on the fixed score window (min_score=-0.5, max_score=0.8): these
    bounds are taken from the reference (sarus-tech/dp-rag), where they were
    calibrated for the cosine score distribution of Snowflake arctic-embed.
    This benchmark standardizes retrieval on that same embedder
    (arctic-embed-m-v1.5) across all methods, so the window is matched to the
    embedder's cosine range and is kept as a shared standard convention rather
    than re-calibrated per dataset (see experiments/configs/full/dp_rag.yaml,
    models.embedder).
    """

    def __init__(
        self,
        index,
        epsilon: float,
        p: float = 0.3,
        alpha: float = 5.0,
        min_score: float = -0.5,
        max_score: float = 0.8,
        seed: int = 42,
        verbose: bool = False,
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0 for DP retrieval.")
        if not (0.0 < p <= 1.0):
            raise ValueError("p must be in (0, 1] for top-p selection.")
        if alpha <= 0:
            raise ValueError("alpha must be > 0 for weight function.")

        self.indexer = index
        self.epsilon = float(epsilon)
        self.p = float(p)
        self.alpha = float(alpha)
        self.min_score = float(min_score)
        self.max_score = float(max_score)
        self.seed = int(seed)
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------
    # Internal: score every document (shared logic, same as DPRetriever)
    # ------------------------------------------------------------------

    def _score_all_documents(
        self, query_vec_1d: np.ndarray
    ) -> tuple[np.ndarray, list[dict]]:
        """Score ALL N documents — see DPRetriever._score_all_documents."""
        metadata: list[dict] = getattr(self.indexer, "metadata", [])
        embeddings: np.ndarray | None = getattr(
            self.indexer, "embeddings_matrix", None
        )

        if embeddings is not None:
            scores = (embeddings @ query_vec_1d).astype(np.float64)
            return scores, metadata

        index = getattr(self.indexer, "index", None)
        if index is None and hasattr(self.indexer, "get_index"):
            index = self.indexer.get_index()
        ntotal = int(getattr(index, "ntotal", 0))

        q2d = query_vec_1d[np.newaxis, :].astype(np.float32)
        raw_scores, raw_indices = index.search(q2d, ntotal)

        scores = np.full(len(metadata), -2.0, dtype=np.float64)
        for j, idx in enumerate(raw_indices[0]):
            if 0 <= idx < len(metadata):
                scores[idx] = float(raw_scores[0][j])
        return scores, metadata

    # ------------------------------------------------------------------
    # Exponential mechanism
    # ------------------------------------------------------------------

    def _exp_mechanism_top_p(
        self, scores: np.ndarray, rng: np.random.Generator
    ) -> float:
        """Sample threshold τ via the continuous exponential mechanism (top-p).

        Weight function: w(s) = exp(alpha·(s−max_score)/(max_score−min_score))
        Utility: U(τ) = -|Σ w(sᵢ)·1[sᵢ≥τ] − p·Σ w(sᵢ)|
        Sensitivity: Δ = 1 (each document contributes w(s) ≤ 1).
        """
        sorted_scores = np.sort(scores)
        sorted_scores = np.insert(sorted_scores, 0, -1.0)
        sorted_scores = np.insert(sorted_scores, len(sorted_scores), 1.0)
        sorted_scores = np.clip(sorted_scores, self.min_score, self.max_score)

        sorted_score_probs = np.exp(
            self.alpha
            * (sorted_scores - self.max_score)
            / (self.max_score - self.min_score)
        )
        sorted_utilities = -np.abs(
            np.sum(sorted_score_probs) * (1 - self.p)
            - np.cumsum(sorted_score_probs)
        )
        delta_scores = np.diff(sorted_scores)
        pdf = np.exp(self.epsilon * sorted_utilities[:-1] / 2) * delta_scores
        pdf_sum = float(np.sum(pdf))
        if pdf_sum <= 0:
            if self.verbose:
                print(
                    "[DP-RAG WARNING] _exp_mechanism_top_p: pdf_sum <= 0 "
                    "(degenerate score distribution). Falling back to median threshold."
                )
            return float(sorted_scores[len(sorted_scores) // 2])
        pdf /= pdf_sum
        return float(rng.choice(sorted_scores[:-1], p=pdf))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, query_vector: np.ndarray, k: int = None) -> list[dict]:  # noqa: ARG002
        """Retrieve documents using the DP exponential mechanism (top-p)."""
        query_vec = np.array(query_vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        scores, metadata = self._score_all_documents(query_vec)
        if len(scores) == 0:
            return []

        scores = np.clip(scores, -1.0, 1.0)

        # The run seed goes through the hash rather than being added to the offset:
        # under `seed + offset` two runs seeded s and s+1 land on the same stream
        # whenever their queries' offsets differ by one, which correlates releases a
        # seed sweep is supposed to keep independent.
        query_offset = int(
            hashlib.sha256(f"{self.seed}|".encode("utf-8") + query_vec.tobytes())
            .hexdigest()[:8],
            16,
        )
        rng = np.random.default_rng(query_offset)

        tau_dp = self._exp_mechanism_top_p(scores, rng)

        selected = [
            {
                "text": str(meta.get("text", "")) if isinstance(meta, dict) else "",
                "metadata": meta,
                "score": float(scores[i]),
            }
            for i, meta in enumerate(metadata)
            if float(scores[i]) > tau_dp
        ]

        # No argmax fallback and no cap, for the reasons given in DPRetriever.retrieve:
        # both replace a document rather than add one, and each costs a factor 2 in the
        # generation sensitivity the whole method is accounted at.
        rng_shuffle = random.Random(query_offset + 1)
        rng_shuffle.shuffle(selected)
        return selected
