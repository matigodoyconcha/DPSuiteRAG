"""Budget-aware retrievers for Private-RAG (MURAG / MURAG-ADA).

Reference: Wu, Wang, Zhang & Wang, "Private-RAG: Answering Multiple Queries with
LLMs while Keeping Your Data Private", arXiv:2511.07637 (2025).

Only the *retrieval* step differs between MURAG and MURAG-ADA; both hand their
Top-K documents to the same single-query generation subroutine (Algorithm 7, see
``generator.py``). Retrieval is made budget-aware via an individual privacy filter
(``budget_manager.BudgetManager``): every document carries a remaining budget
``E(z) = M * eps_q`` and is dropped once it can no longer pay a query.

- ``BudgetAwareRetriever`` (MURAG, Alg. 1): a **fixed, public** relevance threshold
  ``tau`` (data-independent, no DP cost). All documents in the active set whose
  score exceeds ``tau`` are charged ``eps_q``; the Top-K by relevance are returned.
- ``AdaptiveBudgetAwareRetriever`` (MURAG-ADA, Alg. 2): a **query-specific** threshold
  released via noisy prefix sums over score bins (Laplace noise, costs ``eps_thr``),
  then a DP-RAG step over the resulting active set (costs ``eps_RAG``).

Privacy unit: one indexed entry (the document's row in
``indexer.embeddings_matrix``) is one document ``z`` (same convention as the other
per-query methods; see ``dp_methods/per_query/dp_rag/retriever.py``). All N
documents are scored with a single BLAS matrix-vector multiply so that the
budget/threshold filtering is evaluated over the complete corpus.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from core.retriever import BaseRetriever
from dp_methods.per_query.private_rag.budget_manager import BudgetManager


def _score_all_documents(
    indexer: Any, query_vec_1d: np.ndarray
) -> tuple[np.ndarray, list[dict]]:
    """Return cosine scores for ALL N indexed documents and their metadata.

    Uses ``indexer.embeddings_matrix`` (one matrix-vector multiply, O(N*d)) when
    available, otherwise falls back to an exhaustive FAISS search. Mirrors
    ``dp_methods/per_query/dp_rag/retriever.py`` so the budget filter sees every
    document, not just a pre-filtered top-M.
    """
    metadata: list[dict] = getattr(indexer, "metadata", [])
    embeddings: np.ndarray | None = getattr(indexer, "embeddings_matrix", None)

    if embeddings is not None:
        scores = (embeddings @ query_vec_1d).astype(np.float64)
        return scores, metadata

    index = getattr(indexer, "index", None)
    if index is None and hasattr(indexer, "get_index"):
        index = indexer.get_index()
    ntotal = int(getattr(index, "ntotal", 0))

    q2d = query_vec_1d[np.newaxis, :].astype(np.float32)
    raw_scores, raw_indices = index.search(q2d, ntotal)

    scores = np.full(len(metadata), -2.0, dtype=np.float64)
    for j, idx in enumerate(raw_indices[0]):
        if 0 <= idx < len(metadata):
            scores[idx] = float(raw_scores[0][j])
    return scores, metadata


def _normalise_query(query_vector: np.ndarray) -> np.ndarray:
    query_vec = np.array(query_vector, dtype=np.float32).flatten()
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm
    return query_vec


def _result_dict(meta: Any, score: float) -> dict:
    text = str(meta.get("text", "")) if isinstance(meta, dict) else ""
    return {"text": text, "metadata": meta, "score": float(score)}


class BudgetAwareRetriever(BaseRetriever):
    """MURAG retriever (Alg. 1): fixed public threshold + individual budget filter.

    Per query (verbatim from the paper):
        A_t   = { z | E(z) >= eps_q }                 # active set
        D_qt  = { z in A_t | r(z, q) > tau }          # fixed public threshold tau
        for z in D_qt: E(z) <- E(z) - eps_q           # charge ALL above threshold
        D_qt^k = Top-K(D_qt, k, r)                    # select top-k by relevance
    and ``D_qt^k`` is returned for generation. The threshold ``tau`` is a public,
    data-independent constant, so it consumes no privacy budget; the whole cost is
    the ``eps_q`` charged per retrieved (above-threshold) document.

    Args:
        indexer: A built index exposing ``embeddings_matrix`` / ``metadata``.
        budget_manager: Shared :class:`BudgetManager` (stateful across queries).
        epsilon_q: Per-query budget charged to each above-threshold document.
        threshold: Fixed public relevance threshold ``tau`` (cosine scale).
        max_docs: Optional hard cap on the number of returned documents (defaults
            to the ``k`` passed to :meth:`retrieve`).
        seed: Unused by MURAG (retrieval is deterministic); kept for a uniform
            constructor across the two variants.
    """

    def __init__(
        self,
        indexer: Any,
        budget_manager: BudgetManager,
        epsilon_q: float = 1.0,
        threshold: float = 0.0,
        max_docs: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ):
        if epsilon_q <= 0:
            raise ValueError(f"epsilon_q must be > 0, got {epsilon_q}")
        self.indexer = indexer
        self.budget_manager = budget_manager
        self.epsilon_q = float(epsilon_q)
        self.threshold = float(threshold)
        self.max_docs = int(max_docs) if max_docs is not None else None
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.last_run_stats: dict[str, Any] = {}

    def retrieve(self, query_vector: np.ndarray, k: int = 5) -> list[dict]:
        query_vec = _normalise_query(query_vector)
        scores, metadata = _score_all_documents(self.indexer, query_vec)
        n = len(scores)
        if n == 0:
            self.last_run_stats = {"n_active": 0, "n_charged": 0, "n_returned": 0}
            return []

        top_k = int(k) if k is not None else 5
        cap = self.max_docs if self.max_docs is not None else top_k

        # A_t: documents that still have budget for a full query (>= eps_q).
        active = self.budget_manager.active_ids(self.epsilon_q, range(n))
        # D_qt: active documents above the public relevance threshold tau.
        above = [i for i in active if float(scores[i]) > self.threshold]

        # Charge eps_q to EVERY above-threshold document (before Top-K), matching
        # the paper's loop order: cost is incurred by all retrieved documents, not
        # only the top-k eventually used for generation.
        self.budget_manager.consume_many(above, self.epsilon_q)

        # D_qt^k: top-k above-threshold documents by relevance.
        above.sort(key=lambda i: float(scores[i]), reverse=True)
        selected_idx = above[:cap]
        selected = [_result_dict(metadata[i], scores[i]) for i in selected_idx]

        self.last_run_stats = {
            "n_active": len(active),
            "n_charged": len(above),
            "n_returned": len(selected),
            "threshold": self.threshold,
        }
        if self.verbose:
            print(
                f"[MURAG] active={len(active)} above_tau={len(above)} "
                f"returned={len(selected)} (tau={self.threshold}, eps_q={self.epsilon_q})"
            )
        return selected


class AdaptiveBudgetAwareRetriever(BaseRetriever):
    """MURAG-ADA retriever (Alg. 2): adaptive threshold via noisy prefix sums.

    The per-query budget is split ``eps_q = eps_thr + eps_RAG``. A query-specific
    threshold ``tau_t`` is released privately by walking similarity-score bins from
    highest to lowest and accumulating a *noisy* prefix sum of eligible-document
    counts until it exceeds ``k``:

        for each bin i (high -> low):
            A_i   = { z | r(z,q) in bin_i, E(z) >= eps_thr }
            s_hat = s_hat + |A_i| + Lap(1/eps_thr)     # noisy prefix sum
            E(z) <- E(z) - eps_thr   for z in A_i      # charge processed bins
            if s_hat >= k: tau_t = lower_edge(bin_i); break

    Then a DP-RAG step runs over the active set above ``tau_t``:

        A_t'  = { z in processed bins | E(z) >= eps_RAG }
        E(z) <- E(z) - eps_RAG   for z in A_t'         # charge the active set
        D_qt  = Top-K(A_t', k, r)                       # returned for generation

    Because each document falls in exactly one bin, charging a bin does not affect
    another bin's eligibility, so a document pays at most ``eps_thr + eps_RAG =
    eps_q`` per query. The Laplace noise makes the released threshold DP; its cost
    is the ``eps_thr`` charged to the documents in the processed bins (individual
    filter). The paper uses DPR inner-product bins (70-100 @ 0.2); here scores are
    cosine, so the bin range/size are exposed as config knobs.

    Args:
        indexer: A built index exposing ``embeddings_matrix`` / ``metadata``.
        budget_manager: Shared :class:`BudgetManager` (stateful across queries).
        epsilon_q: Total per-query budget (``eps_thr + eps_RAG``).
        epsilon_thr: Budget spent on the noisy-prefix-sum threshold release.
        score_min, score_max, bin_size: Cosine-score binning grid (high -> low).
        max_docs: Optional hard cap on returned documents (defaults to ``k``).
        seed: Base seed; the per-query Laplace RNG is ``seed + hash(query_vec)``.
    """

    def __init__(
        self,
        indexer: Any,
        budget_manager: BudgetManager,
        epsilon_q: float = 2.0,
        epsilon_thr: float = 1.0,
        score_min: float = 0.0,
        score_max: float = 1.0,
        bin_size: float = 0.05,
        max_docs: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ):
        if epsilon_thr <= 0:
            raise ValueError(f"epsilon_thr must be > 0, got {epsilon_thr}")
        if epsilon_q <= epsilon_thr:
            raise ValueError(
                f"epsilon_q ({epsilon_q}) must exceed epsilon_thr ({epsilon_thr}) "
                "so that eps_RAG = eps_q - eps_thr > 0."
            )
        if bin_size <= 0 or score_max <= score_min:
            raise ValueError("Require bin_size > 0 and score_max > score_min.")
        self.indexer = indexer
        self.budget_manager = budget_manager
        self.epsilon_q = float(epsilon_q)
        self.epsilon_thr = float(epsilon_thr)
        self.epsilon_rag = float(epsilon_q) - float(epsilon_thr)
        self.score_min = float(score_min)
        self.score_max = float(score_max)
        self.bin_size = float(bin_size)
        self.n_bins = int(np.ceil((self.score_max - self.score_min) / self.bin_size))
        self.max_docs = int(max_docs) if max_docs is not None else None
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.last_run_stats: dict[str, Any] = {}

    def _query_rng(self, query_vec: np.ndarray) -> np.random.Generator:
        offset = int(hashlib.sha256(query_vec.tobytes()).hexdigest()[:8], 16)
        return np.random.default_rng(self.seed + offset)

    def retrieve(self, query_vector: np.ndarray, k: int = 5) -> list[dict]:
        query_vec = _normalise_query(query_vector)
        scores, metadata = _score_all_documents(self.indexer, query_vec)
        n = len(scores)
        if n == 0:
            self.last_run_stats = {"n_active": 0, "n_charged": 0, "n_returned": 0}
            return []

        target_k = int(k) if k is not None else 5
        cap = self.max_docs if self.max_docs is not None else target_k
        rng = self._query_rng(query_vec)

        # Group budget-eligible documents (E(z) >= eps_thr) into descending bins.
        # Bin 0 is the highest-score bin; documents below score_min are ignored.
        bins: list[list[int]] = [[] for _ in range(self.n_bins)]
        for j in range(n):
            if not self.budget_manager.can_query(j, self.epsilon_thr):
                continue
            s = float(scores[j])
            if s < self.score_min:
                continue
            b = int((self.score_max - s) // self.bin_size)
            b = min(max(b, 0), self.n_bins - 1)
            bins[b].append(j)

        # Step 1: noisy prefix sums, high -> low, charging eps_thr per processed bin.
        s_tilde = 0.0
        accumulated: list[int] = []
        tau_t = self.score_min
        lap_scale = 1.0 / self.epsilon_thr
        n_bins_processed = 0
        for i in range(self.n_bins):
            a_i = bins[i]
            s_tilde += len(a_i) + float(rng.laplace(0.0, lap_scale))
            self.budget_manager.consume_many(a_i, self.epsilon_thr)
            accumulated.extend(a_i)
            n_bins_processed = i + 1
            if s_tilde >= target_k:
                tau_t = self.score_max - (i + 1) * self.bin_size
                break

        # Step 2: DP-RAG over the active set above tau_t; charge eps_RAG to all of it.
        active = [
            j for j in accumulated if self.budget_manager.can_query(j, self.epsilon_rag)
        ]
        self.budget_manager.consume_many(active, self.epsilon_rag)
        active.sort(key=lambda j: float(scores[j]), reverse=True)
        selected_idx = active[:cap]
        selected = [_result_dict(metadata[j], scores[j]) for j in selected_idx]

        self.last_run_stats = {
            "n_active": len(active),
            "n_charged": len(accumulated),
            "n_returned": len(selected),
            "tau_t": float(tau_t),
            "n_bins_processed": n_bins_processed,
        }
        if self.verbose:
            print(
                f"[MURAG-ADA] tau_t={tau_t:.4f} bins_processed={n_bins_processed} "
                f"active={len(active)} returned={len(selected)} "
                f"(eps_thr={self.epsilon_thr}, eps_RAG={self.epsilon_rag})"
            )
        return selected
