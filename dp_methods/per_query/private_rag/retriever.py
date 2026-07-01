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
