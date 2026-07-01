"""Private-RAG: multi-query differentially private RAG (MURAG / MURAG-ADA).

Wu, Wang, Zhang & Wang, "Private-RAG: Answering Multiple Queries with LLMs while
Keeping Your Data Private", arXiv:2511.07637 (2025).

Multi-query method. A per-document individual privacy filter (``BudgetManager``,
``E(z) = M * eps_q``) makes the accumulated privacy loss depend on how often each
document is retrieved, not on the number of queries. Only the *retrieval* differs
between the two variants; both hand their Top-K documents to the same pure-DP
single-query subroutine (Algorithm 7, ``PrivateRAGGenerator``):

- ``BudgetAwareRetriever`` (MURAG): fixed public relevance threshold.
- ``AdaptiveBudgetAwareRetriever`` (MURAG-ADA): query-specific threshold released
  via noisy prefix sums.

The generator exposes ``.m`` / ``.k`` / ``generate_vote(...)`` so it drops into the
existing ``DPVoteRAGPipeline`` — no dedicated pipeline class is required.
"""

from .budget_manager import BudgetManager
from .generator import PrivateRAGGenerator
from .retriever import AdaptiveBudgetAwareRetriever, BudgetAwareRetriever

__all__ = [
    "BudgetManager",
    "BudgetAwareRetriever",
    "AdaptiveBudgetAwareRetriever",
    "PrivateRAGGenerator",
]
