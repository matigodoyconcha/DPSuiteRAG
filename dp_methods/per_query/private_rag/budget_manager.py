"""Per-document privacy budget manager for Private-RAG (MURAG / MURAG-ADA).

Reference: Wu, Wang, Zhang & Wang, "Private-RAG: Answering Multiple Queries with
LLMs while Keeping Your Data Private", arXiv:2511.07637 (2025).

The multi-query methods MURAG (Alg. 1) and MURAG-ADA (Alg. 2) rely on an
**individual privacy filter**: every document ``z`` in the corpus is given an
initial budget ``E(z) = M * eps_q`` (``M`` = maximum retrievals per document,
``eps_q`` = per-query budget). Each time a document participates in a query it is
charged, and once its remaining budget can no longer cover a query it is dropped
from the active set. The accumulated privacy loss therefore depends on how often
each document is retrieved, not on the number of queries.

Privacy accounting is done through the individual Renyi DP filter of Feldman &
Zrnic (2021, Corollary 3.3) at Renyi order ``alpha = infinity``, which is exactly
pure DP. At that order composition of the per-participation costs is simply
**additive**, so this manager just tracks a running per-document remaining budget
and the method-level guarantee is ``E(z) <= M * eps_q`` (Theorems 1 and 2).

This is a plain auxiliary component: it does not inherit from any of the abstract
pipeline classes.
"""

from __future__ import annotations

from typing import Any, Hashable, Iterable

# Numerical tolerance so that a document whose remaining budget equals the cost
# (up to floating-point error) is still considered able to pay for it.
_EPS_TOL = 1e-9


class BudgetManager:
    """Tracks the remaining per-document privacy budget (individual DP filter).

    Args:
        initial_budget: Initial budget ``E(z) = M * eps_q`` assigned to every
            document. A document is only allowed to participate in a query while
            its remaining budget covers the query's cost.
        tol: Numerical tolerance used by :meth:`can_query` when comparing the
            remaining budget against a cost.

    The manager is stateful across the whole query stream (multi-query setting)
    and is reset between independent runs via :meth:`reset`.
    """

    def __init__(self, initial_budget: float, tol: float = _EPS_TOL):
        if initial_budget <= 0:
            raise ValueError(f"initial_budget must be > 0, got {initial_budget}")
        self.initial_budget = float(initial_budget)
        self.tol = float(tol)
        # doc_id -> remaining budget. Unseen documents implicitly hold the full
        # initial budget; they are materialised lazily on first consume().
        self._remaining: dict[Hashable, float] = {}

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def remaining(self, doc_id: Hashable) -> float:
        """Return the remaining budget for ``doc_id`` (full budget if unseen)."""
        return self._remaining.get(doc_id, self.initial_budget)

    def can_query(self, doc_id: Hashable, cost: float) -> bool:
        """Whether ``doc_id`` still has budget to pay ``cost`` for a query.

        Implements the active-set condition ``E(z) >= cost`` from the paper's
        algorithms (``cost = eps_q`` for MURAG, ``eps_thr`` / ``eps_RAG`` for the
        two steps of MURAG-ADA).
        """
        return self.remaining(doc_id) >= float(cost) - self.tol

    def active_ids(
        self, cost: float, candidate_ids: Iterable[Hashable]
    ) -> list[Hashable]:
        """Return the subset of ``candidate_ids`` that can still pay ``cost``.

        The candidate set is supplied by the caller (e.g. every indexed document)
        because unseen documents are not stored explicitly.
        """
        return [doc_id for doc_id in candidate_ids if self.can_query(doc_id, cost)]

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def consume(self, doc_id: Hashable, cost: float) -> None:
        """Charge ``cost`` against ``doc_id``'s remaining budget.

        Callers are expected to have checked :meth:`can_query` first (the paper
        only charges documents that pass the active-set condition). The remaining
        budget is floored at 0 to avoid negative residue from float error.
        """
        new_remaining = self.remaining(doc_id) - float(cost)
        self._remaining[doc_id] = max(0.0, new_remaining)

    def consume_many(self, doc_ids: Iterable[Hashable], cost: float) -> int:
        """Charge ``cost`` against every id in ``doc_ids``. Returns the count charged."""
        n = 0
        for doc_id in doc_ids:
            self.consume(doc_id, cost)
            n += 1
        return n

    def reset(self) -> None:
        """Restore every document to the full initial budget."""
        self._remaining.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self, cost: float | None = None) -> dict[str, Any]:
        """Summarise budget consumption so far.

        Args:
            cost: If given, ``n_exhausted`` counts documents that can no longer
                pay this cost (i.e. would be dropped from the active set).

        Only documents that have been charged at least once are tracked, so the
        counts are relative to the touched set, not the whole (possibly unseen)
        corpus.
        """
        touched = len(self._remaining)
        total_consumed = sum(
            self.initial_budget - rem for rem in self._remaining.values()
        )
        out: dict[str, Any] = {
            "initial_budget": self.initial_budget,
            "n_tracked": touched,
            "total_consumed": float(total_consumed),
        }
        if cost is not None:
            out["n_exhausted"] = sum(
                1 for rem in self._remaining.values() if rem < float(cost) - self.tol
            )
        return out
