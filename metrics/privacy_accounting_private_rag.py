"""Privacy accounting for Private-RAG (MURAG / MURAG-ADA), arXiv:2511.07637.

Unlike the token-composition accountants used by the other methods, Private-RAG's
accounting is closed-form. The per-document **individual Renyi DP filter** (Feldman
& Zrnic, 2021, Corollary 3.3) is applied at Renyi order ``alpha = infinity``, which
is exactly pure DP; at that order composition is simply additive, so the method-
level guarantee is the per-document budget cap:

    Theorem 1 (MURAG)  / Theorem 2 (MURAG-ADA):
        the method satisfies ``epsilon``-DP provided the initial budget assigned to
        every document ``z`` is at most ``epsilon``.

With ``E(z) = M * eps_q`` the headline guarantee is therefore ``epsilon = M * eps_q``
(with ``delta = 0`` — everything is pure DP). ``M`` is the maximum number of queries
a single document may serve (paper main results: ``M = 1``).

The single-query subroutine (Algorithm 7) is itself ``eps``-DP by Lemma 1: it is an
AboveThreshold instance with at most ``c = floor(eps / eps_0)`` exponential-mechanism
releases, each ``eps_0``-DP. For MURAG the subroutine budget is ``eps_q``; for
MURAG-ADA the per-query budget splits as ``eps_q = eps_thr + eps_RAG`` (threshold
release + generation), and a retrieved document is charged the full ``eps_q``.
"""

from __future__ import annotations

import math
from typing import Any


def method_epsilon(m_retrievals: int, epsilon_q: float) -> float:
    """Headline (epsilon)-DP guarantee ``M * eps_q`` (Theorems 1 and 2)."""
    if m_retrievals <= 0:
        raise ValueError(f"m_retrievals (M) must be >= 1, got {m_retrievals}")
    if epsilon_q <= 0:
        raise ValueError(f"epsilon_q must be > 0, got {epsilon_q}")
    return float(m_retrievals) * float(epsilon_q)


def subroutine_max_releases(epsilon: float, epsilon_0: float) -> int:
    """Number of private token releases ``c = floor(eps / eps_0)`` (Algorithm 7)."""
    if epsilon <= 0 or epsilon_0 <= 0:
        raise ValueError(f"epsilon and epsilon_0 must be > 0, got {epsilon}, {epsilon_0}")
    return int(math.floor(epsilon / epsilon_0))


def subroutine_epsilon(epsilon: float, epsilon_0: float) -> float:
    """Worst-case spend of the Algorithm 7 subroutine: ``floor(eps/eps_0) * eps_0``.

    Equals ``epsilon`` when ``eps_0`` divides it; otherwise the largest multiple of
    ``eps_0`` not exceeding ``epsilon`` (the remainder cannot buy another release).
    Always ``<= epsilon``, so it is a valid upper bound.
    """
    return float(subroutine_max_releases(epsilon, epsilon_0)) * float(epsilon_0)


def summarize(
    *,
    variant: str,
    m_retrievals: int,
    epsilon_q: float,
    epsilon_0: float,
    epsilon_thr: float | None = None,
) -> dict[str, Any]:
    """JSON-serialisable accounting summary for one Private-RAG configuration.

    Args:
        variant: ``"murag"`` or ``"murag_ada"``.
        m_retrievals: maximum retrievals per document ``M``.
        epsilon_q: per-query budget.
        epsilon_0: per-token budget of the Algorithm 7 subroutine.
        epsilon_thr: MURAG-ADA threshold-release budget (required for ``murag_ada``).
    """
    variant = str(variant).lower()
    if variant not in {"murag", "murag_ada"}:
        raise ValueError(f"variant must be 'murag' or 'murag_ada', got {variant!r}")

    headline = method_epsilon(m_retrievals, epsilon_q)
    out: dict[str, Any] = {
        "variant": variant,
        "M": int(m_retrievals),
        "epsilon_q": float(epsilon_q),
        "epsilon_0": float(epsilon_0),
        "epsilon": headline,  # headline M * eps_q
        "delta": 0.0,         # pure DP (individual RDP filter at alpha = infinity)
    }

    if variant == "murag":
        out["epsilon_generation"] = subroutine_epsilon(epsilon_q, epsilon_0)
        out["max_releases"] = subroutine_max_releases(epsilon_q, epsilon_0)
    else:  # murag_ada
        if epsilon_thr is None:
            raise ValueError("epsilon_thr is required for the murag_ada variant.")
        if not (0.0 < epsilon_thr < epsilon_q):
            raise ValueError(
                f"epsilon_thr must be in (0, epsilon_q); got {epsilon_thr} vs {epsilon_q}"
            )
        epsilon_rag = float(epsilon_q) - float(epsilon_thr)
        out["epsilon_thr"] = float(epsilon_thr)
        out["epsilon_rag"] = epsilon_rag
        out["epsilon_generation"] = subroutine_epsilon(epsilon_rag, epsilon_0)
        out["max_releases"] = subroutine_max_releases(epsilon_rag, epsilon_0)
    return out
