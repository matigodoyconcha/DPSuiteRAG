"""
Privacy accounting for DP-RAG.

The DP-RAG headline ε is a single Privacy-Loss-Distribution (PLD) composition of
the retrieval exponential mechanism with the T per-token generation exponential
mechanisms — see ``pld_dp_rag_total``. This matches the reference implementation
(`dp_rag_engine.py`, which composes the two PLDs once).

Reference:
- Grislain, "Differentially Private Retrieval Augmented Generation", arXiv:2412.19291
"""

from __future__ import annotations


def pld_dp_rag_total(
    eps_retrieval: float,
    token_epsilon: float,
    num_tokens: int,
    delta: float = 1e-3,
) -> float:
    """Exact PLD composition of the full DP-RAG mechanism — the headline ε.

    DP-RAG composes one retrieval exponential mechanism with T per-token
    generation exponential mechanisms. ``token_epsilon`` is the per-token budget
    the generator already calibrated (binary search over PLD) so that the
    T-token generation alone equals (eps_g, delta)-DP. The honest total is a
    SINGLE Privacy-Loss-Distribution composition of

        retrieval(eps_r)  ⊕  T × generation(token_epsilon)

    read off at delta. This matches the reference (`dp_rag_engine.py`, which
    composes the two PLDs once) and yields ≈5.3 at the paper's operating point
    (eps_r=0.5, eps_g=5.0, T=70, delta=1e-3) — tighter than the naive sum
    eps_r + eps_g = 5.5 and far tighter than re-composing token_epsilon over T
    with basic/RDP/advanced bounds (which double-counts the calibration and
    inflates the budget).

    Falls back to the naive union bound eps_r + T·token_epsilon only when
    `dp_accounting` is unavailable (it is present wherever generation runs).
    """
    if num_tokens <= 0 or token_epsilon <= 0:
        return float(eps_retrieval)
    try:
        from dp_accounting.pld.privacy_loss_distribution import (
            from_privacy_parameters,
            identity,
        )
        from dp_accounting.pld.common import DifferentialPrivacyParameters
    except Exception:
        return float(eps_retrieval) + num_tokens * float(token_epsilon)

    pld = identity()
    if eps_retrieval > 0:
        pld = pld.compose(
            from_privacy_parameters(
                DifferentialPrivacyParameters(epsilon=float(eps_retrieval))
            )
        )
    single_token = from_privacy_parameters(
        DifferentialPrivacyParameters(epsilon=float(token_epsilon))
    )
    for _ in range(int(num_tokens)):
        pld = pld.compose(single_token)
    return float(pld.get_epsilon_for_delta(delta))
