"""zCDP Privacy Accounting for DP-SynRAG.

Reference:
- Bun & Steinke (2016), "Concentrated Differential Privacy:
  Simplifications, Extensions, and Lower Bounds", arXiv:1605.02065
- DP-SynRAG paper (arXiv:2510.06719v2), Section "Privacy Analysis", Theorem 1

Paper formula (Theorem 1):
    rho = K/(2 sigma_h^2)
        + L * [ epsilon_theta_s^2/8
              + 1/(2 sigma_mu^2)
              + T/2 * (c / tau)^2 ]
    epsilon = rho + sqrt(4 rho * ln(1/delta))

Component sources:
  - rho_h: Gaussian mechanism on keyword histogram (L2 sensitivity = sqrt(K))
  - rho_theta_s: exponential mechanism for similarity threshold (zCDP via Lemma 3)
  - rho_mu: Gaussian mechanism on per-cluster mean embedding (L2 sensitivity = 1)
  - rho_token: token prediction via clipped-logit aggregation with softmax

The three per-cluster terms are repeated L times because each document can
appear in up to L clusters (group-privacy amplification: each underlying
sample contributes to at most L independent zCDP mechanisms).

NOTE: Paper uses tau=0 in experiments, but Theorem 1 proof assumes tau>0.
When tau=0 (greedy/argmax), privacy proof does NOT apply exactly; code uses
approximation rho_token = T*c^2/2 as a fallback (see greedy_fallback note below).
"""

from __future__ import annotations

import math
from typing import NamedTuple


class ZCDPCompositionResult(NamedTuple):
    """Result of zCDP privacy accounting for DP-SynRAG."""

    epsilon: float
    """epsilon parameter for (epsilon, delta)-DP, after zCDP->DP conversion."""

    delta: float
    """delta parameter for (epsilon, delta)-DP."""

    rho: float
    """Total zCDP parameter rho."""

    method: str
    """Method identifier (always "zcdp" here)."""

    components: dict[str, float]
    """Breakdown of rho into its contributing terms (paper formula)."""


def zcdp_dp_synrag(
    K: int,
    sigma_h: float,
    L: int,
    epsilon_theta_s: float,
    sigma_mu: float,
    T: int,
    c: float,
    tau: float,
    delta: float = 1e-3,
) -> ZCDPCompositionResult:
    """Compute (epsilon, delta)-DP from zCDP for DP-SynRAG (paper formula).

    Args:
        K: keywords extracted per document (Stage 1a).
        sigma_h: Gaussian noise stddev on the keyword histogram (Stage 1a).
        L: max clusters a document can be assigned to (Stage 1b).
        epsilon_theta_s: epsilon for the per-cluster threshold exponential
            mechanism (Stage 1c).
        sigma_mu: Gaussian noise stddev on the per-cluster mean embedding
            (Stage 1c).
        T: synthetic tokens generated per cluster (Stage 2d).
        c: L_inf clipping bound on per-document logits (Stage 2d).
        tau: softmax temperature for token sampling (Stage 2d).
            tau <= 1e-6 triggers the greedy fallback (see note below).
        delta: failure probability for (epsilon, delta)-DP conversion.

    Returns:
        ZCDPCompositionResult.

    Greedy fallback (tau ~ 0):
        The paper's formula T/2 * (c/tau)^2 diverges as tau -> 0. In greedy
        mode we replace the per-token term by the zCDP bound of a pure
        epsilon-DP exponential mechanism with sensitivity c:
            rho_token_per_step = c^2 / 2
        and apply it T times:
            rho_token = T * c^2 / 2.
        This is documented in the README; users wanting an exact match with
        the paper's stochastic regime should set tau > 0.
    """
    if K <= 0:
        raise ValueError("K must be >= 1.")
    if sigma_h <= 0:
        raise ValueError("sigma_h must be > 0.")
    if L <= 0:
        raise ValueError("L must be >= 1.")
    if epsilon_theta_s <= 0:
        raise ValueError("epsilon_theta_s must be > 0.")
    if sigma_mu <= 0:
        raise ValueError("sigma_mu must be > 0.")
    if T <= 0:
        raise ValueError("T must be >= 1.")
    if c <= 0:
        raise ValueError("c must be > 0.")
    if tau < 0:
        raise ValueError("tau must be >= 0.")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}.")

    if tau <= 1e-6:
        import warnings
        warnings.warn(
            "tau <= 1e-6 (greedy/argmax mode): The paper's Theorem 1 assumes tau > 0. "
            "The privacy bound for tau=0 is an approximation (exponential mechanism fallback). "
            "For rigorous privacy guarantees, use tau > 0.",
            UserWarning,
            stacklevel=2,
        )

    rho_h = K / (2.0 * sigma_h * sigma_h)
    rho_theta_s = (epsilon_theta_s * epsilon_theta_s) / 8.0
    rho_mu = 1.0 / (2.0 * sigma_mu * sigma_mu)

    if tau <= 1e-6:
        rho_token = (T * c * c) / 2.0
        token_mode = "greedy_exp_mech_fallback"
    else:
        rho_token = (T / 2.0) * (c / tau) * (c / tau)
        token_mode = "softmax_paper_formula"

    rho_cluster = L * (rho_theta_s + rho_mu + rho_token)
    rho = rho_h + rho_cluster
    epsilon = rho + math.sqrt(4.0 * rho * math.log(1.0 / delta))

    return ZCDPCompositionResult(
        epsilon=epsilon,
        delta=delta,
        rho=rho,
        method="zcdp",
        components={
            "rho_h": rho_h,
            "rho_theta_s": rho_theta_s,
            "rho_mu": rho_mu,
            "rho_token_per_cluster": rho_token,
            "rho_cluster_total": rho_cluster,
            "rho_total": rho,
            "L_groups": float(L),
            "token_mode": token_mode,  # informational, not numeric
        },
    )


def print_zcdp_analysis(result: ZCDPCompositionResult) -> str:
    """Pretty-print a zCDP accounting result (mirrors print_privacy_analysis)."""
    lines = ["=" * 70]
    lines.append("DP-SynRAG ZCDP PRIVACY ACCOUNTING")
    lines.append("=" * 70)
    lines.append(f"  epsilon = {result.epsilon:.6f}")
    lines.append(f"  delta   = {result.delta:.2e}")
    lines.append(f"  rho     = {result.rho:.6f}")
    lines.append("\n  Components:")
    for key, val in result.components.items():
        if isinstance(val, (int, float)):
            lines.append(f"    {key}: {val:.6f}")
        else:
            lines.append(f"    {key}: {val}")
    lines.append("=" * 70)
    return "\n".join(lines)
