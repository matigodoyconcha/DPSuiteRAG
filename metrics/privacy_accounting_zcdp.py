"""zCDP Privacy Accounting for DP-SynRAG.

Reference:
- Bun & Steinke (2016), "Concentrated Differential Privacy:
  Simplifications, Extensions, and Lower Bounds", arXiv:1605.02065
- DP-SynRAG paper (arXiv:2510.06719v2), Section "Privacy Analysis", Theorem 1

Paper formula (Theorem 1):
    rho = K/(2 sigma_h^2)
        + L * [ epsilon_theta_s^2/8        # code; paper writes 1/(8*eps^2) — see DELIBERATE DEVIATION note
              + 1/(2 sigma_mu^2)
              + T/2 * (c / tau)^2 ]        # requires tau > 1e-6; raises ValueError otherwise
    epsilon = rho + sqrt(4 rho * ln(1/delta))

Component sources:
  - rho_h: Gaussian mechanism on keyword histogram (L2 sensitivity = sqrt(K))
  - rho_theta_s: exponential mechanism for similarity threshold (zCDP via Lemma 3)
  - rho_mu: Gaussian mechanism on per-cluster mean embedding (L2 sensitivity = 1)
  - rho_token: token prediction via clipped-logit aggregation with softmax

DELIBERATE DEVIATION — rho_theta_s formula:
  Paper Theorem 1 writes:  rho_theta_s = 1 / (8 * epsilon_theta_s^2)
  This code computes:       rho_theta_s = epsilon_theta_s^2 / 8
  These are reciprocals and agree only at epsilon_theta_s = 1.

  Rationale: an eps0-DP mechanism satisfies (eps0^2/8)-zCDP via the bounded-range
  refinement (an eps0-DP exponential mechanism has bounded range eps0; Cesar & Rogers
  2021, building on Bun & Steinke 2016). The threshold EM here is eps_theta_s-DP
  (sensitivity-1 utility, sampled proportional to exp(eps_theta_s * u / 2)), so
  rho_theta_s = eps_theta_s^2 / 8. The paper's 1/(8*eps^2) is dimensionally inverted
  (rho must INCREASE with eps, since a larger eps means a sharper, less-private EM)
  and appears to be a typo in Theorem 1; the code matches the standard bound.

  Practical impact: the two formulas agree at epsilon_theta_s = 1 (paper default).
  They diverge for epsilon_theta_s sweeps: code gives SMALLER rho for eps > 1
  (less conservative) and LARGER rho for eps < 1.  For benchmarking, always
  calibrate epsilon_theta_s to epsilon_total rather than fixing it to 1.

The three per-cluster terms are repeated L times because each document can
appear in up to L clusters (group-privacy amplification: each underlying
sample contributes to at most L independent zCDP mechanisms).

NOTE: Paper uses tau=0 in experiments, but Theorem 1 proof assumes tau>0.
When tau<=1e-6, privacy accounting raises ValueError — the formula T/2*(c/tau)^2
diverges and no valid privacy guarantee can be computed.  Use tau > 0 or account
for greedy sampling via a separate mechanism (not implemented here).
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
            Must be > 1e-6; tau <= 1e-6 raises ValueError (see module note).
        delta: failure probability for (epsilon, delta)-DP conversion.

    Returns:
        ZCDPCompositionResult.
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
    if tau <= 1e-6:  # also covers tau < 0
        raise ValueError(
            f"tau={tau} <= 1e-6: the paper's formula T/2*(c/tau)^2 diverges at tau=0 "
            "and no valid privacy guarantee can be computed.  Set tau > 0 to use the "
            "Theorem 1 accounting.  (Paper experiments use tau=0 / greedy sampling but "
            "do not report a privacy bound for that regime.)"
        )
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}.")

    rho_h = K / (2.0 * sigma_h * sigma_h)
    # Deliberate deviation: code uses eps^2/8 (standard zCDP bound for EM).
    # Paper Theorem 1 writes 1/(8*eps^2). See module docstring for full rationale.
    rho_theta_s = (epsilon_theta_s * epsilon_theta_s) / 8.0
    rho_mu = 1.0 / (2.0 * sigma_mu * sigma_mu)

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
