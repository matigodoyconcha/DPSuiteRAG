"""zCDP Privacy Accounting for DP-SynRAG.

Reference:
- Bun & Steinke (2016), "Concentrated Differential Privacy:
  Simplifications, Extensions, and Lower Bounds", arXiv:1605.02065
- DP-SynRAG paper (arXiv:2510.06719v2), Section "Privacy Analysis", Theorem 1

Theorem 1 formula:
    rho = K/(2 sigma_h^2)
        + L * [ epsilon_theta_s^2/8
              + 1/(2 sigma_mu^2)
              + T/2 * (c / tau)^2 ]        # requires tau > 1e-6; raises ValueError otherwise
    epsilon = rho + sqrt(4 rho * ln(1/delta))

Component sources:
  - rho_h: Gaussian mechanism on keyword histogram (L2 sensitivity = sqrt(K))
  - rho_theta_s: exponential mechanism for similarity threshold (zCDP via Lemma R)
  - rho_mu: Gaussian mechanism on per-cluster mean embedding (L2 sensitivity = 1)
  - rho_token: token prediction via clipped-logit aggregation with softmax

rho_theta_s = epsilon_theta_s^2 / 8:
  The threshold EM is eps_theta_s-DP (sensitivity-1 utility, sampled proportional
  to exp(eps_theta_s * u / 2)). An eps0-DP exponential mechanism has bounded range
  eps0, hence satisfies (eps0^2/8)-zCDP (Cesar & Rogers 2021, building on
  Bun & Steinke 2016). This matches Theorem 1 in the thesis (anexos.tex, Lemma R).

  Note: rho correctly INCREASES with eps (a larger eps means a sharper, less-private
  EM). The original DP-SynRAG arXiv paper (2510.06719v2) writes the reciprocal
  1/(8*eps^2). This is not merely dimensionally suspect (rho should grow with eps,
  not shrink) -- it is provably not what the paper computed: at the paper's own
  eps_theta_s=0.4 (Table 7) and L=5, the reciprocal term alone
  (L * 1/(8*0.4^2) = 3.9) already exceeds the total rho ~= 2.2 that the paper's own
  epsilon_total=10 budget allows, making the paper's own headline result infeasible
  under its printed formula. The value consistent with the paper's own token-term
  derivation (Lemma 3) is eps^2/8, which the thesis and this code use.

The three per-cluster terms are repeated L times because each document can
appear in up to L clusters (group-privacy amplification: each underlying
sample contributes to at most L independent zCDP mechanisms).

NOTE: The paper's own synthetic-generation exponential mechanism (Algorithm 1,
Stage 2d) runs at tau=1.0 (Table 7); the tau=0 the paper mentions elsewhere
(Sec. 6.1) is the unrelated downstream RAG-reader decoding temperature, which
operates over the already-DP synthetic corpus and is free post-processing.
When tau<=1e-6, privacy accounting raises ValueError — the formula T/2*(c/tau)^2
diverges and no valid privacy guarantee can be computed at tau=0, matching
Theorem 1's requirement that tau>0.
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
            "Theorem 1 accounting.  (The paper's own synthetic-generation exponential "
            "mechanism runs at tau=1.0, Table 7; tau=0 there refers only to the "
            "unrelated downstream RAG-reader decoding temperature.)"
        )
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}.")

    rho_h = K / (2.0 * sigma_h * sigma_h)
    # eps^2/8: standard zCDP bound for an eps-DP exponential mechanism via the
    # bounded-range refinement. Matches thesis Theorem 1 (see module docstring).
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


def rho_for_epsilon(epsilon: float, delta: float) -> float:
    """Invert ``epsilon = rho + sqrt(4 rho ln(1/delta))`` for rho.

    Substituting ``u = sqrt(rho)`` turns it into ``u^2 + 2u sqrt(ln(1/delta)) - eps = 0``,
    whose positive root gives

        rho = (sqrt(ln(1/delta) + eps) - sqrt(ln(1/delta)))^2
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    ln_inv_delta = math.log(1.0 / delta)
    return (math.sqrt(ln_inv_delta + epsilon) - math.sqrt(ln_inv_delta)) ** 2


def solve_c_for_epsilon(
    K: int,
    sigma_h: float,
    L: int,
    epsilon_theta_s: float,
    sigma_mu: float,
    T: int,
    tau: float,
    target_epsilon: float,
    delta: float = 1e-3,
) -> float:
    """Clipping bound ``c`` whose composed budget hits ``target_epsilon`` exactly.

    Turns ``epsilon`` from an output into an input, which is what lets DP-SynRAG be
    placed on the same axis as the rest of the benchmark: the other methods are handed
    a budget and calibrate their noise to it, while this one used to be swept over ``c``
    and have its epsilon read off afterwards.

    The map is closed form, so no search is needed — every stage but the token one has
    a budget fixed by its own parameters, and what remains goes to the token stage:

        rho_token = (rho_target - rho_h) / L - rho_theta_s - rho_mu
        c         = tau * sqrt(2 rho_token / T)

    Only the token stage moves, so a target below what the other three stages already
    spend is unreachable — and where that floor sits is a property of those stage
    parameters, not of the method. Use :func:`calibrate_dp_synrag` to scale all four
    stages together whenever the whole epsilon grid has to be reachable.

    Raises:
        ValueError: if the stages other than the token one already exceed
            ``target_epsilon``, leaving no budget to clip against. Lower ``K``, raise
            ``sigma_h``/``sigma_mu``, or lower ``epsilon_theta_s``.
    """
    if L <= 0:
        raise ValueError("L must be >= 1.")
    if T <= 0:
        raise ValueError("T must be >= 1.")
    if tau <= 1e-6:
        raise ValueError(f"tau={tau} <= 1e-6: the token term T/2*(c/tau)^2 diverges.")

    rho_target = rho_for_epsilon(target_epsilon, delta)
    rho_h = K / (2.0 * sigma_h * sigma_h)
    rho_theta_s = (epsilon_theta_s * epsilon_theta_s) / 8.0
    rho_mu = 1.0 / (2.0 * sigma_mu * sigma_mu)

    rho_token = (rho_target - rho_h) / L - rho_theta_s - rho_mu
    if rho_token <= 0.0:
        fixed_rho = rho_h + L * (rho_theta_s + rho_mu)
        raise ValueError(
            f"the histogram, threshold and mean stages alone need rho={fixed_rho:.6g}, "
            f"which already meets or exceeds the rho={rho_target:.6g} implied by "
            f"epsilon={target_epsilon:.6g} at delta={delta:.3g}. No budget is left for "
            "the token stage."
        )
    return tau * math.sqrt(2.0 * rho_token / T)


def calibrate_dp_synrag(
    K: int,
    sigma_h: float,
    L: int,
    epsilon_theta_s: float,
    sigma_mu: float,
    T: int,
    c: float,
    tau: float,
    target_epsilon: float,
    delta: float = 1e-3,
) -> dict[str, float]:
    """Scale all four build stages to a target budget, keeping their proportions.

    :func:`solve_c_for_epsilon` moves only the token stage, so the histogram, threshold
    and centroid stages keep whatever budget their own parameters imply, and below some
    target those three alone already exhaust it — leaving the method unable to reach the
    low end of the grid at all. Where that floor sits depends entirely on the stage
    parameters: with the paper's Table 7 values it is around ``epsilon = 4`` at
    ``delta = 1e-5``, but with noisier reference settings it can sit above 20, i.e.
    above the paper's own headline operating point.

    Here the reference parameters fix the *ratio* between the four stages and a single
    scalar moves the whole budget, so any target is reachable and no stage is tuned
    per operating point. Since every stage's rho is quadratic in its parameter, the
    scaling is ``sqrt(rho_target / rho_reference)`` throughout: noise scales divide by
    it, budgets and clipping bounds multiply by it.

    Returns the scaled ``sigma_h``, ``epsilon_theta_s``, ``sigma_mu`` and ``c``,
    together with the ``scale`` applied and the ``achieved_epsilon``.
    """
    reference = zcdp_dp_synrag(
        K=K, sigma_h=sigma_h, L=L, epsilon_theta_s=epsilon_theta_s,
        sigma_mu=sigma_mu, T=T, c=c, tau=tau, delta=delta,
    )
    rho_target = rho_for_epsilon(target_epsilon, delta)
    scale = math.sqrt(rho_target / reference.rho)

    scaled = {
        "sigma_h": sigma_h / scale,
        "epsilon_theta_s": epsilon_theta_s * scale,
        "sigma_mu": sigma_mu / scale,
        "c": c * scale,
    }
    achieved = zcdp_dp_synrag(
        K=K, L=L, T=T, tau=tau, delta=delta,
        sigma_h=scaled["sigma_h"],
        epsilon_theta_s=scaled["epsilon_theta_s"],
        sigma_mu=scaled["sigma_mu"],
        c=scaled["c"],
    )
    return {
        **scaled,
        "scale": scale,
        "rho_total": achieved.rho,
        "achieved_epsilon": achieved.epsilon,
        "target_epsilon": float(target_epsilon),
        "delta": float(delta),
    }


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
