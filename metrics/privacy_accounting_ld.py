"""
Privacy accounting for Limited Domain (LD-) Gumbel mechanism (Durfee-Rogers NeurIPS 2019).

This module provides Rényi Differential Privacy (RDP) accounting for the LD-Gumbel mechanism
used in voting-based DP-RAG. It supports:
- Exact RDP composition of multiple LD-Gumbel queries
- Conversion from RDP to approximate (ε,δ)-DP via BBGHS'19 Theorem 21
- Online budget tracking with overflow detection

Reference:
- Durfee & Rogers, "Practical Differentially Private Top-k Selection with Pay-what-you-get 
  Composition" (NeurIPS 2019)
- Mironov, "Rényi Differential Privacy" (IEEE SPW 2017)
- BBGHS, "Near Optimal Differentially Private Principal Component Analysis" (NeurIPS 2019)
- tacchan7412/DPRAG implementation: https://github.com/tacchan7412/DPRAG
"""

from __future__ import annotations

import math
from typing import NamedTuple


class RDPCompositionResult(NamedTuple):
    """Result of RDP composition for LD mechanism."""

    rho_alpha: float
    """Rényi DP order-α divergence (log of expected ratio)."""

    alpha: float
    """Rényi DP order used."""


class ApproxDPResult(NamedTuple):
    """Conversion of RDP to approximate (ε,δ)-DP."""

    epsilon: float
    """ε parameter for (ε,δ)-DP."""

    delta: float
    """δ parameter for (ε,δ)-DP."""

    method: str
    """Method used: 'bbghs' or 'simple'."""

    order_alpha: float
    """Rényi order α used in conversion."""

    details: dict
    """Additional computation details."""


# Backwards-compatibility alias expected by tests
DPCompositionResult = ApproxDPResult


def rdp_exponential_mechanism(alpha: float, epsilon: float) -> float:
    """
    Rényi DP of the exponential mechanism (pure ε-DP).

    For a mechanism that is ε-DP, the RDP order-α divergence is:
    rho = α * ε / (α - 1)

    Args:
        alpha: Rényi order (α > 1).
        epsilon: Mechanism privacy budget ε.

    Returns:
        Rényi DP divergence ρ_α(M).
    """
    if alpha <= 1:
        raise ValueError(f"Rényi order must be > 1, got alpha={alpha}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got epsilon={epsilon}")

    return alpha * epsilon / (alpha - 1.0)


def rdp_limited_domain_gumbel(
    alpha: float,
    epsilon_token: float,
    delta_token: float,
    k_bar: int = None,
    sensitivity: float = 2.0,
    k: int = 1,
) -> float:
    """
    Rényi DP of the Limited Domain Gumbel mechanism (Durfee-Rogers NeurIPS 2019).

    The LD-Gumbel mechanism is (ε,δ)-DP with:
    - ε = epsilon_token (Gumbel scale = sensitivity / epsilon_token)
    - δ = delta_token (privacy-paying threshold includes log(1/delta_token))

    The RDP order-α divergence for an (ε,δ)-DP mechanism is approximated as:
    ρ_α ≈ α * ε / (α - 1) + δ  (conservative bound)

    For more precise bounds when δ is not negligible, we use:
    ρ_α = min(
        α * ε / (α - 1) + δ,                   (simple bound)
        ε²/(2·ln(1/δ)) + δ,                    (Mironov'17)
        α / (α - 1) * (ε + sqrt(ln(1/δ))·√(α/2))  (tighter)
    )

    Args:
        alpha: Rényi order (α > 1), typically 2-200.
        epsilon_token: Privacy budget ε per token.
        delta_token: Privacy parameter δ per token.
        k_bar: Domain size (used for informational purposes; doesn't affect RDP).
        sensitivity: Mechanism sensitivity (default 2 for l0-neighboring).
        k: Documents per query (informational; doesn't affect single-query RDP).

    Returns:
        Rényi DP divergence ρ_α for a single LD-Gumbel query.

    Raises:
        ValueError: If parameters are invalid.
    """
    if alpha <= 1:
        raise ValueError(f"Rényi order must be > 1, got alpha={alpha}")
    if epsilon_token <= 0:
        raise ValueError(f"epsilon_token must be > 0, got {epsilon_token}")
    if delta_token <= 0 or delta_token >= 1:
        raise ValueError(f"delta_token must be in (0, 1), got {delta_token}")

    # Conservative bound: (ε,δ)-DP → RDP
    # ρ_α = α·ε/(α-1) + δ
    simple_bound = alpha * epsilon_token / (alpha - 1.0) + delta_token

    # Tighter bound (Mironov'17, Prop 3.5): log(M(x)) for (ε,δ)-DP
    # ρ_α ≈ ε² / (2·ln(1/δ)) + δ  (when δ is not negligible)
    if delta_token > 1e-10:
        tighter_bound = (epsilon_token ** 2) / (2.0 * math.log(1.0 / delta_token)) + delta_token
    else:
        tighter_bound = float("inf")

    # Return the tightest bound
    return min(simple_bound, tighter_bound)


def compose_rdp_queries(rdp_per_query: list[float], alpha: float) -> float:
    """
    Compose RDP of multiple independent queries (sequential composition).

    For independent mechanisms M_1, ..., M_k with RDP ρ_{α,i}:
    ρ_α(M ∘ ... ∘ M) = ρ_{α,1} + ... + ρ_{α,k}

    Args:
        rdp_per_query: List of RDP divergences (one per query).
        alpha: Rényi order.

    Returns:
        Total RDP divergence ρ_α.
    """
    return sum(rdp_per_query)


def compose_subsampled_mechanism(
    rho_alpha: float,
    alpha: float,
    subsampling_prob: float = None,
    num_queries: int = None,
    prob: float = None,
    niter: int = None,
) -> float:
    """
    Amplification of RDP by subsampling (Theorem 9, Balle et al. 2018).

    When a mechanism is applied to a random subset (subsampling_prob) of a larger dataset,
    the RDP is amplified:

    ρ'_α ≈ log(1 - subsampling_prob + subsampling_prob * exp(ρ_α))

    Composed over num_queries independent subsampled queries:
    ρ_α^comp = num_queries * ρ'_α

    Args:
        rho_alpha: RDP divergence of a single query on full dataset.
        alpha: Rényi order.
        subsampling_prob: Probability of subsampling (typically 1/m for m voters).
        num_queries: Number of independent queries (number of tokens).

    Returns:
        Composed RDP divergence after subsampling amplification.
    """
    # Support legacy parameter names 'prob' and 'niter'
    if subsampling_prob is None and prob is not None:
        subsampling_prob = prob
    if num_queries is None and niter is not None:
        num_queries = niter

    if subsampling_prob is None:
        raise ValueError("subsampling_prob must be provided")
    if num_queries is None:
        raise ValueError("num_queries must be provided")

    if subsampling_prob <= 0 or subsampling_prob > 1:
        raise ValueError(f"subsampling_prob must be in (0, 1], got {subsampling_prob}")
    if num_queries <= 0:
        raise ValueError(f"num_queries must be > 0, got {num_queries}")

    # Amplified RDP for one subsampled query
    exp_rho = math.exp(min(rho_alpha, 100))  # Cap to avoid overflow
    rho_prime = math.log(1.0 - subsampling_prob + subsampling_prob * exp_rho)

    # Compose over num_queries
    return num_queries * rho_prime


def rdp_to_approx_dp_bbghs(
    rho_alpha: float,
    alpha: float,
    delta: float = 1e-5,
) -> ApproxDPResult:
    """
    Convert Rényi DP to approximate (ε,δ)-DP using BBGHS Theorem 21 (NeurIPS 2019).

    Given an (α, ρ_α)-RDP mechanism, we can derive an (ε, δ)-DP guarantee:
    ε = ρ_α / α + log(1/δ) / α

    This is tightest around α* that minimizes ε(α) over a range [2, 200].

    Args:
        rho_alpha: RDP divergence ρ_α.
        alpha: Rényi order α.
        delta: Target δ (default 1e-5, typical for privacy).

    Returns:
        ApproxDPResult with converted ε and δ, method='bbghs'.

    Reference:
        Balle et al., "Near Optimal Differentially Private Principal Component Analysis"
        (NeurIPS 2019), Theorem 21.
    """
    if rho_alpha < 0:
        raise ValueError(f"rho_alpha must be >= 0, got {rho_alpha}")
    if alpha <= 1:
        raise ValueError(f"alpha must be > 1, got {alpha}")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    # Standard conversion: ε = ρ/α + log(1/δ)/α
    log_inv_delta = math.log(1.0 / delta)
    epsilon = rho_alpha / alpha + log_inv_delta / alpha

    return ApproxDPResult(
        epsilon=epsilon,
        delta=delta,
        method="bbghs",
        order_alpha=alpha,
        details={"rho_alpha": rho_alpha, "log_1_delta": log_inv_delta},
    )


def search_optimal_alpha_rdp_to_dp(
    rho_fn,  # Callable[[float], float]: alpha -> rho_alpha
    delta: float = 1e-5,
    alpha_min: int = 2,
    alpha_max: int = 200,
    num_samples: int = 50,
) -> ApproxDPResult:
    """
    Search for the Rényi order α that minimizes ε when converting RDP to (ε,δ)-DP.

    For a given mechanism with RDP divergence ρ_α(α), we find:
    α* = argmin_α { ρ_α(α) / α + log(1/δ) / α }

    This searches over [alpha_min, alpha_max] by evaluating num_samples points.

    Args:
        rho_fn: Function that takes alpha and returns rho_alpha.
        delta: Target δ for conversion.
        alpha_min: Minimum Rényi order to search.
        alpha_max: Maximum Rényi order to search.
        num_samples: Number of points to sample in the range.

    Returns:
        ApproxDPResult for the best α found.
    """
    if alpha_min < 2 or alpha_max <= alpha_min:
        raise ValueError(f"alpha_min={alpha_min}, alpha_max={alpha_max} invalid")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    log_inv_delta = math.log(1.0 / delta)

    best_eps = float("inf")
    best_alpha = float(alpha_min)
    best_rho = None

    # Sample alpha values
    alphas = [alpha_min + (alpha_max - alpha_min) * i / max(1, num_samples - 1) for i in range(num_samples)]

    for alpha in alphas:
        alpha = max(2, alpha)  # Ensure alpha > 1
        try:
            rho_alpha = rho_fn(alpha)
            eps_candidate = rho_alpha / alpha + log_inv_delta / alpha
            if eps_candidate < best_eps:
                best_eps = eps_candidate
                best_alpha = alpha
                best_rho = rho_alpha
        except (ValueError, OverflowError):
            continue

    if best_rho is None:
        raise ValueError("No valid alpha found in search range")

    return ApproxDPResult(
        epsilon=best_eps,
        delta=delta,
        method="bbghs_optimized",
        order_alpha=best_alpha,
        details={"rho_alpha": best_rho, "num_alphas_sampled": len(alphas)},
    )


def make_accountant(
    method: str,
    *,
    target_epsilon: float,
    target_delta: float,
    epsilon_token: float,
    delta_token: float,
    m: int = 30,
    k: int = 1,
    k_bar: int | None = None,
    sensitivity: float = 2.0,
    subsampling_rate: float = 1.0,
):
    """Factory that returns an accountant instance for the requested accounting method.

    Issue 1: 'em_peeling_autodp' returns AutodpLDAccountant (tight EM bound);
    'simple_bound' / 'tighter_bound' return LDGumbelAccountant (current BBGHS approach).

    Args:
        method: 'em_peeling_autodp' | 'simple_bound' | 'tighter_bound' | 'no_accounting'
    """
    kwargs = dict(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epsilon_token=epsilon_token,
        delta_token=delta_token,
        m=m,
        k=k,
        k_bar=k_bar,
        sensitivity=sensitivity,
        subsampling_rate=subsampling_rate,
    )
    if method == "em_peeling_autodp":
        from metrics.privacy_accounting_autodp import AutodpLDAccountant
        return AutodpLDAccountant(**kwargs)
    elif method in ("simple_bound", "tighter_bound", "no_accounting"):
        return LDGumbelAccountant(**kwargs)
    else:
        raise ValueError(
            f"Unknown accounting_method '{method}'. "
            "Use 'em_peeling_autodp', 'simple_bound', 'tighter_bound', or 'no_accounting'."
        )


class LDGumbelAccountant:
    """Online privacy accountant for LD-Gumbel voting mechanism."""

    def __init__(
        self,
        target_epsilon: float,
        target_delta: float,
        epsilon_token: float,
        delta_token: float,
        m: int = 30,
        k: int = 1,
        k_bar: int = None,
        sensitivity: float = 2.0,
        subsampling_rate: float = 1.0,
    ):
        """Initialize online accountant for LD-Gumbel budget tracking.

        Args:
            target_epsilon: Total privacy budget for the query.
            target_delta: Total delta for the query.
            epsilon_token: Privacy budget per token.
            delta_token: Delta parameter per token.
            m: Number of voters (for subsampling amplification, if applicable).
            k: Documents per query.
            k_bar: Domain size (for informational purposes).
            sensitivity: Mechanism sensitivity.
        """
        if target_epsilon <= 0:
            raise ValueError(f"target_epsilon must be > 0, got {target_epsilon}")
        if target_delta <= 0 or target_delta >= 1:
            raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")

        self.target_epsilon = float(target_epsilon)
        self.target_delta = float(target_delta)
        self.epsilon_token = float(epsilon_token)
        self.delta_token = float(delta_token)
        self.m = int(m)
        self.k = int(k)
        self.k_bar = int(k_bar) if k_bar is not None else int(m)
        self.sensitivity = float(sensitivity)
        self.subsampling_rate = float(subsampling_rate)  # Issue 12

        # Online tracking
        self.total_tokens_dp = 0
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.svt_spent_epsilon = 0.0  # Issue B12: SVT cost composed on top of LD
        self.token_budget_log = []  # For debugging

    def reset(self) -> None:
        """Reset online accounting state for a new query/run."""
        self.total_tokens_dp = 0
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.svt_spent_epsilon = 0.0
        self.token_budget_log = []

    def account_token(self, alpha: float = 2.0) -> bool:
        """
        Account for one LD-Gumbel query (token generation).

        Computes the RDP of a single LD-Gumbel query with the stored epsilon_token
        and delta_token, converts to approx-DP, and accumulates in spent_epsilon/delta.

        Args:
            alpha: Rényi order for RDP-to-DP conversion (default 2.0).

        Returns:
            True if budget permits (spent < target), False if budget exhausted.

        Raises:
            ValueError: If parameters are invalid.
        """
        # Compute RDP for one LD-Gumbel token
        rho_alpha = rdp_limited_domain_gumbel(
            alpha=alpha,
            epsilon_token=self.epsilon_token,
            delta_token=self.delta_token,
            k_bar=self.k_bar,
            sensitivity=self.sensitivity,
            k=self.k,
        )

        # B8: Apply Poisson subsampling amplification (Balle et al. 2018)
        # when subsampling_rate < 1; q=1.0 (paper default) is a no-op.
        if self.subsampling_rate < 1.0:
            rho_alpha = compose_subsampled_mechanism(
                rho_alpha=rho_alpha,
                alpha=alpha,
                subsampling_prob=self.subsampling_rate,
                num_queries=1,
            )

        # Convert to approx-DP
        result = rdp_to_approx_dp_bbghs(rho_alpha, alpha, self.target_delta)

        # Accumulate (simple composition: sum of epsilons)
        old_spent = self.spent_epsilon
        self.spent_epsilon += result.epsilon

        # Log
        self.total_tokens_dp += 1
        self.token_budget_log.append(
            {
                "token_idx": self.total_tokens_dp,
                "rho_alpha": rho_alpha,
                "alpha": alpha,
                "eps_increment": result.epsilon,
                "total_eps_spent": self.spent_epsilon,
                "budget_exceeded": self.spent_epsilon > self.target_epsilon,
            }
        )

        # Check budget
        budget_ok = self.spent_epsilon <= self.target_epsilon
        return budget_ok

    def account_svt_step(self, eps_lap: float) -> bool:
        """Account for one Sparse Vector Technique iteration.

        Charges ``2 * eps_lap`` of pure (epsilon, 0)-DP per call via classical
        composition with the LD-Gumbel cost already tracked.  The factor of 2
        comes from the Dwork-Roth SVT analysis (Lap(2/eps) threshold noise +
        Lap(4/eps) query noise).  We compose conservatively: every comparison
        consumes the full SVT cost regardless of the outcome (above/below).

        Args:
            eps_lap: The Laplace-noise epsilon parameter used for the SVT step
                (i.e. ``sparse_eps`` in SparseVoteRAGGenerator).

        Returns:
            True if the budget still permits future tokens, False if the cap
            ``target_epsilon`` has been reached or exceeded.
        """
        if eps_lap <= 0:
            return self.spent_epsilon <= self.target_epsilon
        increment = 2.0 * float(eps_lap)
        self.svt_spent_epsilon += increment
        self.spent_epsilon += increment
        self.token_budget_log.append(
            {
                "kind": "svt",
                "eps_increment": increment,
                "total_eps_spent": self.spent_epsilon,
                "budget_exceeded": self.spent_epsilon > self.target_epsilon,
            }
        )
        return self.spent_epsilon <= self.target_epsilon

    def check_budget(self, alpha: float = 2.0) -> dict:
        """
        Check current budget status without spending.

        Args:
            alpha: Rényi order.

        Returns:
            Dict with keys:
            - 'ok': bool, whether budget permits one more token
            - 'spent_epsilon': float
            - 'spent_delta': float
            - 'remaining_epsilon': float
            - 'tokens_dp': int
            - 'details': dict
        """
        # Simulate accounting one more token
        rho_alpha = rdp_limited_domain_gumbel(
            alpha=alpha,
            epsilon_token=self.epsilon_token,
            delta_token=self.delta_token,
            k_bar=self.k_bar,
            sensitivity=self.sensitivity,
        )
        # B8: keep amplification consistent with account_token.
        if self.subsampling_rate < 1.0:
            rho_alpha = compose_subsampled_mechanism(
                rho_alpha=rho_alpha,
                alpha=alpha,
                subsampling_prob=self.subsampling_rate,
                num_queries=1,
            )
        result = rdp_to_approx_dp_bbghs(rho_alpha, alpha, self.target_delta)
        projected_eps = self.spent_epsilon + result.epsilon

        return {
            "ok": projected_eps <= self.target_epsilon,
            "spent_epsilon": self.spent_epsilon,
            "spent_delta": self.spent_delta,
            "remaining_epsilon": self.target_epsilon - self.spent_epsilon,
            "next_token_cost": result.epsilon,
            "tokens_dp": self.total_tokens_dp,
            "details": result._asdict(),
        }

    def get_summary(self) -> dict:
        """Get final accounting summary."""
        return {
            "total_tokens_dp": self.total_tokens_dp,
            "spent_epsilon": self.spent_epsilon,
            "spent_delta": self.spent_delta,
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "epsilon_budget_ok": self.spent_epsilon <= self.target_epsilon,
            "delta_budget_ok": self.spent_delta <= self.target_delta,
            "budget_log": self.token_budget_log,
        }
