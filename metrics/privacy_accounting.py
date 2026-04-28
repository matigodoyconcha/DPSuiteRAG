"""
Privacy Accounting for DP-RAG using RDP (Rényi Differential Privacy).

Reference:
- Mironov, I., "Rényi Differential Privacy", 2017, https://arxiv.org/abs/1702.08896
- Google Differential Privacy Library: https://github.com/google/differential-privacy

This module provides formal ε-δ accounting for composite mechanisms:
- DP-RAG = DP-Retrieval + DP-ICL (T tokens)
"""

from __future__ import annotations

import math
from typing import NamedTuple


class DPCompositionResult(NamedTuple):
    """Result of DP composition analysis."""

    epsilon: float
    """ε parameter for (ε,δ)-DP"""

    delta: float
    """δ parameter for (ε,δ)-DP"""

    method: str
    """Method used: 'basic', 'rdp', or 'advanced'"""

    eps_retrieval: float
    """ε consumed by retrieval mechanism"""

    eps_generation: float
    """ε per token in generation"""

    eps_total_generation: float
    """Total ε consumed by generation (T * eps_generation)"""

    T: int
    """Number of tokens generated"""

    details: dict[str, float]
    """Additional details (RDP orders, etc.)"""


def basic_composition(
    eps_retrieval: float,
    eps_generation_per_token: float,
    num_tokens: int,
    delta: float = 1e-3,
) -> DPCompositionResult:
    """
    Compute ε-δ using basic composition.

    **WARNING: This gives loose bounds and is only for reference.**

    Basic composition states:
    epsilon_total ≤ epsilon_retrieval + T * epsilon_generation

    This is the naive union bound and is very loose for large T.

    Args:
        eps_retrieval: ε from retrieval mechanism
        eps_generation_per_token: ε per token in generation
        num_tokens: Number of tokens generated (T)
        delta: Failure probability (typically 1e-3)

    Returns:
        DPCompositionResult with basic composition bound
    """
    eps_total = eps_retrieval + (num_tokens * eps_generation_per_token)

    return DPCompositionResult(
        epsilon=eps_total,
        delta=delta,
        method="basic",
        eps_retrieval=eps_retrieval,
        eps_generation=eps_generation_per_token,
        eps_total_generation=num_tokens * eps_generation_per_token,
        T=num_tokens,
        details={
            "composition_type": "union_bound",
            "note": "Loose bound; use RDP for tighter accounting",
        },
    )


def rdp_to_eps_delta(
    rdp_order: float,
    rdp_value: float,
    delta: float = 1e-3,
) -> float:
    """
    Convert RDP(α, ρ) to (ε, δ)-DP using the conversion formula.

    Formula:
    ε = (1/(α-1)) * log((1/δ) * ρ)

    where:
    - α: RDP order (α > 1, typically 1.5, 2, 4, 8, ...)
    - ρ: RDP value
    - δ: Failure probability

    Args:
        rdp_order: RDP order α (must be > 1)
        rdp_value: RDP value ρ at that order
        delta: Failure probability δ

    Returns:
        ε value for (ε,δ)-DP
    """
    if rdp_order <= 1.0:
        raise ValueError(f"RDP order must be > 1, got {rdp_order}")
    if rdp_value < 0:
        raise ValueError(f"RDP value must be >= 0, got {rdp_value}")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0,1), got {delta}")

    if rdp_value == 0:
        return 0.0

    return (1.0 / (rdp_order - 1.0)) * math.log((1.0 / delta) * rdp_value)


def rdp_composition(
    eps_retrieval: float,
    eps_generation_per_token: float,
    num_tokens: int,
    delta: float = 1e-3,
    rdp_orders: tuple[float, ...] = (1.5, 2.0, 4.0, 8.0),
) -> DPCompositionResult:
    """
    Compute ε-δ using RDP (Rényi Differential Privacy) composition.

    **RECOMMENDED for tighter bounds.**

    For mechanisms with ε-DP guarantee:
    RDP_ρ(order α) = (1/(α-1)) * log(E[exp((α-1)*log(M(x,y)/M(x,y')))])

    For exponential mechanism with Δ=1 and ε:
    RDP_ρ(α) ≈ ε² * α / 2  (for small ε)

    This module provides better bounds than basic composition.

    Args:
        eps_retrieval: ε from retrieval mechanism
        eps_generation_per_token: ε per token in generation
        num_tokens: Number of tokens generated (T)
        delta: Failure probability (typically 1e-3)
        rdp_orders: RDP orders to compute (tuple of α values)

    Returns:
        DPCompositionResult with RDP composition bound
    """
    if not rdp_orders:
        raise ValueError("rdp_orders must be non-empty")

    # RDP value of a single exponential mechanism with ε and Δ=1
    # For exponential mechanism: RDP_ρ(α) = (1/(α-1)) * log(E[exp((α-1)*ln(p/q))])
    # With ε-DP guarantee, we can bound this using moment composition

    # For exponential mechanism with ε and sensitivity Δ=1:
    # RDP_ρ(α) ≈ ε² * α / 2 (moment method)
    def rdp_exponential_mechanism(alpha: float, epsilon: float) -> float:
        """RDP value of exponential mechanism with given epsilon."""
        if alpha <= 1.0:
            return float('inf')
        # Moment method bound for exponential mechanism
        return (epsilon ** 2) * alpha / 2.0

    # RDP composition over T independent calls:
    # RDP_ρ(α, T samples) = T * RDP_ρ(α, 1 sample)
    def rdp_composition_moments(
        rdp_single: float, num_samples: int, alpha: float
    ) -> float:
        """Compose RDP over multiple samples."""
        return num_samples * rdp_single

    # Compute RDP values for retrieval
    rdp_retrieval = {
        alpha: rdp_exponential_mechanism(alpha, eps_retrieval)
        for alpha in rdp_orders
    }

    # Compute RDP values for generation (T samples)
    rdp_generation = {
        alpha: rdp_composition_moments(
            rdp_exponential_mechanism(alpha, eps_generation_per_token),
            num_tokens,
            alpha,
        )
        for alpha in rdp_orders
    }

    # Total RDP (sum of independent mechanisms)
    rdp_total = {
        alpha: rdp_retrieval[alpha] + rdp_generation[alpha]
        for alpha in rdp_orders
    }

    # Convert RDP to ε-δ for each order and find the best (minimum ε)
    eps_candidates = [
        rdp_to_eps_delta(alpha, rdp_total[alpha], delta)
        for alpha in rdp_orders
    ]
    eps_min = min(eps_candidates)

    # Find which order gave the best result
    best_order = rdp_orders[eps_candidates.index(eps_min)]

    details = {
        "best_rdp_order": float(best_order),
        "rdp_retrieval": rdp_retrieval[best_order],
        "rdp_generation": rdp_generation[best_order],
        "rdp_total": rdp_total[best_order],
        "all_eps_candidates": {
            float(alpha): float(eps) for alpha, eps in zip(rdp_orders, eps_candidates)
        },
    }

    return DPCompositionResult(
        epsilon=eps_min,
        delta=delta,
        method="rdp",
        eps_retrieval=eps_retrieval,
        eps_generation=eps_generation_per_token,
        eps_total_generation=num_tokens * eps_generation_per_token,
        T=num_tokens,
        details=details,
    )


def advanced_composition(
    eps_retrieval: float,
    eps_generation_per_token: float,
    num_tokens: int,
    delta: float = 1e-3,
) -> DPCompositionResult:
    """
    Compute ε-δ using advanced composition (Theorem 3.3 from Dwork et al., 2010).

    **MOST ACCURATE for large T.**

    Advanced composition gives:
    For k mechanisms with (ε_i, δ_i)-DP:
    The composition satisfies (ε', δ')-DP where:
    - ε' = sqrt(2*k*ln(1/δ')*ε²) + k*ε*(exp(ε)-1)
    - δ' = 1 - (1 - δ_1) * ... * (1 - δ_k) ≈ Σδ_i for small δ_i

    Args:
        eps_retrieval: ε from retrieval mechanism
        eps_generation_per_token: ε per token in generation
        num_tokens: Number of tokens generated (T)
        delta: Failure probability (typically 1e-3)

    Returns:
        DPCompositionResult with advanced composition bound
    """
    k = 1 + num_tokens  # 1 retrieval + T generations
    epsilon_values = [eps_retrieval] + [eps_generation_per_token] * num_tokens
    delta_values = [delta / k] * k  # Allocate delta equally

    # Compute total delta (union bound approximation)
    delta_total = sum(delta_values)

    # Advanced composition formula (simplified version)
    # For uniform epsilon, this becomes:
    # ε² = 2*k*ln(1/δ)*ε²_component + k*ε_component*(exp(ε_component)-1)

    eps_avg = sum(epsilon_values) / len(epsilon_values)

    # Advanced composition bounds for uniform eps
    term1 = 2.0 * k * math.log(1.0 / delta_total) * (eps_avg ** 2)
    exp_eps = math.exp(eps_avg)
    term2 = k * eps_avg * (exp_eps - 1.0)

    eps_total = math.sqrt(term1) + term2

    details = {
        "k_mechanisms": k,
        "term1": term1,
        "term2": term2,
        "delta_total": delta_total,
        "method_note": "Advanced composition (Dwork et al., 2010)",
    }

    return DPCompositionResult(
        epsilon=eps_total,
        delta=delta_total,
        method="advanced",
        eps_retrieval=eps_retrieval,
        eps_generation=eps_generation_per_token,
        eps_total_generation=num_tokens * eps_generation_per_token,
        T=num_tokens,
        details=details,
    )


def compute_privacy_accounting(
    eps_retrieval: float,
    eps_generation_per_token: float,
    num_tokens: int,
    delta: float = 1e-3,
    methods: tuple[str, ...] = ("basic", "rdp", "advanced"),
) -> dict[str, DPCompositionResult]:
    """
    Compute privacy accounting using multiple composition methods.

    Allows comparison of different accounting methods to validate
    that RDP or advanced composition give tighter bounds than basic.

    Args:
        eps_retrieval: ε from retrieval mechanism
        eps_generation_per_token: ε per token in generation
        num_tokens: Number of tokens generated
        delta: Failure probability (typically 1e-3)
        methods: Which methods to compute ("basic", "rdp", "advanced")

    Returns:
        Dictionary mapping method name to DPCompositionResult

    Example:
        results = compute_privacy_accounting(
            eps_retrieval=1.0,
            eps_generation_per_token=0.5,
            num_tokens=64,
            delta=1e-3,
            methods=("basic", "rdp", "advanced"),
        )
        print(f"RDP bound: ε={results['rdp'].epsilon:.4f}")
        print(f"Advanced bound: ε={results['advanced'].epsilon:.4f}")
    """
    valid_methods = {"basic", "rdp", "advanced"}
    requested = set(methods)
    if not requested.issubset(valid_methods):
        invalid = requested - valid_methods
        raise ValueError(f"Invalid methods: {invalid}. Valid: {valid_methods}")

    results = {}

    if "basic" in requested:
        results["basic"] = basic_composition(
            eps_retrieval, eps_generation_per_token, num_tokens, delta
        )

    if "rdp" in requested:
        results["rdp"] = rdp_composition(
            eps_retrieval, eps_generation_per_token, num_tokens, delta
        )

    if "advanced" in requested:
        results["advanced"] = advanced_composition(
            eps_retrieval, eps_generation_per_token, num_tokens, delta
        )

    return results


def print_privacy_analysis(
    results: dict[str, DPCompositionResult],
    verbose: bool = False,
) -> str:
    """
    Pretty-print privacy accounting results.

    Args:
        results: Dictionary from compute_privacy_accounting
        verbose: Include additional details

    Returns:
        Formatted string
    """
    lines = ["=" * 70]
    lines.append("PRIVACY ACCOUNTING ANALYSIS")
    lines.append("=" * 70)

    for method_name in ["basic", "rdp", "advanced"]:
        if method_name not in results:
            continue

        result = results[method_name]
        lines.append(f"\n{method_name.upper()} Composition:")
        lines.append(f"  ε = {result.epsilon:.6f}")
        lines.append(f"  δ = {result.delta:.2e}")
        lines.append(f"  Method: {result.method}")

        if verbose:
            lines.append(f"  Components:")
            lines.append(f"    - Retrieval: ε_r = {result.eps_retrieval:.6f}")
            lines.append(f"    - Generation: {result.T} tokens × {result.eps_generation:.6f} = {result.eps_total_generation:.6f}")
            if result.details:
                lines.append(f"  Details:")
                for key, val in sorted(result.details.items()):
                    if not isinstance(val, dict):
                        lines.append(f"    {key}: {val}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
