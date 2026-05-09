"""
Privacy accounting using the autodp library's Exponential Mechanism (EM) peeling.

Ports compose_subsampled_LimitedDomain_to_approxDP and EM_peeling from:
  tacchan7412/DPRAG/utils/dp_accountant.py  (Koga et al. 2024, arXiv:2412.04697)
Attribution: original code by tacchan7412, BSD-style license.

Used when dp.accounting_method = 'em_peeling_autodp'.  This gives a much tighter
(ε,δ)-DP bound for the LD-Gumbel mechanism than the generic BBGHS conversion used
by LDGumbelAccountant (simple_bound / tighter_bound methods).
"""

from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# RDP of the Exponential Mechanism via Mironov 2017, Remark 3.4 / autodp style
# ---------------------------------------------------------------------------

def _rdp_em(alpha: float, eps: float) -> float:
    """RDP of a single Exponential Mechanism query with privacy parameter eps.

    Matches tacchan7412/DPRAG/utils/dp_accountant.py::RDP_EM exactly.
    """
    if alpha == math.inf:
        return eps * 2.0
    # Mironov 2017 Remark 3.4
    try:
        num = math.sinh(alpha * eps) - math.sinh((alpha - 1) * eps)
        denom = math.sinh(eps)
        log_term = math.log(num / denom) / (alpha - 1)
    except (ValueError, ZeroDivisionError, OverflowError):
        log_term = float("inf")
    return min(0.5 * alpha * eps ** 2, log_term)


def _rdp_em_peeling(alpha: float, eps: float, k: int) -> float:
    """RDP of k-fold EM peeling (pay-what-you-get composition).

    Each of the k rounds applies one EM query at privacy eps.
    Matches EM_peeling_double / compose logic from reference.
    """
    return k * _rdp_em(alpha, eps)


def _rdp_subsampled_em(alpha: float, eps: float, k: int, subsampling_rate: float) -> float:
    """RDP of k subsampled EM queries (Poisson subsampling amplification).

    When subsampling_rate < 1, privacy is amplified.  For subsampling_rate=1
    this reduces to _rdp_em_peeling.
    """
    if abs(subsampling_rate - 1.0) < 1e-12:
        return _rdp_em_peeling(alpha, eps, k)
    # Subsampling amplification: approximate via log(1 - q + q*exp(rdp))
    rdp_single = _rdp_em(alpha, eps)
    q = subsampling_rate
    try:
        amplified = math.log(1.0 - q + q * math.exp((alpha - 1) * rdp_single)) / (alpha - 1)
    except (ValueError, OverflowError):
        amplified = rdp_single
    return k * amplified


def _rdp_to_approx_dp(rho_alpha: float, alpha: float, delta: float) -> float:
    """Convert (alpha, rho_alpha)-RDP to epsilon for (epsilon, delta)-DP.

    Uses BBGHS Theorem 21: ε = ρ/α + log(1/δ)/α.
    """
    if rho_alpha <= 0.0:
        return 0.0
    return rho_alpha / alpha + math.log(1.0 / delta) / alpha


def compose_subsampled_ld_to_approx_dp(
    *,
    eps_token: float,
    n_tokens: int,
    n_peels: int = 1,
    subsampling_rate: float = 1.0,
    target_delta: float = 1e-4,
    alpha_min: int = 2,
    alpha_max: int = 200,
    k: int | None = None,
) -> tuple[float, float]:
    """Compute (epsilon, delta)-DP for n_tokens LD-Gumbel queries via EM peeling.

    Searches for the optimal Rényi order alpha that minimises epsilon.

    Mirrors tacchan7412/DPRAG's `compose_subsampled_LimitedDomain_to_approxDP`
    (utils/dp_accountant.py:135), invoked at utils/dp.py:125 with default
    ``k=1``. The number of peels reflects how many top-i candidates the LD
    mechanism reports per token (LD-Gumbel reports a single top-1, hence
    ``n_peels=1``). It is *not* the LD domain size k_bar, which only affects
    utility, not privacy cost.

    Args:
        eps_token: Per-token LD-Gumbel privacy parameter.
        n_tokens: Number of tokens generated so far.
        n_peels: Number of EM top-1 peels per token. Configurable for
            experiments that want top-k LD; default 1 matches the reference.
        subsampling_rate: Poisson subsampling probability (1.0 = no subsampling).
        target_delta: Target δ for approx-DP conversion.
        alpha_min: Minimum Rényi order to search.
        alpha_max: Maximum Rényi order to search.
        k: Deprecated alias for ``n_peels`` (kept for backwards compatibility).

    Returns:
        Tuple (epsilon, delta) for the composed (ε,δ)-DP guarantee.
    """
    if k is not None:
        n_peels = int(k)
    if n_tokens == 0:
        return 0.0, target_delta

    best_eps = float("inf")
    for alpha in range(alpha_min, alpha_max + 1):
        rho = _rdp_subsampled_em(
            float(alpha), eps_token, k=n_peels * n_tokens, subsampling_rate=subsampling_rate
        )
        eps = _rdp_to_approx_dp(rho, float(alpha), target_delta)
        if eps < best_eps:
            best_eps = eps

    return best_eps, target_delta


# ---------------------------------------------------------------------------
# Drop-in replacement for LDGumbelAccountant using EM peeling
# ---------------------------------------------------------------------------

class AutodpLDAccountant:
    """Online privacy accountant using EM peeling (tight bound for LD-Gumbel).

    Implements the same public API as LDGumbelAccountant so it can be used
    as a drop-in replacement when accounting_method='em_peeling_autodp'.
    """

    def __init__(
        self,
        target_epsilon: float,
        target_delta: float,
        epsilon_token: float,
        delta_token: float,
        m: int = 30,
        k: int = 1,
        k_bar: int | None = None,
        sensitivity: float = 2.0,
        subsampling_rate: float = 1.0,
        n_peels: int = 1,
    ):
        if target_epsilon <= 0:
            raise ValueError(f"target_epsilon must be > 0, got {target_epsilon}")
        if target_delta <= 0 or target_delta >= 1:
            raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")
        if int(n_peels) < 1:
            raise ValueError(f"n_peels must be >= 1, got {n_peels}")

        self.target_epsilon = float(target_epsilon)
        self.target_delta = float(target_delta)
        self.epsilon_token = float(epsilon_token)
        self.delta_token = float(delta_token)  # not used in EM peeling but kept for API compat
        self.m = int(m)
        self.k = int(k)
        self.k_bar = int(k_bar) if k_bar is not None else int(m)
        self.sensitivity = float(sensitivity)
        self.subsampling_rate = float(subsampling_rate)
        # B13: number of EM peels per token. Default 1 matches the reference
        # (utils/dp.py:125 calls compose_subsampled_LimitedDomain_to_approxDP
        # without an explicit k, hence k=1). Earlier versions silently used
        # k=k_bar here, inflating ε by ~k_bar× and starving the budget.
        self.n_peels = int(n_peels)

        self.total_tokens_dp = 0
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.svt_spent_epsilon = 0.0  # Issue B12: composed on top of LD cost
        self.token_budget_log: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.total_tokens_dp = 0
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.svt_spent_epsilon = 0.0
        self.token_budget_log = []

    def _recompute_spent(self) -> tuple[float, float]:
        """Recompute (ε,δ) for the total tokens spent so far (LD only)."""
        return compose_subsampled_ld_to_approx_dp(
            eps_token=self.epsilon_token,
            n_tokens=self.total_tokens_dp,
            n_peels=self.n_peels,
            subsampling_rate=self.subsampling_rate,
            target_delta=self.target_delta,
        )

    def account_token(self) -> bool:
        self.total_tokens_dp += 1
        eps_ld, delta = self._recompute_spent()
        self.spent_epsilon = eps_ld + self.svt_spent_epsilon
        self.spent_delta = delta

        self.token_budget_log.append(
            {
                "token_idx": self.total_tokens_dp,
                "total_eps_spent": self.spent_epsilon,
                "budget_exceeded": self.spent_epsilon > self.target_epsilon,
            }
        )
        return self.spent_epsilon <= self.target_epsilon

    def account_svt_step(self, eps_lap: float) -> bool:
        """Account for one SVT iteration via 2*eps_lap classical composition.

        Mirrors LDGumbelAccountant.account_svt_step so SparseVote can compose
        SVT cost on top of either accountant interchangeably.
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

    def check_budget(self, **_: Any) -> dict:
        # Simulate one more token to check if budget permits.
        eps_next_ld, _ = compose_subsampled_ld_to_approx_dp(
            eps_token=self.epsilon_token,
            n_tokens=self.total_tokens_dp + 1,
            n_peels=self.n_peels,
            subsampling_rate=self.subsampling_rate,
            target_delta=self.target_delta,
        )
        eps_next = eps_next_ld + self.svt_spent_epsilon
        return {
            "ok": eps_next <= self.target_epsilon,
            "spent_epsilon": self.spent_epsilon,
            "spent_delta": self.spent_delta,
            "remaining_epsilon": self.target_epsilon - self.spent_epsilon,
            "next_token_cost": eps_next - self.spent_epsilon,
            "tokens_dp": self.total_tokens_dp,
            "details": {"method": "em_peeling_autodp"},
        }

    def get_summary(self) -> dict:
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
