"""
Privacy accounting for the Limited Domain Gumbel mechanism (DPVoteRAG).

Primary path — advanced+sequential composition (paper-faithful):
    Koga et al. (arXiv:2412.04697, §3.2) derives privacy via advanced composition
    (Dwork et al., "Fully Adaptive Composition", eq. 2) and sequential composition.
    The `AutodpLDAccountant` uses `num_queries_best`, which takes the tighter of the
    two, matching the reference implementation (tacchan7412/DPRAG/utils/privacy.py).

Secondary path — EM-peeling RDP (tighter in some regimes, not paper-specified):
    `compose_subsampled_ld_to_approx_dp` models each LD-Gumbel token as EM peeling
    (Mironov 2017 RDP) and converts via BBGHS'19 Theorem 21 (corrected formula):
        ε(α) = max(ρ(α) + log((α-1)/α) − (log δ + log α)/(α−1), 0)
    This path is retained for experimentation but is NOT the paper's method.

Attribution: `num_queries_*` ported from tacchan7412/DPRAG/utils/privacy.py (BSD-style).
"""

from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Advanced + Sequential composition (paper-faithful, ported from reference)
# ---------------------------------------------------------------------------

def num_queries_advanced_composition(
    per_query_eps: float,
    per_query_delta: float,
    target_eps: float,
    target_delta: float,
) -> int:
    """Maximum tokens under advanced composition (Dwork et al., eq. 2).

    Ported verbatim from tacchan7412/DPRAG/utils/privacy.py.
    """
    num_tokens = 0
    while True:
        delta = target_delta - num_tokens * per_query_delta
        if delta <= 0:
            return max(num_tokens - 1, 0)
        eps = math.sqrt(2.0 * math.log(1.0 / delta) * num_tokens * per_query_eps ** 2)
        eps += num_tokens * per_query_eps * (
            (math.exp(per_query_eps) - 1.0) / (math.exp(per_query_eps) + 1.0)
        )
        if eps > target_eps:
            return max(num_tokens - 1, 0)
        num_tokens += 1


def num_queries_sequential_composition(
    per_query_eps: float,
    per_query_delta: float,
    target_eps: float,
    target_delta: float,
) -> int:
    """Maximum tokens under sequential composition."""
    num_tokens = 0
    while True:
        if (num_tokens * per_query_eps > target_eps) or (
            num_tokens * per_query_delta > target_delta
        ):
            return max(num_tokens - 1, 0)
        num_tokens += 1


def num_queries_best(
    per_query_eps: float,
    per_query_delta: float,
    target_eps: float,
    target_delta: float,
) -> int:
    """Tighter of advanced and sequential composition (paper default)."""
    adv = num_queries_advanced_composition(
        per_query_eps, per_query_delta, target_eps, target_delta
    )
    seq = num_queries_sequential_composition(
        per_query_eps, per_query_delta, target_eps, target_delta
    )
    return max(adv, seq)


# ---------------------------------------------------------------------------
# RDP of the Exponential Mechanism via Mironov 2017, Remark 3.4 / autodp style
# (used by the secondary EM-peeling path only)
# ---------------------------------------------------------------------------

def _rdp_em(alpha: float, eps: float) -> float:
    """RDP of a single Exponential Mechanism query with privacy parameter eps."""
    if alpha == math.inf:
        return eps * 2.0
    try:
        num = math.sinh(alpha * eps) - math.sinh((alpha - 1) * eps)
        denom = math.sinh(eps)
        log_term = math.log(num / denom) / (alpha - 1)
    except (ValueError, ZeroDivisionError, OverflowError):
        log_term = float("inf")
    return min(0.5 * alpha * eps ** 2, log_term)


def _rdp_em_peeling(alpha: float, eps: float, k: int) -> float:
    """RDP of k-fold EM peeling."""
    return k * _rdp_em(alpha, eps)


def _rdp_subsampled_em(alpha: float, eps: float, k: int, subsampling_rate: float) -> float:
    """RDP of k subsampled EM queries (Poisson subsampling amplification)."""
    if abs(subsampling_rate - 1.0) < 1e-12:
        return _rdp_em_peeling(alpha, eps, k)
    rdp_single = _rdp_em(alpha, eps)
    q = subsampling_rate
    try:
        amplified = math.log(1.0 - q + q * math.exp((alpha - 1) * rdp_single)) / (alpha - 1)
    except (ValueError, OverflowError):
        amplified = rdp_single
    return k * amplified


def _rdp_to_approx_dp(rho_alpha: float, alpha: float, delta: float) -> float:
    """Convert (alpha, rho_alpha)-RDP to epsilon via BBGHS'19 Theorem 21.

    Corrected formula: ε = max(ρ(α) + log((α-1)/α) − (log δ + log α)/(α−1), 0)
    The earlier version incorrectly computed ε = ρ/α + log(1/δ)/α.
    """
    if rho_alpha <= 0.0:
        return 0.0
    if alpha <= 1.0:
        return float("inf")
    try:
        eps = (
            rho_alpha
            + math.log((alpha - 1.0) / alpha)
            - (math.log(delta) + math.log(alpha)) / (alpha - 1.0)
        )
        return max(eps, 0.0)
    except (ValueError, ZeroDivisionError, OverflowError):
        return float("inf")


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
    """(ε,δ)-DP for n_tokens LD-Gumbel queries via EM peeling (secondary path).

    This is the EM-peeling RDP path, retained for experimentation. The paper
    (Koga et al., arXiv:2412.04697) uses advanced+sequential composition instead
    (see `num_queries_best` and `AutodpLDAccountant`).
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
# Online accountant: paper-faithful advanced+sequential composition
# ---------------------------------------------------------------------------

class AutodpLDAccountant:
    """Online privacy accountant for LD-Gumbel using advanced+sequential composition.

    Uses `num_queries_best` (ported from the reference implementation) to decide
    T_max: the maximum number of DP tokens that fit within (target_epsilon, target_delta).

    Delta accumulates as T × delta_token (sequential delta composition), matching
    the reference (tacchan7412/DPRAG/utils/privacy.py).
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
        self.delta_token = float(delta_token)
        self.m = int(m)
        self.k = int(k)
        self.k_bar = int(k_bar) if k_bar is not None else int(m)
        self.sensitivity = float(sensitivity)
        self.subsampling_rate = float(subsampling_rate)
        self.n_peels = int(n_peels)

        self.total_tokens_dp = 0
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.svt_spent_epsilon = 0.0
        self.token_budget_log: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.total_tokens_dp = 0
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.svt_spent_epsilon = 0.0
        self.token_budget_log = []

    def _t_max(self) -> int:
        """T_max allowed tokens under (target_epsilon - svt_spent, target_delta)."""
        remaining_eps = self.target_epsilon - self.svt_spent_epsilon
        if remaining_eps <= 0:
            return 0
        return num_queries_best(
            per_query_eps=self.epsilon_token,
            per_query_delta=self.delta_token,
            target_eps=remaining_eps,
            target_delta=self.target_delta,
        )

    def _recompute_spent(self) -> tuple[float, float]:
        """Compute (ε,δ) consumed by total_tokens_dp LD steps.

        Uses the best of sequential and advanced composition (paper-faithful).
        Delta accumulates as T × delta_token (C2 fix).
        """
        T = self.total_tokens_dp
        if T == 0:
            return 0.0, 0.0

        # C2: delta accumulates sequentially
        delta_spent = T * self.delta_token

        # Sequential: ε = T × ε_token
        eps_seq = T * self.epsilon_token

        # Advanced composition with remaining delta slack
        delta_remaining = self.target_delta - delta_spent
        if delta_remaining > 0:
            try:
                eps_adv = math.sqrt(
                    2.0 * T * math.log(1.0 / delta_remaining) * self.epsilon_token ** 2
                ) + T * self.epsilon_token * (
                    (math.exp(self.epsilon_token) - 1.0)
                    / (math.exp(self.epsilon_token) + 1.0)
                )
            except (ValueError, OverflowError):
                eps_adv = eps_seq
        else:
            eps_adv = float("inf")

        eps_spent = min(eps_seq, eps_adv)
        return eps_spent, delta_spent

    def account_token(self) -> bool:
        self.total_tokens_dp += 1
        eps_ld, delta = self._recompute_spent()
        self.spent_epsilon = eps_ld + self.svt_spent_epsilon
        self.spent_delta = delta

        within_budget = self.total_tokens_dp <= self._t_max()
        self.token_budget_log.append(
            {
                "token_idx": self.total_tokens_dp,
                "total_eps_spent": self.spent_epsilon,
                "budget_exceeded": not within_budget,
            }
        )
        return within_budget

    def account_svt_step(self, eps_lap: float) -> bool:
        """Account for one SVT step at the per-token charge of the Koga paper.

        Koga et al. (arXiv:2412.04697 §3.3) allocates ε_token-Lap per token to
        the SVT side, with noise scales Lap(2/ε_token-Lap) for τ and
        Lap(4/ε_token-Lap) for the agreement count.  The per-step Laplace cost
        is therefore ε_token-Lap (= sparse_eps), composed sequentially across
        tokens — not 2×ε_token-Lap.
        """
        if eps_lap <= 0:
            return self.spent_epsilon <= self.target_epsilon
        increment = float(eps_lap)
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
        """Return whether one more DP token fits within the budget."""
        ok = (self.total_tokens_dp + 1) <= self._t_max()
        return {
            "ok": ok,
            "spent_epsilon": self.spent_epsilon,
            "spent_delta": self.spent_delta,
            "remaining_epsilon": self.target_epsilon - self.spent_epsilon,
            "tokens_dp": self.total_tokens_dp,
            "t_max": self._t_max(),
            "details": {"method": "advanced_sequential_composition"},
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
            "t_max": self._t_max(),
            "budget_log": self.token_budget_log,
        }
