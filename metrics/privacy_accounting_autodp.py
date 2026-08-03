"""
Privacy accounting for the Limited Domain Gumbel mechanism (DPVoteRAG).

Primary path — EM-peeling RDP (matches the reference runtime accountant):
    The reference `LimitedDomain` engine (tacchan7412/DPRAG/utils/dp.py) defaults to
    `compose_method='rdp'`, i.e. `get_dp_expense()` calls
    `compose_subsampled_LimitedDomain_to_approxDP` (utils/dp_accountant.py): each
    LD-Gumbel token is modelled as EM peeling (Mironov 2017 RDP for the exponential
    mechanism), composed over T tokens, converted via BBGHS'19 Theorem 21:
        ε(α) = max(ρ(α) + log((α-1)/α) − (log δ + log α)/(α−1), 0)
    Delta follows the reference split: the BBGHS conversion is evaluated at the
    residual `delta1` and the reported total is `delta1 + T·delta0·subsampling_rate`
    (with `delta1 = delta0 = delta_token` by default, exactly as in the reference).
    `AutodpLDAccountant(compose_method='rdp', ...)` implements this online.

Secondary path — advanced+sequential composition (reference `compose_method='simple'`):
    `num_queries_best` takes the tighter of advanced composition (Dwork et al.,
    "Fully Adaptive Composition", eq. 2) and sequential composition. Selectable via
    `compose_method='simple'`; retained for comparison, looser than the RDP path.

Attribution: EM/BBGHS analysis ported from tacchan7412/DPRAG/utils/dp_accountant.py
(itself from VITA-Group/DP-OPT); `num_queries_*` from utils/privacy.py (BSD-style).
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
        # Overflow at large eps: fall back to the conservative zCDP bound
        # (0.5*alpha*eps**2) selected by the min below — an upper bound, so sound.
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
# Online accountant: EM-peeling RDP + BBGHS (default), matching the reference runtime
# ---------------------------------------------------------------------------

class AutodpLDAccountant:
    """Online privacy accountant for LD-Gumbel.

    Default `compose_method='rdp'` (EM-peeling RDP + BBGHS conversion, delta split
    `delta1 + T·delta0`) matches the reference runtime, whose `LimitedDomain` engine
    defaults to `compose_method='rdp'` → `compose_subsampled_LimitedDomain_to_approxDP`
    (tacchan7412/DPRAG/utils/dp_accountant.py). `_t_max` walks T up while ε(T) and
    δ(T) stay within budget.

    `compose_method='simple'` selects the advanced+sequential composition
    (`num_queries_best`, ported from utils/privacy.py) instead — looser, kept for
    comparison.
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
        compose_method: str = "rdp",
        delta1: float | None = None,
        alpha_max: int = 200,
    ):
        if target_epsilon <= 0:
            raise ValueError(f"target_epsilon must be > 0, got {target_epsilon}")
        if target_delta <= 0 or target_delta >= 1:
            raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")
        if int(n_peels) < 1:
            raise ValueError(f"n_peels must be >= 1, got {n_peels}")
        if compose_method not in ("rdp", "simple"):
            raise ValueError(f"compose_method must be 'rdp' or 'simple', got {compose_method!r}")

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
        self.compose_method = compose_method
        # Reference default: residual delta1 == per-token delta0 (utils/dp.py:39).
        self.delta1 = float(delta1) if delta1 is not None else float(delta_token)
        self.alpha_max = int(alpha_max)

        self.total_tokens_dp = 0
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.svt_spent_epsilon = 0.0
        self.token_budget_log: list[dict[str, Any]] = []
        self._t_max_cache_key: float | None = None
        self._t_max_cache = 0

    def reset(self) -> None:
        self.total_tokens_dp = 0
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.svt_spent_epsilon = 0.0
        self.token_budget_log = []
        self._t_max_cache_key = None
        self._t_max_cache = 0

    def _ld_rdp_epsilon(self, n_tokens: int) -> float:
        """ε for n_tokens LD-Gumbel steps via EM-peeling RDP + BBGHS conversion.

        Mirrors `compose_subsampled_LimitedDomain_to_approxDP` (reference
        utils/dp_accountant.py): T-fold composition of the exponential-mechanism
        RDP, minimised over α and converted to (ε,δ) at the residual delta1.
        """
        if n_tokens <= 0:
            return 0.0
        best = float("inf")
        # alpha grid {2..alpha_max-1}, matching the reference `range(1, alpha_max)`
        # (alpha=1 ⇒ inf) for exact numeric parity with dp_accountant.py.
        for alpha in range(2, self.alpha_max):
            rho = _rdp_subsampled_em(
                float(alpha),
                self.epsilon_token,
                k=self.n_peels * n_tokens,
                subsampling_rate=self.subsampling_rate,
            )
            eps = _rdp_to_approx_dp(rho, float(alpha), self.delta1)
            if eps < best:
                best = eps
        return best

    def _total_delta(self, n_tokens: int) -> float:
        """Reported total delta: delta1 + T·delta0·subsampling_rate (reference split)."""
        if n_tokens <= 0:
            return 0.0
        return self.delta1 + self.delta_token * self.subsampling_rate * n_tokens

    def _t_max(self) -> int:
        """T_max allowed tokens under (target_epsilon - svt_spent, target_delta)."""
        remaining_eps = self.target_epsilon - self.svt_spent_epsilon
        if remaining_eps <= 0:
            return 0

        if self.compose_method == "simple":
            return num_queries_best(
                per_query_eps=self.epsilon_token,
                per_query_delta=self.delta_token,
                target_eps=remaining_eps,
                target_delta=self.target_delta,
            )

        # RDP path: ε(T) and δ(T) are monotone in T; walk up to the first breach.
        # Cache on remaining_eps so the per-token calls stay O(T_max · α_max) total.
        key = round(remaining_eps, 12)
        if self._t_max_cache_key == key:
            return self._t_max_cache
        t = 0
        while True:
            nxt = t + 1
            if self._total_delta(nxt) > self.target_delta:
                break
            if self._ld_rdp_epsilon(nxt) > remaining_eps:
                break
            t = nxt
        self._t_max_cache_key = key
        self._t_max_cache = t
        return t

    def _recompute_spent(self) -> tuple[float, float]:
        """Compute (ε,δ) consumed by total_tokens_dp LD steps.

        Default (compose_method='rdp'): EM-peeling RDP + BBGHS, with the reference
        delta split. compose_method='simple': best of advanced/sequential composition.
        """
        T = self.total_tokens_dp
        if T == 0:
            return 0.0, 0.0

        if self.compose_method == "rdp":
            return self._ld_rdp_epsilon(T), self._total_delta(T)

        # 'simple': advanced+sequential composition, ported from the reference
        # utils/dp.py::ExpMechanism.get_dp_expense (compose_method='simple').
        # k = n_peels·T composition steps, l = T queries. ε = min over three bounds
        # (sequential, advanced, zCDP-style), each using the FIXED residual delta1
        # (not the remaining budget). δ follows the reference split δ = 2·l·δ0 + δ1.
        k = self.n_peels * T
        eps = self.epsilon_token
        ln_inv_delta1 = math.log(1.0 / self.delta1)
        eps_candidates = [
            k * eps,
            k * eps * (math.exp(eps) - 1.0) / (math.exp(eps) + 1.0)
            + eps * math.sqrt(2.0 * k * ln_inv_delta1),
            k * eps ** 2 / 2.0 + eps * math.sqrt(0.5 * k * ln_inv_delta1),
        ]
        eps_spent = min(eps_candidates)
        delta_spent = 2.0 * T * self.delta_token + self.delta1
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
        """Account for one Sparse Vector Technique epoch (a threshold instantiation).

        Under the standard SVT analysis (Lyu-Su-Li 2017), one Above-Threshold epoch
        — a single noisy threshold τ̂ = τ + Lap(2/sparse_eps) used until the first
        above-threshold release — is `sparse_eps`-DP. Below-threshold comparisons and
        the agreement-count noise Lap(4/sparse_eps) are covered by that same budget and
        are NOT charged separately. The caller invokes this once per threshold
        instantiation (initial + each resample); epochs compose sequentially into
        `svt_spent_epsilon`.

        The reference (tacchan7412/DPRAG) does not compose this SVT cost online;
        charging it here is a deliberate, more conservative hardening that keeps the
        reported ε a valid upper bound. See dpvoterag/README.md.
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
            "details": {
                "method": (
                    "em_peeling_rdp_bbghs"
                    if self.compose_method == "rdp"
                    else "advanced_sequential_composition"
                )
            },
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


def release_ceiling(
    *,
    loop_cap: int,
    target_epsilon: float,
    target_delta: float,
    epsilon_token: float,
    delta_token: float,
    m: int = 30,
    k: int = 1,
    k_bar: int | None = None,
    compose_method: str = "rdp",
    sparse_epsilon: float | None = None,
) -> int:
    """Largest number of DP token releases the online budget can ever permit.

    The vote generators stop for two unrelated reasons. One is the data: an EOS, a
    Limited-Domain miss, a sparse test that stays below threshold. Those vary between
    neighbouring corpora, so the count a run happened to reach is not a bound and must
    not be charged. The other is the budget itself — a guard that refuses the next
    release once the composed cost would exceed ``target_epsilon``. That guard reads
    only the configuration, never the corpus, so the count it permits *is* a uniform
    ceiling, and it is normally far below the decoding length.

    Charging the decoding length instead leaves the reported epsilon inflated and, worse,
    identical for every point of a sweep over ``target_epsilon`` — the knob stops moving
    the method along the shared axis at all.

    ``sparse_epsilon`` selects the DPSparseVoteRAG call order, where one threshold epoch
    precedes the loop and another follows every private release, so the epochs eat into
    the same budget as the votes. Leave it None for plain DPVoteRAG.

    Returns 0 when the budget cannot buy a single release; the caller decides whether
    that is a skipped cell or an unreportable epsilon.
    """
    if loop_cap <= 0:
        return 0
    probe = AutodpLDAccountant(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epsilon_token=epsilon_token,
        delta_token=delta_token,
        m=m,
        k=k,
        k_bar=k_bar,
        compose_method=compose_method,
    )
    if sparse_epsilon is None:
        return max(0, min(loop_cap, probe._t_max()))

    probe.account_svt_step(sparse_epsilon)
    releases = 0
    while releases < loop_cap and probe.check_budget()["ok"]:
        probe.account_token()
        releases += 1
        probe.account_svt_step(sparse_epsilon)
    return releases


def ld_gumbel_feasible(*, epsilon_token: float, delta_token: float, m: int,
                       sensitivity: float = 2.0) -> tuple[bool, float]:
    """Whether a Limited-Domain Gumbel vote can ever be released, and its threshold.

    The bottom-out symbol sits at ``h_perp = h_k + 1 + log(min(sens,...)/delta)/eps``
    votes, and the largest count any token can reach is ``m``. Once the per-token
    budget is small enough that ``h_perp >= m``, no histogram the voters can produce
    clears the threshold: every step returns the null symbol and the answer is empty.

    The accounting stays valid there — an empty answer leaks nothing — so nothing
    raises, and a sweep can be composed, verified against the shared axis and shipped
    while every one of its cells is unable to emit a token. This is what makes the
    check worth running before a grid rather than after it.

    Returns ``(feasible, h_perp)``; the caller decides whether an infeasible point is
    a skipped cell or a target reported as out of reach.
    """
    if epsilon_token <= 0:
        return False, math.inf
    h_perp = 1.0 + math.log(sensitivity / delta_token) / epsilon_token
    return h_perp < m, h_perp
