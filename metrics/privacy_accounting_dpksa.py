"""Privacy accounting for DP-KSA (Tang et al., arXiv:2602.14374).

DP-KSA releases keywords via two DP mechanisms composed under Renyi DP
(Remark 5.2 of the paper):

- **FindBestK** — exponential mechanism (Gumbel-max) over the count-gap utility
  ``d_k = H_(k) - H_(k+1)``. Under the disjoint sample-and-aggregate partition the
  gap sensitivity is ``Delta d_k = 2`` (Lemma KSA-1). With Gumbel scale ``b`` the
  generic exponential-mechanism analysis certifies pure ``eps_em``-DP with
  ``eps_em = 2 * Delta / b = 4 / b``. (Tang et al. claim ``eps_k``-DP using
  Gumbel scale ``2/eps_k``; the reconstruction in the thesis records the residual
  factor of 2, so we always account with ``eps_em = 4 / b``.)
- **TopKWithPTR** — a Gaussian Propose-Test-Release test on ``max(2, d_k)`` with
  noise ``N(0, (2*sigma)^2)``. Its RDP curve is ``alpha / (2 * sigma^2)`` and it
  is ``delta_ptr``-approximate (the false-release probability).

The two are composed by RDP addition at a common order ``alpha`` and converted to
``(eps, delta)``-DP (Theorem 5.1 / A.6). The total ``delta`` has two additive
parts: ``delta_ptr`` (PTR false release) and ``delta_conv`` (RDP -> DP conversion
slack).

The exponential-mechanism RDP curve and the BBGHS'19 RDP -> DP conversion are
reused from :mod:`metrics.privacy_accounting_autodp`.
"""

from __future__ import annotations

import math
from typing import Any

from metrics.privacy_accounting_autodp import _rdp_em, _rdp_to_approx_dp


def compose_dpksa_to_dp(
    *,
    gumbel_scale: float,
    sigma: float,
    delta_ptr: float,
    delta_conv: float,
    gap_sensitivity: float = 2.0,
    alpha_min: int = 2,
    alpha_max: int = 200,
) -> tuple[float, float]:
    """Compose FindBestK + TopKWithPTR via RDP and convert to ``(eps, delta)``-DP.

    Args:
        gumbel_scale: Gumbel noise scale ``b`` used by FindBestK.
        sigma: PTR test noise parameter (the test noise is ``N(0, (2*sigma)^2)``).
        delta_ptr: PTR false-release probability (approximate-DP term).
        delta_conv: delta slack reserved for the RDP -> DP conversion.
        gap_sensitivity: sensitivity ``Delta d_k`` of the count gap (2 by Lemma KSA-1).
        alpha_min, alpha_max: integer Renyi orders scanned for the tightest bound.

    Returns:
        ``(epsilon, delta)`` with ``delta = delta_ptr + delta_conv``.
    """
    if gumbel_scale <= 0.0 or sigma <= 0.0:
        return float("inf"), delta_ptr + delta_conv

    # DPKSA-07: eps_em = 2*Delta/b = 4/b (Delta=2 by Lemma KSA-1). The paper's main
    # text claims eps_k-DP with Gumbel(2/eps_k) i.e. eps_em=2/b, but that contradicts
    # its own Def A.7 / Thm A.11 (Delta=2) — it under-reports by 2x. We account with
    # the correct 4/b (~1.3x more noise than the paper's literal 2/b). See module docstring.
    eps_em_pure = 2.0 * gap_sensitivity / gumbel_scale  # generic EM pure-DP parameter
    rdp_ptr_coeff = 1.0 / (2.0 * sigma * sigma)          # alpha-coefficient of the Gaussian test

    best_eps = float("inf")
    for alpha in range(alpha_min, alpha_max + 1):
        total_rdp = _rdp_em(float(alpha), eps_em_pure) + alpha * rdp_ptr_coeff
        eps = _rdp_to_approx_dp(total_rdp, float(alpha), delta_conv)
        if eps < best_eps:
            best_eps = eps

    return best_eps, delta_ptr + delta_conv


def calibrate_dpksa(
    *,
    target_epsilon: float,
    target_delta: float,
    gap_sensitivity: float = 2.0,
    ptr_delta_fraction: float = 0.5,
    ptr_noise_coupling: float = 1.0,
    alpha_min: int = 2,
    alpha_max: int = 200,
    search_iters: int = 60,
) -> dict[str, float]:
    """Calibrate FindBestK and TopKWithPTR noise to a target ``(eps, delta)``-DP.

    Treats ``epsilon`` as an input budget (as the paper sweeps eps in {1,2,3,5,8}).
    The PptR noise is coupled to the Gumbel scale by ``sigma = ptr_noise_coupling *
    gumbel_scale`` so that the composed ``epsilon`` is monotone in a single scalar,
    which is then bisected to match ``target_epsilon``.

    Returns a dict with ``gumbel_scale``, ``sigma``, ``eps_k`` (= the paper's
    FindBestK ``eps_k`` for Gumbel scale ``2/eps_k``), ``delta_ptr``, ``delta_conv``
    and the ``achieved_epsilon``.
    """
    if target_epsilon <= 0.0:
        raise ValueError(f"target_epsilon must be > 0, got {target_epsilon}")
    if not (0.0 < target_delta < 1.0):
        raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")
    if not (0.0 < ptr_delta_fraction < 1.0):
        raise ValueError(f"ptr_delta_fraction must be in (0, 1), got {ptr_delta_fraction}")

    delta_ptr = target_delta * ptr_delta_fraction
    delta_conv = target_delta - delta_ptr

    def achieved_eps(gumbel_scale: float) -> float:
        sigma = ptr_noise_coupling * gumbel_scale
        eps, _ = compose_dpksa_to_dp(
            gumbel_scale=gumbel_scale,
            sigma=sigma,
            delta_ptr=delta_ptr,
            delta_conv=delta_conv,
            gap_sensitivity=gap_sensitivity,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        return eps

    # achieved_eps is monotonically DECREASING in gumbel_scale (more noise -> less eps).
    b_lo, b_hi = 1e-4, 1e6  # b_lo => large eps, b_hi => small eps
    for _ in range(search_iters):
        b_mid = math.sqrt(b_lo * b_hi)  # geometric midpoint (log-scale bisection)
        if achieved_eps(b_mid) > target_epsilon:
            b_lo = b_mid  # too little noise -> increase scale
        else:
            b_hi = b_mid
    gumbel_scale = b_hi  # conservative side: achieved_eps(b_hi) <= target_epsilon
    sigma = ptr_noise_coupling * gumbel_scale
    achieved = achieved_eps(gumbel_scale)

    # Enforce (not just assume) the calibration's upper-bound property: the chosen
    # noise level must compose to epsilon <= target. If the search bracket failed
    # to cover the target, fail loudly rather than silently over-spend the budget.
    if achieved > target_epsilon + 1e-9:
        raise RuntimeError(
            f"DP-KSA calibration did not reach the target: achieved epsilon "
            f"{achieved:.6g} > target {target_epsilon:.6g}. Widen the search bracket."
        )

    return {
        "gumbel_scale": float(gumbel_scale),
        "sigma": float(sigma),
        "eps_k": float(2.0 / gumbel_scale),  # paper parameterizes Gumbel(2/eps_k)
        "delta_ptr": float(delta_ptr),
        "delta_conv": float(delta_conv),
        "achieved_epsilon": float(achieved),
        "target_epsilon": float(target_epsilon),
        "target_delta": float(target_delta),
    }


def summarize(calibration: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serialisable copy of a calibration dict (for run logs)."""
    return {key: float(val) for key, val in calibration.items()}
