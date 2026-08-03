"""The shipped grids land on the shared targets, so a point is a privacy level.

Without this the configs drift from analysis/derive_epsilon_grids.py silently: the
sweeps keep running and keep producing curves, and every cross-method comparison in
the write-up quietly compares different privacy levels against each other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.privacy_accounting_autodp import release_ceiling
from metrics.workload_accounting import DEFAULT_DELTA, method_epsilon_total

#: The common axis the full/ configs are derived against.
TARGETS = [1.0, 3.0, 10.0, 30.0, 100.0]
QUERIES = 100
#: The inversion is numeric and the YAML rounds; a point may miss by a few percent.
TOLERANCE = 0.05


def _cfg(name: str) -> dict:
    path = PROJECT_ROOT / "experiments" / "configs" / "full" / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dpvoterag_points():
    cfg = _cfg("dpvoterag")
    block = cfg["dpvoterag"]
    T = int(block.get("tmax_absolute") or block.get("max_length", 50))
    for total in cfg["epsilon_values"]:
        token = float(total) / T
        ceiling = release_ceiling(
            loop_cap=T, target_epsilon=float(total),
            target_delta=float(cfg.get("delta_total", 1e-4)),
            epsilon_token=token, delta_token=float(block["delta_token"]),
            m=int(block["m"]), k=int(block.get("k", 1)),
            compose_method=str(block.get("compose_method", "rdp")),
        )
        yield dict(epsilon_token=token, max_tokens=ceiling,
                   delta_token=float(block["delta_token"]))


def _dpsparsevoterag_points():
    cfg = _cfg("dpsparsevoterag")
    block = cfg["dpsparsevoterag"]
    c = int(block["max_private_tokens"])
    f = float(block["sparse_fraction"])
    for total in cfg["epsilon_values"]:
        token = float(total) / c
        yield dict(epsilon_token=(1.0 - f) * token, max_private_tokens=c,
                   delta_token=float(block["delta_token"]),
                   sparse_epsilon=f * token, n_epochs=1 + c)


def _dpksa_points():
    from metrics.privacy_accounting_dpksa import calibrate_dpksa

    cfg = _cfg("dpksa")
    for eps in cfg["epsilon_values"]:
        cal = calibrate_dpksa(
            target_epsilon=float(eps), target_delta=float(cfg["delta"]),
            ptr_noise_coupling=float(cfg["dpksa"].get("ptr_noise_coupling", 0.25)),
        )
        yield dict(gumbel_scale=cal["gumbel_scale"], sigma=cal["sigma"],
                   delta_ptr=cal["delta_ptr"])


def _private_rag_points(name="private_rag"):
    cfg = _cfg(name)
    m = int(cfg["private_rag"].get("M", 1))
    for eps_q in cfg["epsilon_q_values"]:
        yield dict(m_retrievals=m, epsilon_q=float(eps_q))


CASES = {
    "dpvoterag": _dpvoterag_points,
    "dpsparsevoterag": _dpsparsevoterag_points,
    "dpksa": _dpksa_points,
    "private_rag": _private_rag_points,
    "private_rag_ada": lambda: _private_rag_points("private_rag_ada"),
}


@pytest.mark.parametrize("method", sorted(CASES))
def test_the_shipped_grid_lands_on_the_shared_targets(method):
    points = list(CASES[method]())
    assert len(points) == len(TARGETS), f"{method} sweeps {len(points)} points, not {len(TARGETS)}"
    for target, params in zip(TARGETS, points):
        achieved = method_epsilon_total(
            method, queries=QUERIES, delta=DEFAULT_DELTA, **params
        ).epsilon
        assert achieved == pytest.approx(target, rel=TOLERANCE), (
            f"{method} point for target {target} composes to {achieved:.3f}"
        )


@pytest.mark.parametrize("method", sorted(CASES))
def test_the_grid_is_ordered(method):
    """A sweep that is not monotone in privacy plots a curve that doubles back."""
    achieved = [
        method_epsilon_total(method, queries=QUERIES, delta=DEFAULT_DELTA, **p).epsilon
        for p in CASES[method]()
    ]
    assert achieved == sorted(achieved), achieved


def test_the_vote_methods_hold_the_answer_length_across_the_sweep():
    """Every point must release the same number of tokens.

    If the budget decided the answer length instead, a point further along the sweep
    would score better partly because it answered at greater length, and the curve
    would mix utility-from-length with utility-from-less-noise.
    """
    ceilings = {p["max_tokens"] for p in _dpvoterag_points()}
    assert len(ceilings) == 1, f"answer length varies across the sweep: {sorted(ceilings)}"


@pytest.mark.parametrize("method", ["dpvoterag", "dpsparsevoterag"])
def test_the_per_token_budget_is_derived_not_pinned(method):
    cfg = _cfg(method)
    assert cfg[method].get("epsilon_token_from_budget") is True


def test_dp_synrag_points_reach_the_shared_targets():
    """The four stages divide one budget and must be derived together.

    Writing only `c` into the config left the three noise scales at the previous
    point's calibration: the reported epsilon stopped matching the target, and with
    the scales pinned there was a floor below which no c could reach at all.
    """
    from experiments.run_dp_synrag import _sweep_points
    from metrics.privacy_accounting_zcdp import zcdp_dp_synrag

    cfg = _cfg("dp_synrag")
    sg = cfg["synthetic_generation"]
    targets = sg["epsilon_targets"]
    assert targets == TARGETS, "dp_synrag sweeps a different axis from the other methods"

    for target, point in zip(targets, _sweep_points(cfg, sg)):
        achieved = zcdp_dp_synrag(
            K=int(cfg["keyword_extractor"]["K"]),
            sigma_h=point["sigma_h"],
            L=int(cfg["clustering"]["L"]),
            epsilon_theta_s=point["epsilon_theta_s"],
            sigma_mu=point["sigma_mu"],
            T=int(sg["T"]),
            c=point["c"],
            tau=float(sg["tau"]),
            delta=float(cfg["delta"]),
        ).epsilon
        assert achieved == pytest.approx(float(target), rel=TOLERANCE), (
            f"dp_synrag target {target} composes to {achieved:.3f}"
        )


def test_dp_synrag_moves_all_four_stage_parameters():
    from experiments.run_dp_synrag import _sweep_points

    cfg = _cfg("dp_synrag")
    points = _sweep_points(cfg, cfg["synthetic_generation"])
    for key in ("c", "sigma_h", "sigma_mu", "epsilon_theta_s"):
        values = [p[key] for p in points]
        assert len(set(values)) == len(values), f"{key} does not move across the sweep"


@pytest.mark.parametrize("method", ["dpvoterag", "dpsparsevoterag"])
def test_the_grid_is_reported_when_the_vote_cannot_fire(method):
    """A composable grid is not necessarily an executable one.

    The Limited-Domain Gumbel refuses to release once its bottom-out threshold passes
    the vote count m, and it does so without raising: the cells compose, verify against
    the shared axis, and return empty answers. The feasibility of every shipped point
    is recorded here so the state is a decision rather than a discovery mid-run.
    """
    from metrics.privacy_accounting_autodp import ld_gumbel_feasible

    cfg = _cfg(method)
    block = cfg[method]
    divisor = (
        int(block.get("tmax_absolute") or block.get("max_length", 50))
        if method == "dpvoterag"
        else int(block["max_private_tokens"])
    )
    share = 1.0 if method == "dpvoterag" else (1.0 - float(block["sparse_fraction"]))
    feasible = [
        ld_gumbel_feasible(
            epsilon_token=share * float(total) / divisor,
            delta_token=float(block["delta_token"]),
            m=int(block["m"]),
        )[0]
        for total in cfg["epsilon_values"]
    ]
    # Known and accepted: at Q=100 the shared targets sit far below this method's
    # feasible frontier, so every point returns the null symbol. The run is kept as
    # the measurement behind that claim; this asserts the state has not changed
    # silently in either direction.
    assert feasible == [False] * len(feasible), (
        f"{method} feasibility changed: {feasible} — re-derive the grid and update this test"
    )
