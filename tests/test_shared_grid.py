"""The shipped grids are the shared axis crossed with the workload size.

Each config names privacy targets and the workload sizes the guarantee has to cover;
the runners derive their own budget knob for each pair. These tests hold the pairing
honest — that every cell composes to the target it claims, that a per-query method has
to spend less on each question as the stream grows while a stream-covering one does
not, and that a cell the mechanism cannot execute is known rather than discovered
after the GPU time is spent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.privacy_accounting_autodp import ld_gumbel_feasible
from metrics.workload_accounting import (
    DEFAULT_DELTA,
    METHOD_KNOB_BUILDERS,
    METHOD_REGIMES,
    budget_for_target,
    epsilon_total,
)

#: The inversion is numeric and the mechanisms are stepwise; a cell may miss by a few
#: percent without meaning the pairing has drifted.
TOLERANCE = 0.05

#: Methods whose grid is declared in `full/`. vanilla prices no mechanism.
METHODS = ["dp_rag", "dpvoterag", "dpsparsevoterag", "private_rag", "private_rag_ada"]


def _cfg(name: str) -> dict:
    path = PROJECT_ROOT / "experiments" / "configs" / "full" / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _grid(name: str) -> tuple[list[float], list[int]]:
    cfg = _cfg(name)
    targets = cfg.get("epsilon_targets") or (
        cfg.get("synthetic_generation", {}) or {}
    ).get("epsilon_targets")
    return [float(t) for t in targets], [int(q) for q in cfg["workload_queries"]]


def _structural(name: str) -> dict:
    """The parameters a method's knob builder needs, read from the config that runs."""
    cfg = _cfg(name)
    if name == "dp_rag":
        from analysis.derive_epsilon_grids import _dp_rag_retrieval_ratio

        return dict(
            retrieval_ratio=_dp_rag_retrieval_ratio(cfg),
            max_tokens=int(cfg["max_new_tokens"]),
        )
    if name == "dpvoterag":
        block = cfg["dpvoterag"]
        return dict(
            max_tokens=int(block.get("tmax_absolute") or block["max_length"]),
            delta_token=float(block["delta_token"]),
        )
    if name == "dpsparsevoterag":
        block = cfg["dpsparsevoterag"]
        return dict(
            max_private_tokens=int(block["max_private_tokens"]),
            delta_token=float(block["delta_token"]),
            sparse_fraction=float(block["sparse_fraction"]),
        )
    return dict(m_retrievals=int(cfg["private_rag"].get("M", 1)))


def _achieved(name: str, knob: float, queries: int) -> float:
    mechanisms = METHOD_KNOB_BUILDERS[name](knob, **_structural(name))
    return epsilon_total(
        mechanisms, queries=queries, delta=DEFAULT_DELTA,
        regime=METHOD_REGIMES.get(name, "per_query"),
    ).epsilon


# --- both axes are declared, and they are the same everywhere -----------------------

@pytest.mark.parametrize("method", METHODS + ["dp_synrag"])
def test_the_config_declares_the_axis_and_the_workload(method):
    targets, workloads = _grid(method)
    assert targets, f"{method} declares no epsilon_targets"
    assert workloads, f"{method} declares no workload_queries"


def test_every_method_sweeps_the_same_grid():
    """A cell of one method and a cell of another have to be the same condition."""
    grids = {m: _grid(m) for m in METHODS + ["dp_synrag"]}
    assert len({tuple(t) for t, _ in grids.values()}) == 1, grids
    assert len({tuple(q) for _, q in grids.values()}) == 1, grids


# --- every cell lands on the target it claims ---------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_every_cell_composes_to_its_target(method):
    targets, workloads = _grid(method)
    for q in workloads:
        for target in targets:
            try:
                knob = budget_for_target(
                    method, target_epsilon=target, queries=q,
                    delta=DEFAULT_DELTA, **_structural(method)
                )
            except ValueError:
                # Genuinely out of reach; the runner logs such a cell and skips it.
                continue
            achieved = _achieved(method, knob, q)
            assert achieved == pytest.approx(target, rel=TOLERANCE), (
                f"{method} target={target} Q={q} composes to {achieved:.4f}"
            )


# --- the property the whole experiment rests on -------------------------------------

@pytest.mark.parametrize("method", ["dp_rag", "dpvoterag", "dpsparsevoterag"])
def test_a_per_query_method_spends_less_on_a_longer_stream(method):
    targets, workloads = _grid(method)
    for target in targets:
        budgets = [
            budget_for_target(method, target_epsilon=target, queries=q,
                              delta=DEFAULT_DELTA, **_structural(method))
            for q in sorted(workloads)
        ]
        assert budgets == sorted(budgets, reverse=True), (target, budgets)


@pytest.mark.parametrize("method", ["private_rag", "private_rag_ada"])
def test_a_stream_covering_guarantee_does_not_move_with_the_workload(method):
    """MuRAG's per-document filter already covers the stream.

    The workload axis reaches this method through how much of the corpus is still
    unspent, not through its budget — which is why the runner resets the filter every
    Q queries rather than deriving a different number.
    """
    targets, workloads = _grid(method)
    for target in targets:
        budgets = [
            budget_for_target(method, target_epsilon=target, queries=q,
                              delta=DEFAULT_DELTA, **_structural(method))
            for q in workloads
        ]
        assert all(b == pytest.approx(budgets[0], rel=1e-9) for b in budgets), budgets


def test_dp_synrag_derives_all_four_stages_from_its_target():
    """An offline build is paid before any question, so no Q changes it."""
    from experiments.run_dp_synrag import _sweep_points
    from metrics.privacy_accounting_zcdp import zcdp_dp_synrag

    cfg = _cfg("dp_synrag")
    sg = cfg["synthetic_generation"]
    points = _sweep_points(cfg, sg)
    assert points, "dp_synrag derived no operating point"
    for key in ("c", "sigma_h", "sigma_mu", "epsilon_theta_s"):
        values = [p[key] for p in points]
        assert len(set(values)) == len(values), f"{key} does not move across the sweep"
    # Checked on the shared axis, not on the classic zCDP conversion. The two differ,
    # and the axis is the one every method is compared on; calibrating against the
    # other put this method's cells well below the target the rest hit exactly.
    for target, point in zip(sg["epsilon_targets"], points):
        rho = zcdp_dp_synrag(
            K=int(cfg["keyword_extractor"]["K"]), sigma_h=point["sigma_h"],
            L=int(cfg["clustering"]["L"]), epsilon_theta_s=point["epsilon_theta_s"],
            sigma_mu=point["sigma_mu"], T=int(sg["T"]), c=point["c"],
            tau=float(sg["tau"]), delta=float(cfg["delta"]),
        ).rho
        from metrics.workload_accounting import method_epsilon_total

        achieved = method_epsilon_total(
            "dp_synrag", queries=1, delta=DEFAULT_DELTA, rho=rho
        ).epsilon
        assert achieved == pytest.approx(float(target), rel=TOLERANCE)


# --- which cells the mechanism can actually execute ---------------------------------

#: Cells where the Limited-Domain Gumbel can release a token, as (target, Q). Recorded
#: rather than asserted away: a grid that composes correctly can still be one the
#: mechanism refuses to run, and it refuses silently — the vote returns the null symbol
#: and the answer comes back empty. Pinning the map here makes a change in either
#: direction a decision rather than something found after the run.
FEASIBLE = {
    "dpvoterag": {(10.0, 1), (50.0, 1), (50.0, 5)},
    "dpsparsevoterag": {(50.0, 1), (50.0, 5)},
}


@pytest.mark.parametrize("method", sorted(FEASIBLE))
def test_the_feasible_region_of_the_vote_is_where_it_is(method):
    cfg = _cfg(method)
    block = cfg[method]
    structural = _structural(method)
    divisor = structural.get("max_tokens") or structural["max_private_tokens"]
    share = 1.0 - float(block.get("sparse_fraction", 0.0))
    targets, workloads = _grid(method)

    observed = set()
    for q in workloads:
        for target in targets:
            knob = budget_for_target(
                method, target_epsilon=target, queries=q,
                delta=DEFAULT_DELTA, **structural
            )
            ok, _ = ld_gumbel_feasible(
                epsilon_token=share * knob / divisor,
                delta_token=float(block["delta_token"]),
                m=int(block["m"]),
            )
            if ok:
                observed.add((target, q))
    assert observed == FEASIBLE[method], (
        f"{method} feasible region moved: {sorted(observed)} "
        f"vs {sorted(FEASIBLE[method])}"
    )


def test_the_vote_stops_being_able_to_answer_as_the_stream_grows():
    """The finding the grid exists to show, stated as a property.

    Both vote methods can release at the loosest budget on the shortest stream and at
    none of the budgets on the longest, so the frontier runs diagonally through the
    grid rather than cutting off a whole row or column.
    """
    for method, cells in FEASIBLE.items():
        _, workloads = _grid(method)
        shortest, longest = min(workloads), max(workloads)
        assert any(q == shortest for _, q in cells), method
        assert not any(q == longest for _, q in cells), method


def test_dp_rag_never_spends_more_than_its_cell_was_given():
    """The generator solves for its own per-token budget, and may not overshoot.

    A search that stops at whichever endpoint is within tolerance can return a budget
    whose composition sits above the one it was handed, so a cell labelled 50 delivers
    50.02. Undershooting only leaves allowance unspent; overshooting reports a
    guarantee the run did not honour.
    """
    from analysis.derive_epsilon_grids import _dp_rag_retrieval_ratio
    from dp_methods.per_query.dp_rag.generator import DPAggregatedGenerator
    from metrics.privacy_accounting import pld_dp_rag_total

    cfg = _cfg("dp_rag")
    tokens, delta = int(cfg["max_new_tokens"]), float(cfg["delta"])
    ratio = _dp_rag_retrieval_ratio(cfg)
    targets, workloads = _grid("dp_rag")

    class _Budget:
        """Only the three attributes the solver reads."""

        def __init__(self, epsilon):
            self.epsilon, self.delta, self.max_new_tokens = epsilon, delta, tokens

    for target in targets:
        for q in workloads:
            knob = budget_for_target(
                "dp_rag", target_epsilon=target, queries=q,
                delta=DEFAULT_DELTA, retrieval_ratio=ratio, max_tokens=tokens,
            )
            asked = pld_dp_rag_total(
                eps_retrieval=0.0, token_epsilon=knob, num_tokens=tokens, delta=delta
            )
            solved = DPAggregatedGenerator._compute_token_epsilon(_Budget(asked))
            spent = pld_dp_rag_total(
                eps_retrieval=0.0, token_epsilon=solved, num_tokens=tokens, delta=delta
            )
            assert spent <= asked * (1 + 1e-9), (
                f"target={target} Q={q}: composes to {spent:.6f}, was given {asked:.6f}"
            )
            assert spent >= asked * 0.999, (
                f"target={target} Q={q}: leaves {100 * (1 - spent / asked):.2f}% unspent"
            )


def test_the_paper_faithful_sweeps_are_still_read():
    """A paper-faithful config names its own budget, and must not be reinterpreted.

    The two config sets express a sweep in different units. Reading only the shared
    protocol's key turns a paper-faithful run into a sweep of whatever the default is:
    it finishes, writes results, and reproduces nothing — silently, because a missing
    key looks exactly like a config that meant to take the default.
    """
    from experiments.common import privacy_sweep

    expected = {
        "dpvoterag": "epsilon_values",
        "dpsparsevoterag": "epsilon_values",
        "dpksa": "epsilon_values",
        "private_rag": "epsilon_q_values",
        "private_rag_ada": "epsilon_q_values",
        "dp_rag": "epsilon_generation_values",
    }
    for method, paper_key in expected.items():
        path = PROJECT_ROOT / "experiments" / "configs" / "paper_faithful" / f"{method}.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        kind, values, _ = privacy_sweep(cfg, paper_key=paper_key)
        assert kind == "knob", f"{method} paper-faithful config read as shared-axis targets"
        assert values == [float(v) for v in cfg[paper_key]], method


@pytest.mark.parametrize("method", METHODS)
def test_the_shared_configs_are_read_as_targets(method):
    from experiments.common import privacy_sweep

    paper_key = {
        "dp_rag": "epsilon_generation_values",
        "private_rag": "epsilon_q_values",
        "private_rag_ada": "epsilon_q_values",
    }.get(method, "epsilon_values")
    kind, values, workloads = privacy_sweep(_cfg(method), paper_key=paper_key)
    assert kind == "target"
    assert values == _grid(method)[0]
    assert workloads == _grid(method)[1]


def test_a_config_with_no_sweep_at_all_is_an_error():
    """Defaulting would run a grid nobody asked for, and it would look deliberate."""
    from experiments.common import privacy_sweep

    with pytest.raises(KeyError):
        privacy_sweep({"datasets": []}, paper_key="epsilon_values")


def test_murag_runs_one_stream_and_reads_the_axis_from_its_prefixes():
    """A method whose cost is flat in Q needs one stream, not one per workload size.

    Its per-document filter already covers the stream, so a shorter workload is a
    prefix of the longest — the utility over Q questions is the mean of the first Q
    entries of `prefix_curve`. Running the shorter workloads as separate cells would
    cost four times the GPU for the same numbers, and restarting the filter between
    them would hand back documents the guarantee had already spent, turning a flat cost
    into one that pays ceil(N/Q) times over.
    """
    source = (PROJECT_ROOT / "experiments" / "run_private_rag.py").read_text(encoding="utf-8")
    assert "for workload_q in [max(workload_queries)]:" in source, (
        "MuRAG sweeps the workload sizes as separate cells"
    )
    assert "budget_manager.reset()" not in source.split("mia_only_enabled()")[-1].split(
        "for q_idx"
    )[-1], "the per-document filter is restored mid-stream"
    assert "workload_queries_covered" in source


def test_only_the_flat_regimes_collapse_their_workload_sweep():
    """The per-query methods must still run every workload size separately.

    Their budget changes with Q, so the Q=100 stream is not a superset of the Q=1 one:
    at Q=1 each question holds a hundred times the budget. Collapsing them would
    compare a method against itself at the wrong privacy level.
    """
    for method in ("dp_rag", "dpvoterag", "dpsparsevoterag", "dpksa"):
        source = (PROJECT_ROOT / "experiments" / f"run_{method}.py").read_text(encoding="utf-8")
        assert "for workload_q in [max(" not in source, f"{method} collapsed its workload sweep"
