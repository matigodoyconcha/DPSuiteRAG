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

from metrics.privacy_accounting_autodp import ld_gumbel_feasible, release_ceiling
from metrics.workload_accounting import (
    DEFAULT_DELTA,
    METHOD_KNOB_BUILDERS,
    METHOD_REGIMES,
    budget_for_target,
    delta_token_for_workload,
    dpksa_mechanism_delta_for_workload,
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


def _delta_token(block: dict, *, tokens: int, queries: int | None) -> float:
    """The per-token additive delta the runner will use for a cell of this size.

    Derived from the workload where the config asks for it, so that the mass the
    mechanism spends over the stream is the same in every cell. ``queries=None`` is
    the config's own fallback, which is what a paper-faithful run charges.
    """
    if queries is not None and block.get("delta_token_from_workload"):
        return delta_token_for_workload(queries=queries, tokens=tokens)
    return float(block["delta_token"])


def _structural(name: str, queries: int | None = None) -> dict:
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
        tokens = int(block.get("tmax_absolute") or block["max_length"])
        return dict(
            max_tokens=tokens,
            delta_token=_delta_token(block, tokens=tokens, queries=queries),
        )
    if name == "dpsparsevoterag":
        block = cfg["dpsparsevoterag"]
        tokens = int(block["max_private_tokens"])
        return dict(
            max_private_tokens=tokens,
            delta_token=_delta_token(block, tokens=tokens, queries=queries),
            sparse_fraction=float(block["sparse_fraction"]),
        )
    if name == "dpksa":
        block = cfg["dpksa"]
        fraction = float(block["ptr_delta_fraction"])
        # Its PTR test spends a share of the mechanism delta once per query, so the
        # delta itself is what the workload derives — the per-query analogue of the
        # per-token deltas above.
        mechanism_delta = (
            dpksa_mechanism_delta_for_workload(
                queries=queries, ptr_delta_fraction=fraction
            )
            if queries is not None and cfg.get("delta_from_workload")
            else float(cfg["delta"])
        )
        return dict(
            mechanism_delta=mechanism_delta,
            ptr_delta_fraction=fraction,
            ptr_noise_coupling=float(block["ptr_noise_coupling"]),
        )
    return dict(m_retrievals=int(cfg["private_rag"].get("M", 1)))


def _achieved(name: str, knob: float, queries: int) -> float:
    mechanisms = METHOD_KNOB_BUILDERS[name](knob, **_structural(name, queries))
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
                    delta=DEFAULT_DELTA, **_structural(method, q)
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
                              delta=DEFAULT_DELTA, **_structural(method, q))
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
                              delta=DEFAULT_DELTA, **_structural(method, q))
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

#: Cells where the Limited-Domain Gumbel can release a token, as (target, Q). The
#: region is a diagonal: a cell's per-token budget falls with Q while its bottom-out
#: threshold rises with the derived per-token delta, so the mechanism goes quiet from
#: the long-stream, tight-budget corner inwards. Recorded
#: rather than asserted away: a grid that composes correctly can still be one the
#: mechanism refuses to run, and it refuses silently — the vote returns the null symbol
#: and the answer comes back empty. Pinning the map here makes a change in either
#: direction a decision rather than something found after the run.
#:
#: The region is as small as it is because the vote histogram is not monotone in a
#: corpus document: `VOTE_SENSITIVITY_FACTOR` charges each release at twice its Gumbel
#: parameter, so the derivation hands the generator half the per-token budget, and the
#: bottom-out threshold `1 + log(2/delta_token)/eps_token` doubles with it.
FEASIBLE = {
    "dpvoterag": {(10.0, 1), (50.0, 1), (50.0, 5)},
    "dpsparsevoterag": {(10.0, 1), (50.0, 1), (50.0, 5)},
}


@pytest.mark.parametrize("method", sorted(FEASIBLE))
def test_the_feasible_region_of_the_vote_is_where_it_is(method):
    cfg = _cfg(method)
    block = cfg[method]
    share = 1.0 - float(block.get("sparse_fraction", 0.0))
    targets, workloads = _grid(method)

    observed = set()
    for q in workloads:
        # Both the budget and the bottom-out delta depend on the workload size, and
        # they are read once so the feasibility check prices the cell the runner runs.
        structural = _structural(method, q)
        divisor = structural.get("max_tokens") or structural["max_private_tokens"]
        for target in targets:
            knob = budget_for_target(
                method, target_epsilon=target, queries=q,
                delta=DEFAULT_DELTA, **structural
            )
            ok, _ = ld_gumbel_feasible(
                epsilon_token=share * knob / divisor,
                delta_token=structural["delta_token"],
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
    assert "for stream_q in [max(workload_queries)]:" in source, (
        "MuRAG sweeps the workload sizes as separate cells"
    )
    # ...and files that one stream at every size, so no per-Q panel loses the method.
    assert "for workload_q in workload_queries:" in source, (
        "MuRAG files its single stream at one workload size only"
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


# --- what is held fixed while the axis moves ---------------------------------------

#: The answer length every method is run at, and where each config declares it. Every
#: text metric is monotone in it, so a method allowed a longer answer scores higher for
#: a reason that has nothing to do with its mechanism. The paths are listed rather than
#: guessed from the key name because three of these configs also carry a length that is
#: NOT the answer: a synthetic document, a per-document intermediate response, a yes/no
#: filter. Those stay where their own stage needs them.
ANSWER_LENGTH = 30
ANSWER_LENGTH_KEYS = {
    "dp_rag": [("max_new_tokens",)],
    "dpvoterag": [("dpvoterag", "max_length")],
    # `c` bounds the private releases and `tmax_absolute` the loop they run in; both
    # are ceilings on the same answer, so both move with it.
    "dpsparsevoterag": [
        ("dpsparsevoterag", "max_private_tokens"),
        ("dpsparsevoterag", "tmax_absolute"),
    ],
    # Not `max_answer_tokens`: that is the per-document response DP-KSA histograms to
    # find its keywords, an intermediate the scorer never sees.
    "dpksa": [("dpksa", "max_final_tokens")],
    "private_rag": [("private_rag", "t_max")],
    "private_rag_ada": [("private_rag", "t_max")],
    # Not `synthetic_generation.T`, which is the length of a generated document.
    "dp_synrag": [("inference", "max_new_tokens")],
    "vanilla": [("generation", "max_new_tokens")],
}


@pytest.mark.parametrize("method", sorted(ANSWER_LENGTH_KEYS))
def test_every_method_answers_at_the_same_length(method):
    cfg = _cfg(method)
    for path in ANSWER_LENGTH_KEYS[method]:
        node = cfg
        for key in path:
            assert key in node, f"{method}: {'.'.join(path)} is not declared"
            node = node[key]
        assert int(node) == ANSWER_LENGTH, (
            f"{method}: {'.'.join(path)} is {node}, not the protocol's "
            f"{ANSWER_LENGTH}-token answer"
        )


@pytest.mark.parametrize("method", ["dpvoterag", "dpsparsevoterag", "dpksa"])
def test_the_additive_delta_spend_is_the_same_in_every_cell(method):
    """The mechanism's own delta cannot be a constant per release.

    Its cost over the workload is Q releases' worth, so a fixed value buys a hundredth
    of the guarantee at Q=1 that it buys at Q=100 while both cells are reported at the
    same axis delta. Deriving it the other way round fixes the spend, and it is the
    spend, not the per-release figure, that the reported delta covers.

    The two vote methods release once per token and DP-KSA once per query — a different
    mechanism spending a different parameter, held to the same constant so that a cell
    of one and a cell of another are the same condition.
    """
    from metrics.workload_accounting import MECHANISM_DELTA_SHARE

    _, workloads = _grid(method)
    if method == "dpksa":
        cfg = _cfg(method)
        assert cfg.get("delta_from_workload"), (
            f"{method} pins its mechanism delta; its additive spend then grows with Q"
        )
        fraction = float(cfg[method]["ptr_delta_fraction"])
        # The PTR test is what charges it, once per query, and it charges this share.
        spends = {
            q: q * fraction * _structural(method, q)["mechanism_delta"] for q in workloads
        }
    else:
        block = _cfg(method)[method]
        assert block.get("delta_token_from_workload"), (
            f"{method} pins delta_token; its additive spend then grows with Q"
        )
        tokens = int(block.get("max_private_tokens") or block["max_length"])
        spends = {
            q: q * tokens * _structural(method, q)["delta_token"] for q in workloads
        }
    for q, spend in spends.items():
        assert spend == pytest.approx(MECHANISM_DELTA_SHARE * DEFAULT_DELTA), (q, spends)
    # And what is left over is what the RDP -> (epsilon, delta) conversion runs on:
    # spend the whole axis delta additively and there is no epsilon to report at all.
    assert MECHANISM_DELTA_SHARE < 1.0


# --- what the payload actually says --------------------------------------------------

def _emitted_block(method: str, target: float, queries: int) -> dict:
    """The workload block the runner writes for one cell, built the way it builds it.

    Recomputing the numbers from the YAML checks the configuration; it cannot check the
    payload. The two differ where the runner charges something other than what the
    derivation assumed — the vote methods derive their per-release delta from the full
    answer length T but hand the accounting the release ceiling the online budget guard
    permits — so the emitted additive spend is `Q * ceiling * delta_token`. It equals
    the intended `Q * T * delta_token` only while the ceiling is the whole answer, and
    nothing that reads the config alone would notice it stopping.
    """
    from experiments.common import workload_accounting_block

    cfg = _cfg(method)
    structural = _structural(method, queries)
    knob = budget_for_target(
        method, target_epsilon=target, queries=queries,
        delta=DEFAULT_DELTA, **structural
    )
    if method == "dp_rag":
        tokens = structural["max_tokens"]
        return workload_accounting_block(
            method, queries=queries, run_seed=0, answer_tokens=tokens,
            epsilon_retrieval=structural["retrieval_ratio"] * knob,
            token_epsilon=knob, max_tokens=tokens,
        )
    if method in ("private_rag", "private_rag_ada"):
        return workload_accounting_block(
            method, queries=queries, run_seed=0, answer_tokens=None,
            m_retrievals=structural["m_retrievals"], epsilon_q=knob,
        )

    block = cfg[method]
    delta_token = structural["delta_token"]
    m, k = int(block["m"]), int(block["k"])
    k_bar = block.get("k_bar")
    if method == "dpvoterag":
        tokens = structural["max_tokens"]
        epsilon_token = knob / tokens
        ceiling = release_ceiling(
            loop_cap=tokens, target_epsilon=knob,
            target_delta=2.0 * tokens * delta_token,
            epsilon_token=epsilon_token, delta_token=delta_token,
            m=m, k=k, k_bar=k_bar,
            compose_method=str(block.get("compose_method", "rdp")),
        )
        return workload_accounting_block(
            method, queries=queries, run_seed=0, answer_tokens=tokens,
            workload_derived_deltas=("delta_token",),
            epsilon_token=epsilon_token, max_tokens=ceiling, delta_token=delta_token,
        )
    tokens = structural["max_private_tokens"]
    fraction = structural["sparse_fraction"]
    epsilon_token = knob / tokens
    ceiling = release_ceiling(
        loop_cap=tokens, target_epsilon=knob,
        target_delta=2.0 * tokens * delta_token,
        epsilon_token=(1.0 - fraction) * epsilon_token, delta_token=delta_token,
        m=m, k=k, k_bar=k_bar,
        compose_method=str(block.get("compose_method", "rdp")),
        sparse_epsilon=fraction * epsilon_token,
    )
    return workload_accounting_block(
        method, queries=queries, run_seed=0, answer_tokens=tokens,
        workload_derived_deltas=("delta_token",),
        epsilon_token=(1.0 - fraction) * epsilon_token,
        max_private_tokens=ceiling, delta_token=delta_token,
        sparse_epsilon=fraction * epsilon_token, n_epochs=1 + ceiling,
    )


@pytest.mark.parametrize("method", METHODS)
def test_the_written_cell_says_what_the_grid_promised(method):
    """Every shipped cell, checked on the payload a reader opens rather than on the YAML.

    Two claims live in that payload: the cell is at the target it is filed under, and
    the mechanism's own additive delta costs the same in every cell. Both are properties
    of the numbers the runner hands the accounting, and those are not the numbers the
    config holds — the release ceiling stands in for the answer length. A test that
    recomputes from the config asserts the derivation, not the record.
    """
    from metrics.workload_accounting import MECHANISM_DELTA_SHARE

    targets, workloads = _grid(method)
    for q in workloads:
        for target in targets:
            try:
                emitted = _emitted_block(method, target, q)
            except ValueError:
                continue  # out of reach; the runner logs the cell and skips it
            assert emitted.get("workload_error") is None, (
                method, target, q, emitted.get("workload_error")
            )
            assert emitted["epsilon_workload"] == pytest.approx(target, rel=TOLERANCE), (
                f"{method} target={target} Q={q} was written as "
                f"{emitted['epsilon_workload']:.4f}"
            )
            spend = emitted["workload_delta_additive"]
            if method in ("dpvoterag", "dpsparsevoterag"):
                assert spend == pytest.approx(MECHANISM_DELTA_SHARE * DEFAULT_DELTA), (
                    f"{method} target={target} Q={q} charges {spend:.4g} of additive "
                    f"delta, not the {MECHANISM_DELTA_SHARE * DEFAULT_DELTA:.4g} its "
                    "per-release delta was derived to spend"
                )
            else:
                # No mechanism of these spends delta by construction; the whole axis
                # delta is conversion slack, and a non-zero figure here would mean one
                # of them silently grew an approximate-DP component.
                assert spend == 0.0, (method, target, q, spend)


@pytest.mark.parametrize("method", ["dpvoterag", "dpsparsevoterag"])
def test_the_written_curve_reaches_every_workload_size(method):
    """The q1 / q100 columns are only comparable if they exist.

    A workload-derived delta is the mass one cell's Q was allowed; charged unchanged at
    another q it overruns the axis delta as soon as q >= 2Q, and the column goes blank
    for three quarters of the grid — which is a fact about the arithmetic, not about the
    mechanism, and is exactly what a reader would take it for.
    """
    targets, workloads = _grid(method)
    for q in workloads:
        for target in targets:
            emitted = _emitted_block(method, target, q)
            curve = emitted["epsilon_workload_by_q"]
            blank = sorted(size for size, eps in curve.items() if eps is None)
            assert not blank, f"{method} target={target} Q={q} prices no epsilon at {blank}"
            assert curve[str(q)] == emitted["epsilon_workload"]
