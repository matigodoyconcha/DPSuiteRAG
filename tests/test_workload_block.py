"""Every method reaches the shared axis, with the parameters its config really holds."""

from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import workload_accounting_block
from metrics.workload_accounting import DEFAULT_DELTA, METHOD_MECHANISMS

RUNNERS = {
    "dp_rag": "run_dp_rag",
    "dpvoterag": "run_dpvoterag",
    "dpsparsevoterag": "run_dpsparsevoterag",
    "dpksa": "run_dpksa",
    "private_rag": "run_private_rag",
    "dp_synrag": "run_dp_synrag",
}


def _config(method: str) -> dict:
    name = "private_rag" if method == "private_rag_ada" else method
    path = PROJECT_ROOT / "experiments" / "configs" / "full" / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# Parameters taken from the shipped configs rather than invented, so the test moves
# when the configs move.
def _params(method: str) -> dict:
    cfg = _config(method)
    block = cfg.get(method, {}) or {}
    # Each config names privacy targets rather than a knob, so the parameters a
    # mechanism is built from are derived the way the runner derives them.
    from metrics.workload_accounting import DEFAULT_DELTA, budget_for_target

    # dp_synrag nests its targets under the stage block that consumes them.
    targets = cfg.get("epsilon_targets") or cfg["synthetic_generation"]["epsilon_targets"]
    target = float(targets[-1])
    if method == "dp_rag":
        from analysis.derive_epsilon_grids import _dp_rag_retrieval_ratio

        ratio = _dp_rag_retrieval_ratio(cfg)
        tokens = int(cfg.get("max_new_tokens", 70))
        knob = budget_for_target(
            "dp_rag", target_epsilon=target, queries=100, delta=DEFAULT_DELTA,
            retrieval_ratio=ratio, max_tokens=tokens,
        )
        return dict(
            epsilon_retrieval=ratio * knob, token_epsilon=knob, max_tokens=tokens,
        )
    if method == "dpvoterag":
        tokens = int(block.get("tmax_absolute") or block.get("max_length", 50))
        knob = budget_for_target(
            "dpvoterag", target_epsilon=target, queries=100, delta=DEFAULT_DELTA,
            max_tokens=tokens, delta_token=float(block["delta_token"]),
        )
        return dict(
            epsilon_token=knob / tokens, max_tokens=tokens,
            delta_token=float(block["delta_token"]),
        )
    if method == "dpsparsevoterag":
        cap = int(block["max_private_tokens"])
        fraction = float(block["sparse_fraction"])
        knob = budget_for_target(
            "dpsparsevoterag", target_epsilon=target, queries=100, delta=DEFAULT_DELTA,
            max_private_tokens=cap, delta_token=float(block["delta_token"]),
            sparse_fraction=fraction,
        )
        per_token = knob / cap
        return dict(
            epsilon_token=(1.0 - fraction) * per_token,
            max_private_tokens=cap,
            delta_token=float(block["delta_token"]),
            sparse_epsilon=fraction * per_token,
        )
    if method == "dpksa":
        from metrics.privacy_accounting_dpksa import calibrate_dpksa

        coupling = float(block.get("ptr_noise_coupling", 0.25))
        knob = budget_for_target(
            "dpksa", target_epsilon=target, queries=100, delta=DEFAULT_DELTA,
            mechanism_delta=float(cfg["delta"]), ptr_noise_coupling=coupling,
        )
        calibration = calibrate_dpksa(
            target_epsilon=knob, target_delta=float(cfg["delta"]),
            ptr_noise_coupling=coupling,
        )
        return dict(
            gumbel_scale=calibration["gumbel_scale"],
            sigma=calibration["sigma"],
            delta_ptr=calibration["delta_ptr"],
        )
    if method == "private_rag":
        m = int(cfg["private_rag"].get("M", 1))
        return dict(
            m_retrievals=m,
            epsilon_q=budget_for_target(
                "private_rag", target_epsilon=target, queries=100,
                delta=DEFAULT_DELTA, m_retrievals=m,
            ),
        )
    return dict(rho=1.5)  # dp_synrag: composed offline, passed straight through


@pytest.mark.parametrize("method", sorted(RUNNERS))
def test_the_shipped_parameters_reach_the_shared_axis(method):
    block = workload_accounting_block(method, queries=100, run_seed=1, **_params(method))
    assert block["workload_delta"] == DEFAULT_DELTA
    assert block["workload_queries"] == 100
    assert block.get("workload_error") is None, block.get("workload_error")
    assert block["epsilon_workload"] > 0


@pytest.mark.parametrize("method", sorted(RUNNERS))
def test_the_block_names_parameters_the_runner_actually_binds(method):
    """Catch a keyword pointing at a variable that exists in no scope.

    Passing an undefined name here raises only when the cell finishes, after the
    generation that produced it has been paid for — and a symbol-table sweep reports
    such a name as merely free, so it has to be checked against what the enclosing
    function assigns.
    """
    path = PROJECT_ROOT / "experiments" / f"{RUNNERS[method]}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "workload_accounting_block"
    )
    referenced = {
        n.id for kw in call.keywords for n in ast.walk(kw.value) if isinstance(n, ast.Name)
    }
    bound = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    bound |= {
        arg.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for arg in node.args.args + node.args.kwonlyargs
    }
    bound |= {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    bound |= {
        alias.asname or alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    bound |= {n.id for node in ast.walk(tree) if isinstance(node, ast.For)
              for n in ast.walk(node.target) if isinstance(n, ast.Name)}
    # `__builtins__` is the builtins module only in __main__; everywhere else it is a
    # dict, so dir() on it returned dict's methods and every real builtin read as
    # undefined. The hand-written fallback list hid that until a call used `min`.
    bound |= set(dir(builtins))

    missing = sorted(referenced - bound)
    assert not missing, f"{RUNNERS[method]} passes undefined name(s) {missing}"


def test_every_method_in_the_registry_has_a_runner():
    assert set(METHOD_MECHANISMS) - {"private_rag_ada"} == set(RUNNERS)


def test_an_impossible_budget_is_reported_not_raised():
    # A finished run must not be lost to a diagnostic: the failure comes back as data.
    block = workload_accounting_block(
        "dpvoterag", queries=100, run_seed=1,
        epsilon_token=2.0, max_tokens=50, delta_token=1e-3,
    )
    assert block["epsilon_workload"] is None
    assert "exhausts the target delta" in block["workload_error"]


def test_the_curve_prices_q1_without_composing():
    """Q=1 is the regime the source papers publish in, and must not compose anything."""
    from metrics.workload_accounting import dpvoterag_mechanisms, epsilon_total

    block = workload_accounting_block(
        "dpvoterag", queries=5, run_seed=1,
        epsilon_token=2.0, max_tokens=5, delta_token=1e-6,
    )
    one_query = epsilon_total(
        dpvoterag_mechanisms(epsilon_token=2.0, max_tokens=5, delta_token=1e-6),
        queries=1,
    ).epsilon
    assert block["epsilon_workload_q1"] == pytest.approx(one_query)
    assert block["epsilon_workload_by_q"]["1"] == pytest.approx(one_query)


def test_the_run_own_q_stays_the_headline():
    block = workload_accounting_block(
        "dpvoterag", queries=5, run_seed=1,
        epsilon_token=2.0, max_tokens=5, delta_token=1e-6,
    )
    assert block["epsilon_workload_by_q"]["5"] == block["epsilon_workload"]


@pytest.mark.parametrize("method,params,grows", [
    ("dpvoterag", dict(epsilon_token=2.0, max_tokens=5, delta_token=1e-6), True),
    ("private_rag", dict(m_retrievals=1, epsilon_q=10.0), False),
    ("dp_synrag", dict(rho=1.5), False),
])
def test_only_the_per_query_methods_degrade_with_the_workload(method, params, grows):
    """The whole point of the axis: a per-document filter and an offline build are flat.

    A per-query method pays again for every question; MuRAG's budget is spent per
    document and DP-SynRAG's before any question is asked, so their curves do not move.
    """
    block = workload_accounting_block(method, queries=5, run_seed=1, **params)
    curve = [block["epsilon_workload_by_q"][str(q)] for q in (1, 10, 100)]
    if grows:
        assert curve[0] < curve[1] < curve[2]
    else:
        assert curve[0] == pytest.approx(curve[1]) == pytest.approx(curve[2])
