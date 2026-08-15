"""The checked-in smoke set is what the generator produces from the full set today.

``experiments/configs/smoke/`` is derived, not written: every structural parameter comes
from ``experiments/configs/full/`` verbatim so the gate cannot test a protocol the grid
no longer runs. Nothing enforced that. A change to the full configs — an answer length,
a delta that stopped being pinned and started being derived — left the smoke set behind,
and `runner.py --smoke` went on green-lighting the previous protocol: it answered at the
old length, and the newly derived code paths were the one thing it never touched.

So the check is byte equality against a fresh generation, not a spot check of the few
keys that happened to move last time.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_DIR = PROJECT_ROOT / "experiments" / "configs" / "smoke"
FULL_DIR = PROJECT_ROOT / "experiments" / "configs" / "full"

REGENERATE = "run scripts/make_smoke_configs.py"

#: The reference conditions. Neither prices a mechanism, so neither has a budget knob or
#: a place on the shared axis: no_rag answers with no document in the prompt and vanilla
#: with the corpus unprotected.
NON_PRIVATE = {"no_rag", "vanilla"}


def _generator():
    """Import scripts/make_smoke_configs.py, which lives outside any package."""
    spec = importlib.util.spec_from_file_location(
        "make_smoke_configs", PROJECT_ROOT / "scripts" / "make_smoke_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_smoke_configs"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory) -> Path:
    """A fresh smoke set, generated from the current full configs into a temp dir."""
    out = tmp_path_factory.mktemp("smoke")
    gen = _generator()
    gen.SMOKE_DIR = out
    assert gen.main() == 0
    return out


def _smoke_config(name: str) -> dict:
    return yaml.safe_load((SMOKE_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def _full_config(name: str) -> dict:
    return yaml.safe_load((FULL_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def test_the_checked_in_smoke_set_is_the_generated_one(regenerated: Path) -> None:
    generated = sorted(p.name for p in regenerated.glob("*.yaml"))
    committed = sorted(p.name for p in SMOKE_DIR.glob("*.yaml"))
    assert generated == committed, (
        f"the smoke set has files the generator does not produce (or is missing some); "
        f"{REGENERATE}"
    )
    stale = [
        name
        for name in generated
        if (regenerated / name).read_text(encoding="utf-8")
        != (SMOKE_DIR / name).read_text(encoding="utf-8")
    ]
    assert not stale, (
        f"smoke configs are out of date with experiments/configs/full/: {stale}. "
        f"They are generated, so the fix is never to edit them: {REGENERATE}"
    )


# The parameters the smoke gate exists to exercise. Each is a mechanism parameter the
# runner now DERIVES from the workload size instead of reading a pinned number, so a
# smoke set that left them off ran a code path the grid does not.
@pytest.mark.parametrize(
    "name, dotted",
    [
        ("dpvoterag", "dpvoterag.delta_token_from_workload"),
        ("dpsparsevoterag", "dpsparsevoterag.delta_token_from_workload"),
        ("dpksa", "delta_from_workload"),
    ],
)
def test_the_workload_derived_deltas_are_enabled_in_the_smoke_set(
    name: str, dotted: str
) -> None:
    node = _smoke_config(name)
    for part in dotted.split("."):
        assert isinstance(node, dict) and part in node, (
            f"smoke/{name}.yaml does not carry {dotted}; the gate would run the pinned-"
            f"delta path and the grid the derived one. {REGENERATE}"
        )
        node = node[part]
    assert node is True, f"smoke/{name}.yaml sets {dotted}={node!r}; {REGENERATE}"


def test_every_method_answers_at_one_length() -> None:
    """The protocol holds the answer length equal across every method and both baselines.

    Every text metric is monotone in it, so a method left at another value is not
    measured on the same scale as the rest — and the smoke run is where that shows up
    for free, since each config names the cap under its own key.
    """
    # Where each method's configured answer length lives — the bound on the string
    # that reaches the scorer, whatever mechanism wrote each token. Two keys look
    # like the answer length and are not: DP-KSA's `max_answer_tokens` is the
    # per-document response its histogram votes over, and DPSparseVoteRAG's
    # `max_private_tokens` counts only the Limited-Domain releases, while its loop
    # also emits budget-free public tokens and runs to `tmax_absolute`.
    caps = {
        "no_rag": "generation.max_new_tokens",
        "vanilla": "generation.max_new_tokens",
        "dp_rag": "max_new_tokens",
        "dp_synrag": "inference.max_new_tokens",
        "dpksa": "dpksa.max_final_tokens",
        "dpvoterag": "dpvoterag.max_length",
        "dpsparsevoterag": "dpsparsevoterag.tmax_absolute",
        "private_rag": "private_rag.t_max",
        "private_rag_ada": "private_rag.t_max",
    }
    lengths = {}
    for name, dotted in caps.items():
        node = _smoke_config(name)
        for part in dotted.split("."):
            node = node[part]
        lengths[name] = int(node)
    assert len(set(lengths.values())) == 1, (
        f"the methods answer at different lengths: {lengths}. The full configs are the "
        f"source of this value, so fix it there and {REGENERATE}"
    )


def test_the_smoke_points_are_coordinates_the_grid_also_runs() -> None:
    """The two budget points and the two workload sizes are taken from the full set.

    A smoke point off the grid's own axis is the one staleness byte equality cannot see:
    the generator names the points itself, so moving the full set's axis leaves the
    generator's output unchanged and the gate calibrating a coordinate nobody will run.
    DP-SynRAG is the deliberate exception — its noise scales are absolute, so over a
    smoke corpus it needs a far looser point or no cluster survives its threshold.
    """
    for path in sorted(SMOKE_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        full = _full_config(path.stem)
        if full.get("workload_queries"):
            missing = set(cfg["workload_queries"]) - set(full["workload_queries"])
            assert not missing, (
                f"{path.name}: workload size(s) {sorted(missing)} are not on the "
                f"protocol's workload axis {full['workload_queries']}; {REGENERATE}"
            )
        if path.stem in NON_PRIVATE | {"dp_synrag"}:
            continue
        missing = set(cfg["epsilon_targets"]) - set(full["epsilon_targets"])
        assert not missing, (
            f"{path.name}: budget point(s) {sorted(missing)} are not on the shared axis "
            f"{full['epsilon_targets']}, so the gate would calibrate a coordinate the "
            f"grid never runs; {REGENERATE}"
        )


def test_the_smoke_sweep_moves_both_axes() -> None:
    """Two points on the shared axis and two workload sizes, for every DP method.

    One point per axis makes the gate unable to see the property the grid rests on: that
    the budget knob moves the method along the shared axis, and that the per-query budget
    and the workload-derived deltas move with Q.
    """
    for path in sorted(SMOKE_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        targets = cfg.get("epsilon_targets") or (
            cfg.get("synthetic_generation") or {}
        ).get("epsilon_targets")
        if targets is None:
            # The reference conditions price no mechanism and have no axis.
            assert path.stem in NON_PRIVATE, f"{path.name} declares no sweep axis"
            continue
        assert len(targets) >= 2, f"{path.name}: one point on the axis ({targets})"
        workloads = cfg.get("workload_queries")
        assert workloads and len(workloads) >= 2, (
            f"{path.name}: one workload size ({workloads}); the derived per-query budget "
            f"and per-release delta would never be seen to move"
        )
