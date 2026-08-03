"""The workload size has to reach the table, or the experiment reports its own noise.

Two cells of a method whose budget does not move with the workload — MuRAG, DP-SynRAG —
differ in nothing the analysis can key on except Q. Left out of the aggregation identity
their utilities are averaged and the spread is reported as seed variance, which turns
the finding the sweep exists to measure into an error bar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.load_results import load_all
from analysis.summary_table import aggregate_seeds, build_dataframe


def _cell(root: Path, q: int, accuracy: float, target: float = 10.0) -> None:
    out = root / "private_rag" / "triviaqa" / "model"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"epsilon_q_5.0_q{q}.json").write_text(json.dumps({
        "method": "private_rag",
        "dataset": "triviaqa",
        "llm": {"model_name": "model"},
        "epsilon_q": 5.0,
        "epsilon_target": target,
        "workload_queries": q,
        "epsilon_workload": target,
        "run_seed": 1,
        "n_queries": 4,
        "metrics": {"contains_accuracy": accuracy},
        "predictions": ["a"] * 4,
        "references": ["a"] * 4,
        "results": [{"prediction": "a", "reference": "a", "query_id": f"q{i}"} for i in range(4)],
    }), encoding="utf-8")


@pytest.fixture
def two_workloads(tmp_path: Path) -> Path:
    _cell(tmp_path, q=1, accuracy=0.70)
    _cell(tmp_path, q=100, accuracy=0.10)
    return tmp_path


def test_the_loader_keeps_both_coordinates(two_workloads: Path):
    records = load_all(two_workloads, recompute_vanilla=False)
    assert len(records) == 2
    assert {r.workload_queries for r in records} == {1, 100}
    assert {r.epsilon_target for r in records} == {10.0}


def test_two_workload_sizes_do_not_collapse_into_one_row(two_workloads: Path):
    df = aggregate_seeds(build_dataframe(load_all(two_workloads, recompute_vanilla=False)))
    assert len(df) == 2, f"the workload axis collapsed:\n{df}"
    assert set(df["Q"]) == {1, 100}
    assert sorted(df["ContainsAccuracy"]) == pytest.approx([0.10, 0.70])
    # Averaging them would report 0.4 with a std of 0.42 and N_seeds=2 — a real effect
    # re-labelled as noise.
    assert set(df["N_seeds"]) == {1}


def test_the_shared_axis_survives_aggregation(two_workloads: Path):
    df = aggregate_seeds(build_dataframe(load_all(two_workloads, recompute_vanilla=False)))
    assert "ε_workload" in df.columns
    assert set(df["ε_workload"]) == {10.0}


def test_a_run_that_declared_no_workload_reports_none(tmp_path: Path):
    """`n_queries` is how many questions were answered, never the declared workload.

    Reading one as the other gives an old run a workload size it never had, off the
    grid, which then silently joins or splits groups once Q is a grouping key.
    """
    out = tmp_path / "dpvoterag" / "triviaqa" / "model"
    out.mkdir(parents=True)
    (out / "epsilon_8.0.json").write_text(json.dumps({
        "method": "dpvoterag", "dataset": "triviaqa", "llm": {"model_name": "model"},
        "epsilon_total": 8.0, "n_queries": 500,
        "metrics": {"contains_accuracy": 0.5},
        "predictions": ["a"], "references": ["a"],
        "results": [{"prediction": "a", "reference": "a"}],
    }), encoding="utf-8")
    record = load_all(tmp_path, recompute_vanilla=False)[0]
    assert record.workload_queries is None
