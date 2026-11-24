"""analysis/plot_results.py draws only what a run holds, and reads streams by prefix."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analysis import plot_results

MODELS = ["org__gen-a", "org__gen-b"]


def _write(run: Path, method: str, dataset: str, model: str, name: str, **payload) -> None:
    path = run / method / dataset / model / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload.setdefault("method", method)
    payload.setdefault("dataset", dataset)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run(tmp_path: Path) -> Path:
    """Two corpora (one without headroom), DP-RAG over a grid, MuRAG as a 10-answer stream."""
    run = tmp_path / "run"
    for model in MODELS:
        for ds, floor, ceiling in (("easy", 0.2, 0.8), ("flat", 0.3, 0.3)):
            _write(run, "no_rag", ds, model, "no_rag.json",
                   metrics={"contains_accuracy": floor})
            _write(run, "vanilla", ds, model, "vanilla.json",
                   metrics={"contains_accuracy": ceiling})
            for eps in (1.0, 10.0):
                for q in (1, 5):
                    _write(run, "dp_rag", ds, model, f"eps_{eps}_q{q}.json",
                           epsilon_target=eps, workload_queries=q, epsilon_workload=eps,
                           metrics={"contains_accuracy": eps / 20 / q})
        # MuRAG answers one stream: the first answer right, the next nine wrong. Its
        # Q=5 file is the accounting label of the same stream and carries no answers.
        results = [{"prediction": "paris", "answers": ["paris"]}]
        results += [{"prediction": "no", "answers": ["rome"]}] * 9
        for eps in (1.0, 10.0):
            _write(run, "private_rag", "easy", model, f"eps_{eps}_q1.json",
                   variant="murag", epsilon_target=eps, workload_queries=1,
                   epsilon_workload=eps, results=results,
                   metrics={"contains_accuracy": 0.1})
            _write(run, "private_rag", "easy", model, f"eps_{eps}_q5.json",
                   variant="murag", epsilon_target=eps, workload_queries=5,
                   epsilon_workload=eps, shared_payload=f"eps_{eps}_q1.json",
                   metrics={"contains_accuracy": 0.1})
    return run


@pytest.fixture
def figures(tmp_path, capsys):
    run = _run(tmp_path)
    out = tmp_path / "figs"
    plot_results.main(["--run-dir", str(run), "--out", str(out),
                       "--workloads", "1", "5", "25"])
    return out, capsys.readouterr().out


def test_every_figure_and_its_csv_is_written(figures):
    out, _ = figures
    for stem in ("reference_range", "tradeoff_q1", "workload_axis"):
        assert (out / f"{stem}.pdf").exists() and (out / f"{stem}.png").exists()
    for csv in ("reference_range.csv", "tradeoff_q1.csv", "workload_axis.csv",
                "results_cells.csv"):
        assert (out / csv).exists()


def test_methods_the_run_does_not_hold_are_reported_and_not_drawn(figures):
    out, log = figures
    assert "absent:" in log and "dpksa" in log
    drawn = set(pd.read_csv(out / "tradeoff_q1.csv").method)
    assert drawn == {"dp_rag", "private_rag"}


def test_a_corpus_without_headroom_stays_out_of_the_method_figures(figures):
    out, log = figures
    assert "flat" in log
    assert set(pd.read_csv(out / "tradeoff_q1.csv").dataset) == {"easy"}
    # ...but it is still on the reference figure, which is where its absence is explained.
    assert "flat" in set(pd.read_csv(out / "reference_range.csv").dataset)


def test_a_stream_method_is_read_over_the_first_q_answers(figures):
    out, _ = figures
    t = pd.read_csv(out / "workload_axis.csv")
    murag = t[(t.method == "private_rag") & (t.epsilon_target == 10.0)]
    by_q = dict(zip(murag.workload_queries, murag.contains_accuracy))
    assert by_q[1.0] == pytest.approx(1.0)
    assert by_q[5.0] == pytest.approx(0.2)


def test_a_workload_longer_than_the_stream_is_left_out_not_padded(figures):
    out, log = figures
    t = pd.read_csv(out / "workload_axis.csv")
    assert 25.0 not in set(t[t.method == "private_rag"].workload_queries)
    assert "Q=25 left out" in log


def test_the_figures_can_be_redrawn_from_the_cell_table(figures, tmp_path):
    out, _ = figures
    again = tmp_path / "again"
    plot_results.main(["--from-csv", str(out / "results_cells.csv"), "--out", str(again)])
    pd.testing.assert_frame_equal(pd.read_csv(out / "tradeoff_q1.csv"),
                                  pd.read_csv(again / "tradeoff_q1.csv"))
