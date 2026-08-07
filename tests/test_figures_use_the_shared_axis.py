"""The figures and the CSV must be drawn on the axis the text claims.

Eight methods are compared at equal `epsilon_workload`: the privacy loss over a
Q-question workload, composed by one accountant at one delta. Every method ALSO carries
its own epsilon, in its own units at its own delta — a per-query PLD composition for
DP-RAG, a closed-form zCDP conversion for DP-SynRAG, an allocated budget for the vote
methods — and those three numbers have nothing to do with one another. Reading the wrong
one moves a point along the very axis the comparison is about: DP-RAG's (Q=100,
target=5) cell reports `epsilon_total = 0.330` and belongs at 5.

These tests pin the consumption: what the plot scripts read, what the CSV's `epsilon`
column means, and that a method flat in Q is present at every Q rather than at one.
"""
from __future__ import annotations

import ast
import csv
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.load_results import CanonicalRecord  # noqa: E402
from experiments.score_results import CSV_FIELDS, _csv_row  # noqa: E402

#: The shipped grid: the privacy levels the sweep asks for, and the workload sizes each
#: is priced over.
TARGETS = (1.0, 3.0, 5.0, 10.0, 50.0)
WORKLOADS = (1, 5, 25, 100)


# --------------------------------------------------------------------------- #
# Synthetic payloads — no model, no GPU, no results tree required
# --------------------------------------------------------------------------- #
def _payload(method: str, target: float, q: int, *, own_epsilon: float, **extra) -> dict:
    """One result file as the runners write it, with both epsilons deliberately apart.

    `own_epsilon` is nowhere near the target on purpose: a consumer that reads it
    instead of the axis cannot pass by coincidence.
    """
    return {
        "method": method,
        "dataset": "triviaqa",
        "llm": {"model_name": "facebook/opt-1.3b"},
        "epsilon_total": own_epsilon,
        "epsilon_target": target,
        "epsilon_workload": target,
        "workload_queries": q,
        "run_seed": 1,
        "metrics": {"contains_accuracy": 0.5, "s2mia_auc": 0.5},
        "predictions": ["a"],
        "references": ["a"],
        **extra,
    }


def _write_tree(root: Path, payloads: list[tuple[str, str, dict]]) -> Path:
    """Lay payloads out as <run>/<method>/<dataset>/<model>/<file>.json."""
    for method, name, payload in payloads:
        out = root / method / payload["dataset"] / "facebook__opt-1.3b" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload), encoding="utf-8")
    return root


# --------------------------------------------------------------------------- #
# The CSV `epsilon` column
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("q", WORKLOADS)
def test_the_csv_epsilon_column_is_the_target_the_cell_was_derived_for(target, q):
    """`epsilon` is the shared axis, so it reads the target every cell was calibrated to.

    The knob was derived by inverting this accountant, so the achieved workload epsilon
    IS the grid coordinate; a column that disagreed would mean the run did not implement
    the point it is filed under.
    """
    row = _csv_row(
        Path("x/c.json"),
        _payload("dp_rag", target, q, own_epsilon=0.330),
        Path("x"),
    )
    assert row["epsilon"] == target
    assert row["epsilon_target"] == target


def test_dp_synrags_inflated_own_epsilon_stays_out_of_the_epsilon_column():
    """The classic zCDP conversion is up to 48% above the axis at the same rho.

    rho + sqrt(4 rho ln(1/delta)) reads 1.4825 where the Renyi conversion reads 1.0, and
    the inflation shrinks with the target (2.7% at 50), so putting it in the `epsilon`
    column stretched the curve non-uniformly — hardest exactly at the private end the
    thesis argues about.
    """
    payload = _payload(
        "dp_synrag", 1.0, 100,
        own_epsilon=1.4825,
        privacy_accounting={"zcdp": {"epsilon": 1.4825, "delta": 1e-2, "rho": 0.0334}},
    )
    row = _csv_row(Path("x/q100.json"), payload, Path("x"))
    assert row["epsilon"] == 1.0
    # Carried, not discarded: both numbers are true of the run and the names say which.
    assert row["epsilon_method_own"] == 1.4825
    assert row["epsilon_classic_zcdp"] == 1.4825


def test_the_secondary_epsilon_columns_have_headers():
    for column in ("epsilon_target", "epsilon_method_own", "epsilon_classic_zcdp"):
        assert column in CSV_FIELDS


def test_a_method_without_a_zcdp_conversion_leaves_that_column_blank():
    # Blank reads "the question does not arise for this method"; repeating the method's
    # own epsilon there would invent a zCDP accounting seven of the eight never did.
    row = _csv_row(Path("x/c.json"), _payload("dpksa", 5.0, 100, own_epsilon=2.0), Path("x"))
    assert row["epsilon_classic_zcdp"] == ""
    assert row["epsilon_method_own"] == 2.0


def test_a_cell_the_accountant_declined_still_names_its_place_on_the_axis():
    """No workload epsilon, but the target is the coordinate the cell was run at.

    Dropping to the method's own epsilon here would move the point to a different x
    without saying so; the target keeps it where the grid put it.
    """
    payload = _payload("dpvoterag", 3.0, 100, own_epsilon=97.0)
    payload["epsilon_workload"] = None
    payload["workload_error"] = "ValueError: additive delta overruns the target"
    row = _csv_row(Path("x/c.json"), payload, Path("x"))
    assert row["epsilon"] == 3.0


def test_a_run_written_before_the_axis_existed_falls_back_to_its_own_epsilon():
    payload = {"method": "dpksa", "dataset": "nq", "epsilon_total": 8.0, "metrics": {}}
    row = _csv_row(Path("x/c.json"), payload, Path("x"))
    assert row["epsilon"] == 8.0
    assert row["epsilon_method_own"] == 8.0


def test_the_writer_still_accepts_the_epsilon_row(tmp_path):
    out = tmp_path / "m.csv"
    row = _csv_row(Path("x/c.json"), _payload("dp_rag", 5.0, 100, own_epsilon=0.33), Path("x"))
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    assert list(csv.DictReader(out.open(encoding="utf-8")))[0]["epsilon"] == "5.0"


# --------------------------------------------------------------------------- #
# analysis/plot_run.py
# --------------------------------------------------------------------------- #
def test_plot_run_reads_the_workload_field_not_the_methods_own(tmp_path):
    from analysis.plot_run import load_run

    payloads = [
        ("dp_rag", f"eps_target{t:g}_q{q}.json", _payload("dp_rag", t, q, own_epsilon=0.33))
        for t in TARGETS for q in WORKLOADS
    ]
    df = load_run(_write_tree(tmp_path, payloads))
    assert sorted(set(df["epsilon"])) == sorted(TARGETS)
    assert 0.33 not in set(df["epsilon"])
    assert sorted(set(df["workload_queries"])) == [float(q) for q in WORKLOADS]


def test_plot_run_falls_back_to_the_target_when_the_accountant_declined():
    from analysis.plot_run import shared_epsilon

    assert shared_epsilon({"epsilon_workload": 5.0, "epsilon_target": 5.0, "epsilon_total": 0.33}) == 5.0
    assert shared_epsilon({"epsilon_target": 5.0, "epsilon_total": 0.33}) == 5.0


def test_plot_run_never_reads_a_methods_own_epsilon_as_the_axis():
    import math

    from analysis.plot_run import shared_epsilon

    assert math.isnan(shared_epsilon({"epsilon_total": 0.33, "epsilon": 0.33}))


def test_plot_run_separates_the_workload_sizes_into_their_own_series():
    """A line may only join points that differ in epsilon alone.

    One series per model joined a method's Q=1 sweep to its Q=100 sweep — two budgets
    calibrated separately — into a single polyline.
    """
    import pandas as pd

    from analysis.plot_run import series_labels

    df = pd.DataFrame({
        "model_short": ["opt-1.3b"] * 4,
        "workload_queries": [1.0, 1.0, 100.0, 100.0],
    })
    assert set(series_labels(df)) == {"opt-1.3b Q=1", "opt-1.3b Q=100"}
    # A run at one workload size needs no such disambiguation.
    one_q = pd.DataFrame({"model_short": ["opt-1.3b"] * 2, "workload_queries": [100.0, 100.0]})
    assert set(series_labels(one_q)) == {"opt-1.3b"}


# --------------------------------------------------------------------------- #
# analysis/plot_tradeoff.py
# --------------------------------------------------------------------------- #
def _record(method: str, target: float, q: int, *, own: float) -> CanonicalRecord:
    return CanonicalRecord(
        method=method, dataset="triviaqa", variant="full", source_file="x.json",
        model="facebook/opt-1.3b",
        epsilon_value=own, epsilon_kind="input_total",
        epsilon_composed=own, epsilon_sum=own,
        epsilon_allocated=own, epsilon_spent=own,
        delta=1e-3, utility={"contains_accuracy": 0.4}, privacy={"s2mia_auc": 0.5},
        n_queries=40, runtime=None,
        epsilon_workload=target, epsilon_target=target, workload_queries=q,
        run_seed=1,
    )


def test_plot_tradeoff_places_every_cell_of_the_grid_at_its_target():
    """The x of every method and cell is its grid coordinate, not its own epsilon.

    The own-epsilon values here span two orders of magnitude across the methods, which
    is the situation the shared axis exists to remove from the figure.
    """
    from analysis.plot_tradeoff import _collect_eps_series

    owns = {"dp_rag": 0.330, "dpvoterag": 97.0, "dpsparsevoterag": 44.5,
            "dp_synrag": 1.4825, "dpksa": 2.0, "private_rag": 12.5}
    records = [
        _record(method, t, q, own=own)
        for method, own in owns.items() for t in TARGETS for q in WORKLOADS
    ]
    by_q = _collect_eps_series(records, "contains_accuracy", "utility")

    assert sorted(by_q) == list(WORKLOADS), "the panels are not one per workload size"
    for q, series in by_q.items():
        assert len(series) == len(owns), f"Q={q} lost a method"
        for label, points in series.items():
            xs = sorted(round(x, 6) for x, _, _ in points)
            assert xs == sorted(TARGETS), f"{label} at Q={q} is drawn at {xs}"


def test_plot_tradeoff_keeps_the_workload_sizes_apart():
    """Q is part of a point's identity: the same target at two Q is two experiments."""
    from analysis.plot_tradeoff import _collect_eps_series

    records = [_record("dp_rag", 5.0, q, own=0.33) for q in WORKLOADS]
    by_q = _collect_eps_series(records, "contains_accuracy", "utility")
    assert sorted(by_q) == list(WORKLOADS)


def test_plot_tradeoff_ignores_a_records_own_epsilon_entirely():
    from analysis.plot_tradeoff import _shared_epsilon

    record = _record("dp_rag", 5.0, 100, own=0.330)
    assert _shared_epsilon(record) == 5.0
    import dataclasses

    stranded = dataclasses.replace(record, epsilon_workload=None, epsilon_target=None)
    assert _shared_epsilon(stranded) is None


def test_the_faceted_tradeoff_figure_renders(tmp_path):
    """The figure is actually drawn, on synthetic records — no model, no results tree."""
    from analysis.plot_tradeoff import _plot_eps_vs_metric, _plot_pareto

    records = [
        _record(method, t, q, own=0.33)
        for method in ("dp_rag", "dp_synrag", "private_rag")
        for t in TARGETS for q in WORKLOADS
    ]
    _plot_eps_vs_metric(records, "triviaqa", "contains_accuracy", "utility", tmp_path, 60)
    _plot_pareto(records, "triviaqa", "contains_accuracy", "s2mia_auc", tmp_path, 60)
    assert (tmp_path / "tradeoff_eps_vs_contains_accuracy_triviaqa.png").exists()
    assert (tmp_path / "pareto_privacy_utility_triviaqa.png").exists()


def test_the_run_diagnostics_render(tmp_path):
    from analysis.plot_run import load_run, plot_metric_heatmap, plot_tradeoff

    payloads = [
        ("dp_rag", f"eps_target{t:g}_q{q}.json", _payload("dp_rag", t, q, own_epsilon=0.33))
        for t in TARGETS for q in WORKLOADS
    ]
    df = load_run(_write_tree(tmp_path / "run", payloads))
    plot_tradeoff(df, "dp_rag", tmp_path / "tradeoff.png",
                  metrics=["contains_accuracy"], logx=True)
    plot_metric_heatmap(df, tmp_path / "heatmap.png")
    assert (tmp_path / "tradeoff.png").exists()
    assert (tmp_path / "heatmap.png").exists()


def test_the_tradeoff_axis_is_labelled_as_the_shared_one():
    """A figure drawn on `epsilon_workload` and labelled "ε" invites the wrong reading."""
    from analysis.plot_run import EPSILON_AXIS_LABEL as run_label
    from analysis.plot_tradeoff import EPSILON_AXIS_LABEL as tradeoff_label

    for label in (run_label, tradeoff_label):
        assert "workload" in label
        assert "epsilon_total" not in label


@pytest.mark.parametrize("script", ["plot_run.py", "plot_tradeoff.py"])
def test_no_plot_script_reads_a_methods_own_epsilon(script):
    """Statically: the own-epsilon fields are not read anywhere a figure is built.

    They are mentioned in the comments that explain why — hence the AST rather than a
    substring search.
    """
    tree = ast.parse((PROJECT_ROOT / "analysis" / script).read_text(encoding="utf-8"))
    own = {"epsilon_total", "epsilon_composed", "epsilon_sum",
           "epsilon_allocated", "epsilon_spent"}
    read: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in own:
            read.add(node.attr)
        if isinstance(node, ast.Constant) and node.value in own:
            read.add(node.value)
    assert not read, f"{script} still reads {sorted(read)} where the shared axis belongs"


# --------------------------------------------------------------------------- #
# MURAG vs MURAG-ADA
# --------------------------------------------------------------------------- #
def _murag_tree(root: Path) -> Path:
    """Both variants as one runner writes them: same method name, different variant."""
    payloads = []
    for variant in ("murag", "murag_ada"):
        payload = _payload("private_rag", 5.0, 100, own_epsilon=5.0)
        payload["variant"] = variant
        payload["n_queries"] = 1
        payloads.append(("private_rag", f"{variant}_q100.json", payload))
    return _write_tree(root, payloads)


def test_the_loader_tells_murag_and_murag_ada_apart(tmp_path):
    """They differ in the retrieval mechanism — a noisy prefix-sum threshold against a
    fixed public one — which is the comparison the pair exists to make. Read as one
    method they share a series, a colour and an aggregation group, and it disappears."""
    from analysis.load_results import load_all

    records = load_all(_murag_tree(tmp_path), recompute_vanilla=False)
    assert {r.method for r in records} == {"private_rag", "private_rag_ada"}


def test_filtering_on_the_split_method_still_finds_its_files(tmp_path):
    """ADA's files live under its parent's directory, so a filter applied to the path
    alone rejected them before the payload could name them."""
    from analysis.load_results import load_all

    records = load_all(_murag_tree(tmp_path), methods=["private_rag_ada"],
                       recompute_vanilla=False)
    assert [r.method for r in records] == ["private_rag_ada"]
    records = load_all(_murag_tree(tmp_path), methods=["private_rag"],
                       recompute_vanilla=False)
    assert [r.method for r in records] == ["private_rag"]


def test_the_run_diagnostics_tell_them_apart_too(tmp_path):
    from analysis.plot_run import load_run

    df = load_run(_murag_tree(tmp_path / "run"))
    assert set(df["method"]) == {"private_rag", "private_rag_ada"}


def test_the_loader_and_the_scorer_split_the_variants_the_same_way(tmp_path):
    """One rule. A table and a figure of the same run must agree on how many methods ran."""
    from analysis.load_results import method_of_payload
    from experiments.score_results import _csv_row

    for variant in ("murag", "murag_ada"):
        payload = _payload("private_rag", 5.0, 100, own_epsilon=5.0)
        payload["variant"] = variant
        assert method_of_payload("private_rag", payload) == _csv_row(
            Path("x/c.json"), payload, Path("x"))["method"]


# --------------------------------------------------------------------------- #
# analysis/summary_table.py
# --------------------------------------------------------------------------- #
def test_the_summary_tables_headline_epsilon_is_the_shared_axis():
    from analysis.summary_table import build_dataframe

    df = build_dataframe([_record("dp_rag", 5.0, 100, own=0.330)])
    assert df["ε"].iloc[0] == "5.000"
    assert df["ε_method_own"].iloc[0] == 0.330


def test_the_summary_table_groups_seeds_on_the_shared_axis():
    """Two seeds of one cell whose own epsilon happens to differ are still one cell.

    Grouping on a method's own units splits cells that sit at one point of the axis and
    merges cells that do not — the one thing an aggregation key must never do.
    """
    import dataclasses

    from analysis.summary_table import aggregate_seeds, build_dataframe

    seeds = [
        dataclasses.replace(_record("dp_rag", 5.0, 100, own=own), run_seed=seed)
        for seed, own in ((1, 0.330), (2, 0.331))
    ]
    out = aggregate_seeds(build_dataframe(seeds))
    assert len(out) == 1, "one cell was reported as two"
    assert out["N_seeds"].iloc[0] == 2
    assert out["ε"].iloc[0] == "5.000"


def test_the_summary_table_keeps_two_workload_sizes_apart():
    import dataclasses

    from analysis.summary_table import aggregate_seeds, build_dataframe

    rows = [
        dataclasses.replace(_record("dp_synrag", 5.0, q, own=6.0), run_seed=1)
        for q in (1, 100)
    ]
    assert len(aggregate_seeds(build_dataframe(rows))) == 2


def test_the_no_dp_baseline_still_reads_as_unbounded():
    import dataclasses

    from analysis.summary_table import build_dataframe

    record = dataclasses.replace(
        _record("vanilla", 0.0, 100, own=0.0),
        epsilon_kind="none", epsilon_workload=None, epsilon_target=None,
    )
    assert build_dataframe([record])["ε"].iloc[0] == "∞ (no DP)"


# --------------------------------------------------------------------------- #
# The flat-regime methods, at every Q
# --------------------------------------------------------------------------- #
def test_dp_synrag_files_its_result_at_every_workload_size():
    """Flat in Q means the same numbers at every Q, and the rows have to be there.

    DP-SynRAG spends its whole budget building the synthetic corpus; answering from it
    is post-processing, so nothing depends on how long the question stream is. Filing
    the one measurement under Q=1 alone did not report that — it removed the method from
    every panel drawn at another Q, so a per-Q comparison silently dropped it while
    MuRAG, filed at Q=100, dropped out of the Q=1 panel instead.
    """
    source = (PROJECT_ROOT / "experiments" / "run_dp_synrag.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    loops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "workload_q"
    ]
    assert loops, "dp_synrag no longer loops over the workload sizes"
    for loop in loops:
        assert isinstance(loop.iter, ast.Name) and loop.iter.id == "workload_sizes", (
            "dp_synrag writes only part of its workload sweep; a sliced or maxed "
            "iterable leaves the method absent from every other Q's panel"
        )
        saved = [
            node for node in ast.walk(loop)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "_save_results"
        ]
        assert saved, "the loop over the workload sizes writes no file"
    # Each record still names the sizes it is a copy of, so the four rows are not read
    # as four measurements.
    assert "workload_queries_covered" in source


def test_the_shared_payload_is_stored_once_and_pointed_at():
    """The four Q records of one cell are one inference run, so one copy of the text.

    They exist so a per-Q filter finds the method at every Q; storing the generated
    corpus and every question's result four times per (dataset, LLM, target) buys
    nothing for that.
    """
    source = (PROJECT_ROOT / "experiments" / "run_dp_synrag.py").read_text(encoding="utf-8")
    assert 'payload["shared_payload"] = canonical_name' in source
    assert 'for heavy in ("synthetic_texts", "results")' in source


def test_murag_also_stores_its_stream_once():
    """MuRAG files at every Q for the same reason DP-SynRAG does, at the same cost.

    Both are flat in Q: one stream answers the questions once and the workload sizes
    only price it differently. Whichever method gained the per-Q rows last is the one
    likely to have kept the duplication, so both are pinned here rather than only the
    one that was fixed first.
    """
    source = (PROJECT_ROOT / "experiments" / "run_private_rag.py").read_text(encoding="utf-8")
    assert 'payload["shared_payload"] = canonical_name' in source
    assert 'for heavy in ("predictions", "references", "results", "prefix_curve")' in source


def test_a_record_pointing_at_a_sibling_is_scored_and_gets_its_own_row(tmp_path):
    """Resolved, scored once, and the borrowed text is not written back."""
    from experiments.score_results import score_file

    canonical = _payload("dp_synrag", 5.0, 1, own_epsilon=6.0)
    canonical["results"] = [{"prediction": "paris", "reference": "paris"}]
    canonical["synthetic_texts"] = ["a synthetic document"]
    canonical.pop("predictions")
    canonical.pop("references")
    (tmp_path / "q1.json").write_text(json.dumps(canonical), encoding="utf-8")

    sibling = _payload("dp_synrag", 5.0, 100, own_epsilon=6.0)
    sibling.pop("predictions")
    sibling.pop("references")
    sibling["shared_payload"] = "q1.json"
    (tmp_path / "q100.json").write_text(json.dumps(sibling), encoding="utf-8")

    scored: dict = {}
    first = score_file(tmp_path / "q1.json", scored)
    second = score_file(tmp_path / "q100.json", scored)

    assert first is not None and second is not None
    # Same numbers, computed once.
    assert second["metrics"]["contains_accuracy"] == first["metrics"]["contains_accuracy"]
    # ...and the sibling on disk still holds no copy of the text.
    on_disk = json.loads((tmp_path / "q100.json").read_text(encoding="utf-8"))
    assert "results" not in on_disk and "synthetic_texts" not in on_disk
    assert on_disk["metrics"]["contains_accuracy"] == first["metrics"]["contains_accuracy"]
    # Its own coordinates are untouched by the borrowing.
    assert on_disk["workload_queries"] == 100
    assert on_disk["epsilon_workload"] == 5.0


def test_the_pointer_is_scored_whichever_order_the_files_come_in(tmp_path):
    from experiments.score_results import score_file

    canonical = _payload("dp_synrag", 5.0, 1, own_epsilon=6.0)
    canonical["results"] = [{"prediction": "paris", "reference": "paris"}]
    canonical.pop("predictions"), canonical.pop("references")
    (tmp_path / "q1.json").write_text(json.dumps(canonical), encoding="utf-8")
    sibling = _payload("dp_synrag", 5.0, 100, own_epsilon=6.0)
    sibling.pop("predictions"), sibling.pop("references")
    sibling["shared_payload"] = "q1.json"
    (tmp_path / "q100.json").write_text(json.dumps(sibling), encoding="utf-8")

    scored: dict = {}
    second = score_file(tmp_path / "q100.json", scored)
    first = score_file(tmp_path / "q1.json", scored)
    assert first["metrics"]["contains_accuracy"] == second["metrics"]["contains_accuracy"]


def test_a_shared_payload_pointer_cannot_leave_its_directory(tmp_path):
    """A result file is data. A path in it is not a licence to read elsewhere."""
    from experiments.score_results import _shared_payload_path

    (tmp_path / "q1.json").write_text("{}", encoding="utf-8")
    for hostile in ("../secret.json", "/etc/passwd", "sub/dir.json"):
        assert _shared_payload_path(tmp_path / "q100.json", {"shared_payload": hostile}) is None
    assert _shared_payload_path(tmp_path / "q100.json", {"shared_payload": "q1.json"}) is not None


def test_the_flat_methods_agree_on_what_a_workload_label_means():
    """Both flat-regime methods must exist at the same Q values.

    Filtering the CSV on Q=100 returned MuRAG and not DP-SynRAG, and on Q=1 the
    opposite, so no per-Q panel ever held both.
    """
    synrag = (PROJECT_ROOT / "experiments" / "run_dp_synrag.py").read_text(encoding="utf-8")
    murag = (PROJECT_ROOT / "experiments" / "run_private_rag.py").read_text(encoding="utf-8")
    assert "for workload_q in workload_sizes:" in synrag
    assert "for workload_q in workload_queries:" in murag
    assert "for workload_q in [max(" not in murag
