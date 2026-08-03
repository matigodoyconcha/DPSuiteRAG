"""The consolidated CSV has to carry the shared axis, and keep the two MuRAGs apart."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.score_results import _csv_row


def _payload(**over):
    base = {
        "method": "private_rag",
        "dataset": "chatdoctor",
        "llm": {"model_name": "facebook/opt-1.3b"},
        "epsilon_total": 10.0,
        "epsilon_workload": 9.99,
        "epsilon_workload_basic": 9.99,
        "workload_queries": 5,
        "metrics": {},
    }
    base.update(over)
    return base


def _row(**over):
    return _csv_row(Path("x/run.json"), _payload(**over), Path("x"))


def test_the_shared_axis_reaches_the_csv():
    row = _row()
    assert row["epsilon_workload"] == 9.99
    assert row["epsilon_workload_basic"] == 9.99
    assert row["workload_queries"] == 5


def test_the_native_epsilon_is_kept_alongside_it():
    # Both are needed: one reproduces the source paper, the other compares methods.
    assert _row()["epsilon"] == 10.0


def test_ada_is_not_merged_into_murag():
    assert _row(variant="murag")["method"] == "private_rag"
    assert _row(variant="murag_ada")["method"] == "private_rag_ada"


def test_a_method_off_the_axis_says_why():
    row = _row(epsilon_workload=None, workload_error="delta exhausted")
    assert row["workload_error"] == "delta exhausted"


def test_a_missing_axis_is_blank_not_absent():
    # DictWriter raises on an unexpected key and writes "" for a missing one; the
    # column must exist for every method, including the non-private baseline.
    row = _csv_row(Path("x/v.json"), {"method": "vanilla", "metrics": {}}, Path("x"))
    assert row["epsilon_workload"] == ""


def test_the_writer_accepts_every_key_the_row_builds(tmp_path):
    """The row and the header must come from one list, or a run dies at write time."""
    from experiments.score_results import CSV_FIELDS

    out = tmp_path / "m.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerow(_row(variant="murag_ada"))
    read = list(csv.DictReader(open(out, encoding="utf-8")))
    assert read[0]["method"] == "private_rag_ada"
    assert read[0]["epsilon_workload"] == "9.99"


def test_the_workload_curve_reaches_the_csv():
    row = _row(epsilon_workload_by_q={"1": 9.98, "5": 49.73, "100": 859.44})
    assert row["epsilon_workload_q1"] == 9.98
    assert row["epsilon_workload_q100"] == 859.44


def test_a_run_without_the_curve_leaves_it_blank():
    # Back-filling the run's own epsilon into every Q would draw a method whose
    # guarantee does not degrade with the workload — the opposite of the finding.
    row = _row()
    assert row["epsilon_workload_q1"] == ""
    assert row["epsilon_workload_q100"] == ""
