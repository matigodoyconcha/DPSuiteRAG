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
    # Both are needed: one reproduces the source paper, the other compares methods. The
    # column named `epsilon` is the one the comparison is drawn on — every figure and
    # table reads it — so it is the shared axis, and the method's own number is kept
    # under a name that says whose units it is in.
    row = _row()
    assert row["epsilon"] == 9.99
    assert row["epsilon_method_own"] == 10.0


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


def test_the_realised_spend_reaches_the_csv():
    row = _row(epsilon_workload_spent=2.5, budget_utilisation=0.25)
    assert row["epsilon_workload_spent"] == 2.5
    assert row["budget_utilisation"] == 0.25


def test_a_run_without_a_spend_measurement_leaves_it_blank():
    # Blank means "not measured"; 0.0 would claim the mechanism released nothing.
    row = _row()
    assert row["epsilon_workload_spent"] == ""
    assert row["budget_utilisation"] == ""


def test_the_answer_length_reaches_the_csv():
    """Every text metric is monotone in it, so the comparison is only readable with it.

    Dropped from the CSV, "these two methods were scored over answers of the same
    length" is a claim a reader has to take from a pair of YAML files instead of from
    the row in front of them.
    """
    assert _row(answer_tokens=30)["answer_tokens"] == 30


def test_a_method_that_answers_at_no_configured_length_is_blank():
    # The accounting block writes the key as None when the runner passes no length;
    # writing 0 would read as a method configured to emit nothing.
    assert _row(answer_tokens=None)["answer_tokens"] == ""
    assert _row()["answer_tokens"] == ""


def test_the_private_answer_columns_reach_the_csv():
    """`answer_rate` alone cannot say whether the privacy mechanism served a cell.

    DPSparseVoteRAG's Sparse Vector Technique routes tokens down a budget-free public
    path, so a cell whose private mechanism provably never fires still reports a
    non-zero answer rate and non-zero accuracy, scored over LM output no mechanism
    touched.
    """
    row = _row(
        method="dpsparsevoterag",
        private_answer_rate=0.0,
        public_only=True,
        tokens_private=0,
        metrics={"answer_rate": 0.35, "contains_accuracy": 0.2},
    )
    assert row["private_answer_rate"] == 0.0
    assert row["public_only"] is True
    assert row["tokens_private"] == 0
    # The column that is not to be read as the mechanism's, kept beside it.
    assert row["answer_rate"] == 0.35


def test_a_served_cell_is_told_apart_from_an_unserved_one():
    row = _row(
        method="dpsparsevoterag", private_answer_rate=0.8, public_only=False, tokens_private=91
    )
    assert row["private_answer_rate"] == 0.8
    assert row["public_only"] is False
    assert row["tokens_private"] == 91


def test_a_method_with_no_public_path_leaves_the_private_columns_blank():
    """Blank is "the question does not arise", which is not "the mechanism did nothing".

    Defaulting these to 0 would mark every method that has no public path to report as
    a cell whose privacy mechanism never fired.
    """
    row = _row()
    assert row["private_answer_rate"] == ""
    assert row["public_only"] == ""
    assert row["tokens_private"] == ""


def test_dp_rags_realised_split_reaches_the_csv():
    # eps_r / (eps_r + eps_g) as the cell ran it: the configured ratio is a constant,
    # this is not, so it has to travel with the row.
    row = _row(method="dp_rag", epsilon_retrieval_share=0.0872)
    assert row["epsilon_retrieval_share"] == 0.0872
    assert _row()["epsilon_retrieval_share"] == ""


def test_every_new_column_survives_a_write_and_read(tmp_path):
    from experiments.score_results import CSV_FIELDS

    out = tmp_path / "m.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerow(_row(
            method="dpsparsevoterag", answer_tokens=30, private_answer_rate=0.0,
            public_only=True, tokens_private=0,
        ))
        w.writerow(_row())
    read = list(csv.DictReader(open(out, encoding="utf-8")))
    assert read[0]["answer_tokens"] == "30"
    assert read[0]["private_answer_rate"] == "0.0"
    assert read[0]["public_only"] == "True"
    assert read[0]["tokens_private"] == "0"
    # The method that does not emit them: empty cells, never zeros.
    assert read[1]["private_answer_rate"] == ""
    assert read[1]["public_only"] == ""
    assert read[1]["tokens_private"] == ""
