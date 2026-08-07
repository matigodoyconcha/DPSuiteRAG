"""Build and write the privacy–utility comparison table.

Usage (module)::

    python -m analysis.summary_table [options]

Options::

    --results-dir   PATH   Root of results/ tree          (default: results)
    --output-dir    PATH   Where to write CSV and TeX      (default: results)
    --datasets      NAME+  Filter by dataset name
    --methods       NAME+  Filter by method name
    --no-recompute-vanilla  Skip vanilla metric recomputation
    --bertscore            Enable BERTScore F1 for vanilla
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from analysis.load_results import CanonicalRecord, load_all


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def _shared_epsilon(record: CanonicalRecord) -> Optional[float]:
    """The record's position on the common privacy axis, or None if it has none.

    ``epsilon_workload`` is the privacy loss over a Q-question workload, composed for all
    eight methods by one accountant at one delta, and it is the only epsilon in this
    table that means the same thing in two different rows. ``epsilon_target`` — the grid
    coordinate the cell was calibrated to reach — stands in for the cells whose
    accountant declined and recorded a ``workload_error`` instead of a number.
    """
    for value in (record.epsilon_workload, record.epsilon_target):
        if value is not None and not math.isnan(value):
            return float(value)
    return None


def _shared_epsilon_display(record: CanonicalRecord) -> str:
    """The headline ε cell: the shared axis, formatted."""
    if record.epsilon_kind == "none":
        return "∞ (no DP)"
    eps = _shared_epsilon(record)
    return "N/A" if eps is None else f"{eps:.3f}"


def _own_epsilon(record: CanonicalRecord) -> Optional[float]:
    """The method's own epsilon, in its own units at its own delta.

    Not comparable with another method's: a per-query PLD composition at δ=1e-3 for
    DP-RAG, a closed-form zCDP conversion for DP-SynRAG, an allocated per-query budget
    for the vote methods. Kept because it is what each source paper reports, and named
    so that it cannot be read as the axis the rows are compared on.
    """
    if record.epsilon_kind in {"na", "none"} or record.epsilon_value is None:
        return None
    return float(record.epsilon_value)


def build_dataframe(records: list[CanonicalRecord]) -> pd.DataFrame:
    """Convert a list of CanonicalRecords into a tidy comparison DataFrame.

    Columns: Method, Dataset, Variant, Model, ε, ε_method_own, ε_kind, ε_sum, ε_rdp,
             ContainsAccuracy, HitRate, MatchAccuracy, BERTScore_F1, ROUGE-L, F1,
             MIA-AUC, AnswerTokens, PrivateAnswerRate, PublicOnly, TokensPrivate,
             Seed, ε_workload, ε_retrieval_share, ε_target, Q, Answered, Runtime_s.

    ``ε`` is the SHARED axis — ``epsilon_workload``, falling back to ``epsilon_target``
    — because that is the number the rows are read against each other on. Each method's
    own epsilon is in ``ε_method_own``, in its own units at its own delta; the two differ
    by more than an order of magnitude for some cells (DP-RAG's (Q=100, target=5) cell
    reports 0.330 for a point that sits at 5 on the axis), so they cannot share a name.

    ``ContainsAccuracy`` is the headline accuracy: every method's paper defines
    "match accuracy" as containment (prediction *includes* a gold answer), not
    exact string equality. ``MatchAccuracy`` (exact match) is kept as a secondary
    column for reference only. ``HitRate`` is the headline for set-valued
    recommendation tasks (Movielens): the prediction includes any of the user's
    held-out favorites; it is empty for the containment datasets.
    """
    rows = []
    for r in records:
        row = {
            "Method": r.method,
            "Dataset": r.dataset,
            "Variant": r.variant,
            "Model": r.model or "",
            "ε": _shared_epsilon_display(r),
            "ε_method_own": _own_epsilon(r),
            "ε_kind": r.epsilon_kind,
            "ε_sum": r.epsilon_sum,
            "ε_rdp": r.epsilon_composed,
            "ContainsAccuracy": r.utility.get("contains_accuracy"),
            "HitRate": r.utility.get("hit_rate"),
            "MatchAccuracy": r.utility.get("match_accuracy"),
            "BERTScore_F1": r.utility.get("bert_score_f1"),
            "ROUGE-L": r.utility.get("rouge_l"),
            "F1": r.utility.get("f1_score"),
            "MIA-AUC": r.privacy.get("s2mia_auc"),
            # The answer length the row's utility was measured over. Every text metric
            # is monotone in it, so it belongs beside the metrics rather than in a config
            # file: two rows are only comparable on accuracy when this matches.
            "AnswerTokens": r.answer_tokens,
            # How often the privacy mechanism actually produced the answer being scored.
            # A method with a budget-free public path (DPSparseVoteRAG's SVT) reports
            # accuracy over text no mechanism touched whenever `PublicOnly` is true, and
            # `ContainsAccuracy` alone cannot show that. Empty for the methods with no
            # public path, where the distinction does not arise.
            "PrivateAnswerRate": r.private_answer_rate,
            # 1 when the mechanism released nothing at all for this row. Aggregated over
            # seeds it reads as the fraction of repetitions that ran entirely public.
            "PublicOnly": None if r.public_only is None else float(r.public_only),
            "TokensPrivate": r.tokens_private,
            "Seed": r.run_seed,
            "ε_workload": r.epsilon_workload,
            # DP-RAG: retrieval's realised share of the total budget. Constant ratio in
            # the config, but it moves across the sweep — see load_results.
            "ε_retrieval_share": r.epsilon_retrieval_share,
            # The cell's two grid coordinates. `ε_target` is the privacy level asked
            # for and `Q` the workload size it was priced over; together they name the
            # experiment point. `Answered` is a different quantity — how many questions
            # the utility was averaged over — and is deliberately not called Answered,
            # which read as if it were Q.
            "ε_target": r.epsilon_target,
            "Q": r.workload_queries,
            "Answered": r.n_queries,
            "Runtime_s": r.runtime,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Numeric columns: cast object columns that slipped through as strings
    for col in ["ε_method_own", "ε_sum", "ε_rdp", "ContainsAccuracy", "HitRate", "MatchAccuracy",
                "BERTScore_F1", "ROUGE-L", "F1", "MIA-AUC", "Runtime_s",
                "PrivateAnswerRate", "PublicOnly", "TokensPrivate",
                "ε_retrieval_share"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Seed aggregation (mean ± std over repeated runs)
# ---------------------------------------------------------------------------

# Identity of one experiment point; runs differing only by random seed collapse into
# one aggregated row. `ε_target` and `Q` are part of that identity: a cell priced for a
# one-question workload and one priced for a hundred are different conditions, and for
# a method whose budget does not move with the workload they are otherwise
# indistinguishable — their utilities would be averaged together and the spread
# reported as seed noise, erasing the axis the sweep exists to measure.
#: `AnswerTokens` is part of the identity and not a metric: every text metric is
#: monotone in the answer length, so two runs that produced answers of different lengths
#: are different conditions, and averaging them would report the length difference as
#: seed noise on the accuracy.
#: `ε` is the shared axis, so the grouping key is the coordinate the rows are compared
#: at. Grouping on a method's own epsilon instead split cells that sit at one point of
#: the axis and merged cells that do not, which is the one thing an aggregation key must
#: never do; `ε_method_own` is aggregated as a value below, not grouped on.
_AGG_ID_COLS = [
    "Method", "Dataset", "Variant", "Model", "ε", "ε_kind", "ε_target", "Q",
    "AnswerTokens",
]
_AGG_METRIC_COLS = [
    "ContainsAccuracy", "HitRate", "MatchAccuracy", "BERTScore_F1", "ROUGE-L", "F1",
    "MIA-AUC", "ε_method_own", "ε_sum", "ε_rdp", "Runtime_s",
    # Carried through so the shared axis survives aggregation; it is identical across
    # seeds of one cell, so a mean over it is the value itself.
    "ε_workload",
    # Whether the privacy mechanism served the cell, aggregated the same way the utility
    # is. Without these the summary keeps the accuracy of a cell whose mechanism never
    # fired and drops the only columns that say so. `PublicOnly` is 0/1 per row, so its
    # mean is the fraction of the aggregated seeds that ran entirely on the public path.
    "PrivateAnswerRate", "PublicOnly", "TokensPrivate",
    # Constant across seeds of one cell, like ε_workload, so the mean is the value.
    "ε_retrieval_share",
]


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse runs that differ only by random seed into mean ± std rows.

    Groups by (Method, Dataset, Variant, Model, ε, ε_target, Q, AnswerTokens) — the
    shared-axis coordinate, never a method's own units — and reports, for each metric,
    the mean and the sample std across the group, plus ``N_seeds`` (the number of
    runs aggregated). With a single run the std is NaN.

    Rows that carry no seed are left alone. Averaging them would invent error bars:
    two runs of the same cell a day apart, with different hyperparameters and no
    repetition relationship, were being reported as "N_seeds=2" with a standard
    deviation that measured the gap between two experiments rather than the spread of
    a mechanism. A repetition is a row whose seed says so.
    """
    if df.empty:
        return df
    if "Seed" not in df.columns or df["Seed"].isna().all():
        return df
    seeded = df[df["Seed"].notna()]
    if seeded.empty:
        return df
    df = seeded

    id_cols = [c for c in _AGG_ID_COLS if c in df.columns]
    metric_cols = [c for c in _AGG_METRIC_COLS if c in df.columns]
    if not id_cols or not metric_cols:
        return df

    grouped = df.groupby(id_cols, dropna=False)
    agg = grouped[metric_cols].agg(["mean", "std"])
    # Flatten the MultiIndex columns: mean -> "<metric>", std -> "<metric>_std".
    agg.columns = [m if stat == "mean" else f"{m}_std" for m, stat in agg.columns]
    counts = grouped.size().rename("N_seeds")
    out = agg.join(counts).reset_index()

    ordered = id_cols + ["N_seeds"]
    ordered += [c for c in out.columns if c not in ordered]
    return out[ordered]


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    """Write CSV, LaTeX, and print to stdout."""
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"[summary_table] CSV written to {csv_path}")

    tex_path = out_dir / "summary_table.tex"
    # Drop display-only columns for the LaTeX table
    latex_cols = [c for c in df.columns if c not in {"ε_kind", "Model", "Variant"}]
    df[latex_cols].to_latex(
        tex_path,
        index=False,
        float_format="%.3f",
        na_rep="--",
        escape=False,
    )
    print(f"[summary_table] LaTeX written to {tex_path}")

    print("\n" + df.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build DP-RAG privacy–utility summary table")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--no-recompute-vanilla", action="store_true")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help="Collapse runs differing only by random seed into mean ± std rows "
        "(adds N_seeds and <metric>_std columns).",
    )
    args = parser.parse_args(argv)

    records = load_all(
        results_dir=args.results_dir,
        datasets=args.datasets,
        methods=args.methods,
        recompute_vanilla=not args.no_recompute_vanilla,
        bertscore=args.bertscore,
    )

    if not records:
        print("[summary_table] No records loaded — check --results-dir and --datasets.")
        return

    df = build_dataframe(records)
    if args.aggregate_seeds:
        df = aggregate_seeds(df)
    write_outputs(df, Path(args.output_dir))


if __name__ == "__main__":
    main()
