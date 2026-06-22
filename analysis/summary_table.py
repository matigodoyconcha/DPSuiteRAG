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
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from analysis.load_results import CanonicalRecord, load_all


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def build_dataframe(records: list[CanonicalRecord]) -> pd.DataFrame:
    """Convert a list of CanonicalRecords into a tidy comparison DataFrame.

    Columns: Method, Dataset, Variant, Model, ε, ε_kind, ε_sum, ε_rdp,
             ContainsAccuracy, MatchAccuracy, BERTScore_F1, ROUGE-L, F1, MIA-AUC,
             N_Queries, Runtime_s.

    ``ContainsAccuracy`` is the headline accuracy: every method's paper defines
    "match accuracy" as containment (prediction *includes* a gold answer), not
    exact string equality. ``MatchAccuracy`` (exact match) is kept as a secondary
    column for reference only.
    """
    rows = []
    for r in records:
        # Format the primary epsilon cell
        if r.epsilon_kind == "na":
            eps_display = "N/A"
        elif r.epsilon_kind == "none":
            eps_display = "∞ (no DP)"
        elif r.epsilon_value is None:
            eps_display = "N/A"
        else:
            eps_display = f"{r.epsilon_value:.3f}"

        row = {
            "Method": r.method,
            "Dataset": r.dataset,
            "Variant": r.variant,
            "Model": r.model or "",
            "ε": eps_display,
            "ε_kind": r.epsilon_kind,
            "ε_sum": r.epsilon_sum,
            "ε_rdp": r.epsilon_composed,
            "ContainsAccuracy": r.utility.get("contains_accuracy"),
            "MatchAccuracy": r.utility.get("match_accuracy"),
            "BERTScore_F1": r.utility.get("bert_score_f1"),
            "ROUGE-L": r.utility.get("rouge_l"),
            "F1": r.utility.get("f1_score"),
            "MIA-AUC": r.privacy.get("mia_auc_bleu"),
            "N_Queries": r.n_queries,
            "Runtime_s": r.runtime,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Numeric columns: cast object columns that slipped through as strings
    for col in ["ε_sum", "ε_rdp", "ContainsAccuracy", "MatchAccuracy", "BERTScore_F1",
                "ROUGE-L", "F1", "MIA-AUC", "Runtime_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Seed aggregation (mean ± std over repeated runs)
# ---------------------------------------------------------------------------

# Identity of one experiment point; runs differing only by random seed collapse
# into one aggregated row.
_AGG_ID_COLS = ["Method", "Dataset", "Variant", "Model", "ε", "ε_kind"]
_AGG_METRIC_COLS = [
    "ContainsAccuracy", "MatchAccuracy", "BERTScore_F1", "ROUGE-L", "F1",
    "MIA-AUC", "ε_sum", "ε_rdp", "Runtime_s",
]


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse runs that differ only by random seed into mean ± std rows.

    Groups by (Method, Dataset, Variant, Model, ε) and reports, for each metric,
    the mean and the sample std across the group, plus ``N_seeds`` (the number of
    runs aggregated). With a single run the std is NaN. This lets you run a config
    over several seeds (distinct result dirs) and report averaged metrics.
    """
    if df.empty:
        return df

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
