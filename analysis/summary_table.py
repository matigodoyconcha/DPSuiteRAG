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
             MatchAccuracy, BERTScore_F1, ROUGE-L, F1, MIA-AUC, EntityLeakage,
             N_Queries, Runtime_s.
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
            "MatchAccuracy": r.utility.get("match_accuracy"),
            "BERTScore_F1": r.utility.get("bert_score_f1"),
            "ROUGE-L": r.utility.get("rouge_l"),
            "F1": r.utility.get("f1_score"),
            "MIA-AUC": r.privacy.get("mia_auc_bleu"),
            "EntityLeakage": r.privacy.get("private_entity_leakage"),
            "N_Queries": r.n_queries,
            "Runtime_s": r.runtime,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Numeric columns: cast object columns that slipped through as strings
    for col in ["ε_sum", "ε_rdp", "MatchAccuracy", "BERTScore_F1",
                "ROUGE-L", "F1", "MIA-AUC", "EntityLeakage", "Runtime_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


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
    write_outputs(df, Path(args.output_dir))


if __name__ == "__main__":
    main()
