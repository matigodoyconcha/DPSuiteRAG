"""Generate privacy–utility trade-off figures for BenchDP-RAG.

Produces three plot families per dataset:

(a) ε vs utility metric (match_accuracy, rouge_l)
(b) ε vs privacy metric (mia_auc_bleu)
(c) Privacy vs utility scatter with Pareto frontier  ← the key comparison

Output format: PNG + PDF saved to ``results/figures/``.

Usage (module)::

    python -m analysis.plot_tradeoff [options]

Options::

    --results-dir   PATH   Root of results/ tree          (default: results)
    --figures-dir   PATH   Output directory for figures    (default: results/figures)
    --datasets      NAME+  Filter datasets
    --methods       NAME+  Filter methods
    --no-recompute-vanilla  Skip vanilla metric recomputation
    --bertscore            Enable BERTScore F1 for vanilla
    --dpi           INT    Figure DPI                      (default: 150)
    --vanilla-file  PATH   Override vanilla result file
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.load_results import CanonicalRecord, load_all, pareto_front

# ---------------------------------------------------------------------------
# Visual style constants
# ---------------------------------------------------------------------------

_METHOD_COLORS = {
    "vanilla": "black",
    "dp_rag": "tab:blue",
    "dpvoterag": "tab:orange",
    "dpsparsevoterag": "tab:green",
    "dp_synrag": "tab:red",
    "private_rag": "tab:pink",
}

_METHOD_MARKERS = {
    "vanilla": "*",
    "dp_rag": "o",
    "dpvoterag": "s",
    "dpsparsevoterag": "D",
    "dp_synrag": "^",
    "private_rag": "X",
}


def _color(method: str) -> str:
    return _METHOD_COLORS.get(method, "tab:gray")


def _marker(method: str) -> str:
    return _METHOD_MARKERS.get(method, "o")


def _save(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (a) & (b)  ε vs metric
# ---------------------------------------------------------------------------

def _collect_eps_series(
    records: list[CanonicalRecord],
    metric_key: str,
    metric_src: str,              # "utility" | "privacy"
) -> dict[str, list[tuple[float, float, str]]]:
    """Return {method_series_label: [(eps, value, variant), ...]}.

    DP-RAG and DPVoteRAG emit two series each.
    """
    series: dict[str, list[tuple[float, float, str]]] = defaultdict(list)

    for r in records:
        val = r.utility.get(metric_key) if metric_src == "utility" else r.privacy.get(metric_key)
        if val is None or math.isnan(val):
            continue

        if r.method == "dp_rag":
            # Series 1 — composed (solid)
            if r.epsilon_composed is not None:
                series["dp_rag_rdp"].append((r.epsilon_composed, val, r.variant))
            # Series 2 — sum (dashed)
            if r.epsilon_sum is not None:
                series["dp_rag_sum"].append((r.epsilon_sum, val, r.variant))

        elif r.method in {"dpvoterag", "dpsparsevoterag"}:
            label_base = r.method
            # Series 1 — allocated (solid)
            if r.epsilon_allocated is not None:
                series[f"{label_base}_allocated"].append((r.epsilon_allocated, val, r.variant))
            # Series 2 — spent (dashed)
            if r.epsilon_spent is not None:
                series[f"{label_base}_spent"].append((r.epsilon_spent, val, r.variant))

        elif r.method == "vanilla":
            series["vanilla"].append((0.0, val, r.variant))

        else:
            if r.epsilon_value is not None:
                series[r.method].append((r.epsilon_value, val, r.variant))

    return dict(series)


def _plot_eps_vs_metric(
    records: list[CanonicalRecord],
    dataset: str,
    metric_key: str,
    metric_src: str,
    out_dir: Path,
    dpi: int,
) -> None:
    ds_records = [r for r in records if r.dataset == dataset]
    if not ds_records:
        return

    series_map = _collect_eps_series(ds_records, metric_key, metric_src)
    if not series_map:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_xscale("log")

    legend_handles = []

    for label, points in series_map.items():
        method = label.split("_")[0] if "_" in label else label
        color = _color(method)
        marker = _marker(method)
        is_dashed = label.endswith(("_sum", "_spent"))
        linestyle = "--" if is_dashed else "-"
        display = _series_display_name(label)

        if method == "vanilla":
            # Horizontal reference line — draw across current xlim later
            vals = [v for _, v, _ in points]
            if vals:
                ax.axhline(
                    sum(vals) / len(vals),
                    color=color,
                    linestyle=":",
                    linewidth=1.5,
                    zorder=1,
                )
                handle = mlines.Line2D([], [], color=color, linestyle=":", label=display)
                legend_handles.append(handle)
            continue

        eps_vals = [e for e, v, _ in points if not math.isnan(e)]
        util_vals = [v for e, v, _ in points if not math.isnan(e)]
        if not eps_vals:
            continue

        paired = sorted(zip(eps_vals, util_vals))
        xs, ys = zip(*paired)

        ax.plot(xs, ys, color=color, marker=marker, linestyle=linestyle, label=display)
        handle = mlines.Line2D(
            [], [],
            color=color, marker=marker, linestyle=linestyle, label=display,
        )
        legend_handles.append(handle)

    ylabel = {
        "match_accuracy": "Match Accuracy",
        "rouge_l": "ROUGE-L",
        "f1_score": "F1 Score",
        "bert_score_f1": "BERTScore F1",
        "mia_auc_bleu": "MIA-AUC (BLEU)",
    }.get(metric_key, metric_key)

    ax.set_xlabel("ε (privacy budget)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs ε — {dataset}")
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    stem = f"tradeoff_eps_vs_{metric_key}_{dataset}"
    _save(fig, out_dir, stem, dpi)
    print(f"[plot_tradeoff] saved {stem}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# (c)  Privacy vs utility Pareto scatter
# ---------------------------------------------------------------------------

def _plot_pareto(
    records: list[CanonicalRecord],
    dataset: str,
    utility_key: str,
    privacy_key: str,
    out_dir: Path,
    dpi: int,
) -> None:
    ds_records = [r for r in records if r.dataset == dataset]
    if not ds_records:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    all_points: list[tuple[float, float]] = []
    legend_handles = []

    for r in ds_records:
        util_val = r.utility.get(utility_key)
        priv_val = r.privacy.get(privacy_key)
        if util_val is None or priv_val is None:
            continue
        if math.isnan(util_val) or math.isnan(priv_val):
            continue

        color = _color(r.method)
        marker = _marker(r.method)

        eps_label = ""
        if r.epsilon_value is not None and not math.isnan(r.epsilon_value):
            eps_label = f"ε={r.epsilon_value:.1f}"

        if r.method == "vanilla":
            ax.scatter(priv_val, util_val, color=color, marker="*", s=200, zorder=5)
            ax.annotate("Vanilla", (priv_val, util_val), fontsize=7, ha="left")
        else:
            ax.scatter(priv_val, util_val, color=color, marker=marker, s=60, zorder=4)
            if eps_label:
                ax.annotate(eps_label, (priv_val, util_val), fontsize=6, ha="left", alpha=0.7)

        all_points.append((priv_val, util_val))

        # Add to legend only once per method
        if not any(getattr(h, "_label", "") == r.method for h in legend_handles):
            handle = mlines.Line2D(
                [], [],
                color=color, marker=marker, linestyle="None",
                markersize=7, label=r.method,
            )
            legend_handles.append(handle)

    # Pareto frontier
    if len(all_points) >= 2:
        front = pareto_front(all_points)
        if len(front) >= 2:
            fx, fy = zip(*front)
            # Step line: lower MIA = better privacy (x axis); higher util = better utility (y)
            ax.step(fx, fy, where="post", color="crimson", linewidth=1.5,
                    linestyle="-", zorder=3, label="Pareto frontier")

    utility_label = {
        "match_accuracy": "Match Accuracy",
        "rouge_l": "ROUGE-L",
        "f1_score": "F1 Score",
    }.get(utility_key, utility_key)

    privacy_label = {
        "mia_auc_bleu": "MIA-AUC (BLEU) ↓",
    }.get(privacy_key, privacy_key)

    ax.set_xlabel(f"{privacy_label}  (lower = more private)")
    ax.set_ylabel(utility_label)
    ax.set_title(f"Privacy vs Utility — {dataset}")
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    stem = f"pareto_privacy_utility_{dataset}"
    _save(fig, out_dir, stem, dpi)
    print(f"[plot_tradeoff] saved {stem}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Series display name
# ---------------------------------------------------------------------------

def _series_display_name(label: str) -> str:
    mapping = {
        "dp_rag_rdp": "DP-RAG (ε composed, solid)",
        "dp_rag_sum": "DP-RAG (ε₊ sum, dashed)",
        "dpvoterag_allocated": "DPVoteRAG (ε allocated)",
        "dpvoterag_spent": "DPVoteRAG (ε spent, dashed)",
        "dpsparsevoterag_allocated": "DPSparseVoteRAG (ε allocated)",
        "dpsparsevoterag_spent": "DPSparseVoteRAG (ε spent, dashed)",
        "vanilla": "Vanilla (no DP)",
        "dp_synrag": "DP-SynRAG",
    }
    return mapping.get(label, label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate DP-RAG trade-off figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--no-recompute-vanilla", action="store_true")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--vanilla-file", default=None)
    args = parser.parse_args(argv)

    records = load_all(
        results_dir=args.results_dir,
        datasets=args.datasets,
        methods=args.methods,
        recompute_vanilla=not args.no_recompute_vanilla,
        bertscore=args.bertscore,
    )

    if not records:
        print("[plot_tradeoff] No records loaded — check --results-dir and --datasets.")
        return

    out_dir = Path(args.figures_dir)
    datasets = sorted({r.dataset for r in records})

    utility_metrics = ["match_accuracy", "rouge_l"]
    privacy_metrics = ["mia_auc_bleu"]

    for dataset in datasets:
        # (a) ε vs utility
        for metric in utility_metrics:
            _plot_eps_vs_metric(records, dataset, metric, "utility", out_dir, args.dpi)

        # (b) ε vs privacy
        for metric in privacy_metrics:
            _plot_eps_vs_metric(records, dataset, metric, "privacy", out_dir, args.dpi)

        # (c) Pareto scatter
        _plot_pareto(records, dataset, "match_accuracy", "mia_auc_bleu", out_dir, args.dpi)


if __name__ == "__main__":
    main()
