"""Generate privacy–utility trade-off figures for BenchDP-RAG.

Produces three plot families per dataset:

(a) ε_workload vs utility metric (contains_accuracy, rouge_l)
(b) ε_workload vs privacy metric (s2mia_auc)
(c) Privacy vs utility scatter with Pareto frontier  ← the key comparison

The x axis of (a) and (b) is the SHARED one: ``epsilon_workload``, the privacy loss over
a Q-question workload composed for every method by one accountant at one delta. Each
method also carries its own epsilon in its own units, and those are not comparable with
one another — see ``_shared_epsilon``. (a) and (b) are faceted by the workload size,
because the same method priced over one question and over a hundred is two operating
points, not one point measured twice.

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
    "no_rag": "tab:cyan",
    "vanilla": "black",
    "dp_rag": "tab:blue",
    "dpvoterag": "tab:orange",
    "dpsparsevoterag": "tab:green",
    "dp_synrag": "tab:red",
    "dpksa": "tab:purple",
    "private_rag": "tab:pink",
    "private_rag_ada": "tab:brown",
}

_METHOD_MARKERS = {
    "no_rag": "*",
    "vanilla": "*",
    "dp_rag": "o",
    "dpvoterag": "s",
    "dpsparsevoterag": "D",
    "dp_synrag": "^",
    "dpksa": "v",
    "private_rag": "X",
    "private_rag_ada": "P",
}


def _method_of(label: str) -> str:
    """Method a series label belongs to.

    Matches the longest registered name the label starts with. Splitting on the first
    underscore mapped `dp_rag`, `dp_rag_rdp`, `dp_synrag`, `private_rag` and
    `private_rag_ada` all to the same fallback grey with the same marker, so five of
    the eight methods were indistinguishable in the headline figure.
    """
    return next(
        (m for m in sorted(_METHOD_COLORS, key=len, reverse=True) if label.startswith(m)),
        label,
    )


def _color(method: str) -> str:
    return _METHOD_COLORS.get(method, "tab:gray")


def _marker(method: str) -> str:
    return _METHOD_MARKERS.get(method, "o")


#: How the x axis is named wherever it is drawn. The methods are only comparable on
#: `epsilon_workload` — the privacy loss over a Q-question workload, composed for all
#: eight by one accountant at one delta — so the label names it rather than the bare "ε",
#: which each method also has in its own units.
EPSILON_AXIS_LABEL = "ε_workload (shared axis, δ=1e-2)"


def _shared_epsilon(record: CanonicalRecord) -> Optional[float]:
    """The record's position on the common privacy axis, or None if it has none.

    ``epsilon_workload`` is the figure the whole benchmark compares on. It falls back to
    ``epsilon_target``, the grid coordinate the cell was calibrated to reach, for the
    cells whose accountant declined and recorded a ``workload_error`` instead.

    Deliberately NOT a fallback: the method's own ``epsilon_value``. That number means a
    per-query PLD composition at δ=1e-3 for DP-RAG, a closed-form zCDP conversion for
    DP-SynRAG and an allocated budget for the vote methods, and mixing the three on one
    axis drew DP-RAG's (Q=100, target=5) cell at x=0.330 beside methods drawn at 5.
    """
    for value in (record.epsilon_workload, record.epsilon_target):
        if value is not None and not math.isnan(value):
            return float(value)
    return None


def _q_label(q: Optional[int]) -> str:
    return "Q = ?" if q is None else f"Q = {int(q)}"


#: Panel key for the reference conditions, which belong in every panel and in none. A
#: string so it can never be confused with a workload size or with `None`, the panel for
#: a record whose workload size was not recorded.
_REFERENCE_KEY = "__reference__"

#: The two conditions that run no privacy mechanism and therefore have no position on
#: the shared axis: the corpus retrieved unprotected, and no corpus at all. Both are
#: drawn as horizontal lines across every panel. Dropping them for want of an epsilon
#: leaves the curves ranked against each other with nothing to say how far above the
#: floor or below the ceiling they sit — and a method that never used the corpus reads
#: the same as one that used it well.
_REFERENCE_METHODS = frozenset({"vanilla", "no_rag"})


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
) -> dict[object, dict[str, list[tuple[float, float, str]]]]:
    """Return {workload_queries: {series_label: [(eps, value, variant), ...]}}.

    The outer key is the workload size the epsilon prices. It is part of the point's
    identity, not a repetition of it: the same method at Q=1 and at Q=100 is two
    operating points whose budgets were calibrated separately, and joining them into one
    line drew a curve through two different experiments.
    """
    series: dict[object, dict[str, list[tuple[float, float, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    # One series per (method, model). Keying on the method alone drew a single
    # polyline through every LLM and every repetition at the same epsilon, so a
    # "curve" was really 3-15 stacked points per x joined in arbitrary order.
    def _key(record) -> str:
        model = (record.model or "").split("/")[-1]
        return f"{record.method}" + (f" [{model}]" if model else "")

    for r in records:
        val = r.utility.get(metric_key) if metric_src == "utility" else r.privacy.get(metric_key)
        if val is None or math.isnan(val):
            continue

        if r.method in _REFERENCE_METHODS:
            # A reference condition has no place on the axis; it is drawn as a horizontal
            # line in every panel, so it is filed under a key of its own rather than
            # under a workload size. `None` would not do: that is the panel for a record
            # whose Q is genuinely unknown, and the two would be drawn as one.
            series[_REFERENCE_KEY][_key(r)].append((math.nan, val, r.variant))
            continue

        eps = _shared_epsilon(r)
        if eps is None:
            continue
        series[r.workload_queries][_key(r)].append((eps, val, r.variant))

    return {q: dict(labels) for q, labels in series.items()}


def _draw_series(
    ax: plt.Axes,
    series_map: dict[str, list[tuple[float, float, str]]],
    legend_handles: list,
    seen_labels: set[str],
) -> bool:
    """Draw one panel's series onto ``ax``; True if anything was plotted."""
    drawn = False
    for label, points in sorted(series_map.items()):
        method = _method_of(label)
        color = _color(method)
        marker = _marker(method)
        display = _series_display_name(label)

        if method in _REFERENCE_METHODS:
            vals = [v for _, v, _ in points]
            if vals:
                ax.axhline(sum(vals) / len(vals), color=color, linestyle=":",
                           linewidth=1.5, zorder=1)
                if display not in seen_labels:
                    seen_labels.add(display)
                    legend_handles.append(
                        mlines.Line2D([], [], color=color, linestyle=":", label=display)
                    )
                drawn = True
            continue

        paired = sorted((e, v) for e, v, _ in points if not math.isnan(e))
        if not paired:
            continue
        xs, ys = zip(*paired)
        ax.plot(xs, ys, color=color, marker=marker, linestyle="-", label=display)
        drawn = True
        if display not in seen_labels:
            seen_labels.add(display)
            legend_handles.append(
                mlines.Line2D([], [], color=color, marker=marker, linestyle="-", label=display)
            )
    return drawn


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

    by_q = _collect_eps_series(ds_records, metric_key, metric_src)
    # The reference conditions belong in every panel rather than in a panel of their own.
    reference = by_q.pop(_REFERENCE_KEY, {})
    if not by_q:
        if not reference:
            return
        by_q = {None: {}}

    ylabel = {
        "contains_accuracy": "Contains Accuracy",
        "match_accuracy": "Match Accuracy",
        "rouge_l": "ROUGE-L",
        "f1_score": "F1 Score",
        "bert_score_f1": "BERTScore F1",
        "s2mia_auc": "AUC-MIA (S2MIA)",
    }.get(metric_key, metric_key)

    # One panel per workload size, sharing the y axis so the panels are read against each
    # other: for a per-query method the curve slides right as Q grows, for a flat-regime
    # method it does not move, and that contrast is the point of the figure.
    q_values = sorted(by_q, key=lambda q: (q is None, q))
    fig, axes = plt.subplots(
        1, len(q_values), figsize=(4.2 * len(q_values) + 1.5, 4.5),
        squeeze=False, sharey=True,
    )
    legend_handles: list = []
    seen_labels: set[str] = set()

    for ax, q in zip(axes[0], q_values):
        ax.set_xscale("log")
        drawn = _draw_series(ax, {**by_q[q], **reference}, legend_handles, seen_labels)
        ax.set_title(_q_label(q), fontsize=10)
        ax.set_xlabel(EPSILON_AXIS_LABEL)
        ax.grid(True, which="both", alpha=0.3)
        if not drawn:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#999")
    axes[0][0].set_ylabel(ylabel)

    fig.suptitle(f"{ylabel} vs ε_workload — {dataset}")
    if legend_handles:
        fig.legend(handles=legend_handles, fontsize=8, loc="upper right")
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

        # Annotated with the shared axis and the workload it was priced over: those two
        # are the point's grid coordinates, and the method's own epsilon named a
        # different quantity for every method on the same scatter.
        eps_label = ""
        eps = _shared_epsilon(r)
        if eps is not None:
            eps_label = f"ε_w={eps:.1f}"
            if r.workload_queries is not None:
                eps_label += f", {_q_label(r.workload_queries)}"

        if r.method in _REFERENCE_METHODS:
            ax.scatter(priv_val, util_val, color=color, marker="*", s=200, zorder=5)
            ax.annotate(_method_display_name(r.method), (priv_val, util_val),
                        fontsize=7, ha="left")
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
        "contains_accuracy": "Contains Accuracy",
        "match_accuracy": "Match Accuracy",
        "rouge_l": "ROUGE-L",
        "f1_score": "F1 Score",
    }.get(utility_key, utility_key)

    privacy_label = {
        "s2mia_auc": "AUC-MIA (S2MIA) ↓",
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
    """Pretty name for a series label, which is ``<method>`` or ``<method> [model]``.

    Every method contributes exactly one series per panel now: they are drawn on the one
    axis they share, so the second series each of DP-RAG and the vote methods used to
    draw — its own composed/summed or allocated/spent epsilon — has no x of its own to
    be drawn against.
    """
    method = _method_of(label)
    return _method_display_name(method) + label[len(method):]


_DISPLAY_NAMES = {
    "no_rag": "No retrieval (zero-shot)",
    "vanilla": "Vanilla (no DP)",
    "dp_rag": "DP-RAG",
    "dpvoterag": "DPVoteRAG",
    "dpsparsevoterag": "DPSparseVoteRAG",
    "dp_synrag": "DP-SynRAG",
    "dpksa": "DP-KSA",
    "private_rag": "MuRAG",
    "private_rag_ada": "MuRAG-ADA",
}


def _method_display_name(method: str) -> str:
    return _DISPLAY_NAMES.get(method, method)


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

    # Headline utility = contains_accuracy (paper definition of "match accuracy"
    # is containment, not exact equality). match_accuracy kept only in the table.
    utility_metrics = ["contains_accuracy", "rouge_l"]
    privacy_metrics = ["s2mia_auc"]

    for dataset in datasets:
        # (a) ε vs utility
        for metric in utility_metrics:
            _plot_eps_vs_metric(records, dataset, metric, "utility", out_dir, args.dpi)

        # (b) ε vs privacy
        for metric in privacy_metrics:
            _plot_eps_vs_metric(records, dataset, metric, "privacy", out_dir, args.dpi)

        # (c) Pareto scatter
        _plot_pareto(records, dataset, "contains_accuracy", "s2mia_auc", out_dir, args.dpi)


if __name__ == "__main__":
    main()
