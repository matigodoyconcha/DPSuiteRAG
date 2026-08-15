"""Generate diagnostic plots for a single benchmark run.

Reads every result JSON under a run directory, normalises them into a tidy
DataFrame (the shared-axis ``epsilon_workload`` + the ``metrics`` dict per file), and
writes a set of PNGs into ``<run-dir>/plots/`` that make it obvious which
(method x dataset x model x epsilon) cells are producing signal and which are
degenerate (zeros / NaN / saturated / abstaining).

Usage:
    python analysis/plot_run.py [--run-dir results/run_YYYYmmdd_HHMMSS]

With no argument it picks the most recent ``run_*`` directory under results/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / server-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imported rather than restated: which payload variants are their own method is one rule,
# and a diagnostic that split them differently from the summary table would be reporting
# a different set of methods for the same run.
from analysis.load_results import method_of_payload  # noqa: E402

# Metrics that live in [0, 1] (higher = better, except mia where 0.5 = no leak).
UTILITY_01 = ["match_accuracy", "contains_accuracy", "rouge_l", "f1_score", "bert_score_f1", "hit_rate"]
PRIVACY_01 = ["s2mia_auc"]
HEATMAP_COLS = UTILITY_01 + PRIVACY_01

#: Whether the attack that produced an AUC had anything to rank. Carried alongside the
#: AUC because the two readings of 0.5 — the corpus was protected, and the mechanism
#: emitted text no attack could separate — are opposite claims that the number alone
#: does not distinguish. Written by metrics.mia; absent from pre-2026-08 records, which
#: load as NaN and are drawn as unknown rather than as either answer.
MIA_SIGNAL_COLS = [
    "s2mia_informative", "s2mia_distinct_score_rate",
    "rag_mia_informative", "rag_mia_yes_no_rate",
]

#: Per-cell diagnostics that are not metrics. What a run *spent* is deliberately absent:
#: it is denominated in each method's own units — a per-query PLD composition here, an
#: allocated budget there — so a figure drawing two of them together would compare
#: numbers that do not mean the same thing, which is what the shared axis exists to stop.
DIAGNOSTIC_COLS = ["n_docs_per_query", "top_p"]

#: Whether the mechanism ran at all in this cell. DPSparseVoteRAG's sparse path routes a
#: token to the public model without spending budget, and 339 of its 447 cells released
#: **zero** private tokens — the DP mechanism never fired, yet the public answer was
#: scored like any other. Read `contains_accuracy` alone and that reads as free utility
#: at a tight budget; it is the generator answering from its weights. Carried into the
#: table so the flat CSV cannot hide what the payload records.
MECHANISM_ACTIVITY_COLS = ["private_answer_rate", "tokens_private", "public_only"]

METHOD_COLORS = {
    "dp_rag": "#1f77b4",
    "dpvoterag": "#9467bd",
    "dpsparsevoterag": "#2ca02c",
    "dp_synrag": "#d62728",
    "dpksa": "#ff7f0e",
    # MURAG and MURAG-ADA are told apart by the payload's variant and get a colour each:
    # they differ in the retrieval mechanism, which is the comparison the pair is for.
    "private_rag": "#e377c2",
    "private_rag_ada": "#8c564b",
    "vanilla": "#7f7f7f",
    "no_rag": "#17becf",
}
MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
# Methods we expect a runner to attempt, for the coverage panel.
EXPECTED_METHODS = [
    "no_rag", "vanilla", "dp_rag", "dpvoterag", "dpsparsevoterag", "dp_synrag", "dpksa",
    "private_rag", "private_rag_ada",
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def short_model(model: str | None) -> str:
    if not model:
        return "—"
    name = model.split("__")[-1].split("/")[-1]
    return name.replace("-instruct", "").replace("-Instruct", "")


#: The payload keys that carry the shared privacy axis, in the order they are trusted.
#: ``epsilon_workload`` is the privacy loss over the record's Q-question workload,
#: composed for every method by one accountant at one delta, and it is the only epsilon
#: in this file that means the same thing in two different rows. ``epsilon_target`` is
#: the grid coordinate the cell was calibrated to reach, used when the accountant
#: declined the cell and recorded a `workload_error` instead of a number.
#:
#: What is deliberately NOT here is each method's own ``epsilon_total``: a per-query PLD
#: composition at delta=1e-3 for DP-RAG, a closed-form zCDP conversion for DP-SynRAG, an
#: allocated budget for the vote methods. Plotting those on one x placed DP-RAG's
#: (Q=100, target=5) cell at 0.330 next to methods drawn at their targets.
_AXIS_KEYS = ("epsilon_workload", "epsilon_target")

#: How the x axis is named wherever it is drawn, so no figure can claim the shared axis
#: while showing a method's own units.
EPSILON_AXIS_LABEL = "ε_workload (shared axis, δ=1e-2)"


def shared_epsilon(payload: dict) -> float:
    """This record's position on the common privacy axis, NaN when it has none."""
    for key in _AXIS_KEYS:
        value = _to_float(payload.get(key))
        if not math.isnan(value):
            return value
    return math.nan


def load_run(run_dir: Path) -> pd.DataFrame:
    """Walk <run-dir>/<method>[/seed_<n>]/<dataset>/<model>/*.json into a tidy frame."""
    rows = []
    for path in sorted(run_dir.glob("*/**/*.json")):
        rel = path.relative_to(run_dir).parts
        if len(rel) < 4:  # method/dataset/model/file.json
            continue
        method, rest = rel[0], list(rel[1:])
        # A repetition sweep inserts seed_<n> between the method and the dataset;
        # reading the levels positionally without accounting for it labels the
        # dataset "seed_1" and shifts the model along with it.
        seed = None
        if rest and rest[0].startswith("seed_"):
            seed = rest.pop(0)
            if len(rest) < 3:
                continue
        dataset, model = rest[0], rest[1]
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {path}: {exc}")
            continue

        metrics = d.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        row = {
            # From the payload, not only the directory: MURAG and MURAG-ADA are one
            # runner writing one method name, and they are two mechanisms.
            "method": method_of_payload(method, d),
            "dataset": dataset,
            "model": model,
            "model_short": short_model(model),
            "epsilon": shared_epsilon(d),
            # The grid coordinate the cell was calibrated to reach, kept beside the
            # epsilon it achieved. Holding the budget fixed while the workload varies
            # needs the coordinate: `epsilon` is a bisection result and two cells of one
            # target land on neighbouring but unequal values, so grouping on it would
            # split a panel that is meant to be one.
            "epsilon_target": _to_float(d.get("epsilon_target")),
            # The workload size the epsilon on this row prices. Two rows of one method at
            # different Q are different operating points, so every plot below separates
            # them; joined into one line they read as a curve through a method's own
            # sweep when they are two sweeps side by side.
            "workload_queries": _to_float(d.get("workload_queries")),
            # Which repetition this row came from; None outside a seed sweep. Without
            # it the repetitions of one cell are indistinguishable and any aggregation
            # silently averages them as if they were separate measurements.
            "seed": seed,
            "file": str(path),
        }
        for k in HEATMAP_COLS + MIA_SIGNAL_COLS + ["perplexity"]:
            row[k] = _to_float(metrics.get(k))
        # AUC-MIA lives in `metrics` for --mia-only runs; older scored files
        # mirror it as a flat top-level `mia_auc`, so fall back to that.
        if math.isnan(row["s2mia_auc"]):
            row["s2mia_auc"] = _to_float(d.get("mia_auc"))

        # Diagnostics the runners write beside the metrics rather than inside them.
        for k in DIAGNOSTIC_COLS:
            row[k] = _to_float(d.get(k))
        # The mechanism-activity fields sit at the top level for some runners and inside
        # the run-stats block for others; both spellings mean the same thing.
        stats = d.get("run_stats") or d.get("dp_spent") or {}
        for k in MECHANISM_ACTIVITY_COLS:
            v = d.get(k)
            if v is None and isinstance(stats, dict):
                v = stats.get(k)
            row[k] = _to_float(v)

        # dpsparse / dpvote diagnostics: abstention + empty-prediction rate.
        preds = d.get("predictions")
        if isinstance(preds, list) and preds:
            n = len(preds)
            row["empty_pred_rate"] = sum(1 for p in preds if not str(p).strip()) / n
        exits = d.get("exit_statuses")
        if isinstance(exits, list) and exits:
            n = len(exits)
            row["abstain_rate"] = sum(1 for e in exits if e == 2) / n  # exit 2 = budget exhausted
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["method", "dataset", "model_short", "workload_queries", "epsilon"]
    ).reset_index(drop=True)


def _q_label(q: float) -> str:
    """The workload size as it appears in a legend; blank for rows that carry none."""
    return "" if q is None or (isinstance(q, float) and math.isnan(q)) else f"Q={int(q)}"


def series_labels(df: pd.DataFrame) -> pd.Series:
    """Name each row's plottable series: its model, and its workload size when it varies.

    A line may only join points that differ in epsilon alone. Naming a series by the
    model alone joined a method's Q=1 sweep to its Q=100 sweep, drawing one zig-zag
    through two different experiments — and for the flat-regime methods, whose epsilon
    does not move with Q at all, it stacked four identical points on one x.
    """
    models = df["model_short"].astype(str)
    if "workload_queries" not in df.columns or df["workload_queries"].nunique(dropna=True) <= 1:
        return models
    return models.str.cat(df["workload_queries"].map(_q_label), sep=" ").str.strip()


def _to_float(v) -> float:
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return math.nan


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_coverage(df: pd.DataFrame, out: Path) -> None:
    """Grid of method x dataset showing #(model,epsilon) cells that ran."""
    datasets = sorted(df["dataset"].unique())
    fig, ax = plt.subplots(figsize=(1.6 * len(datasets) + 4, 0.7 * len(EXPECTED_METHODS) + 2))
    grid = np.full((len(EXPECTED_METHODS), len(datasets)), np.nan)
    for i, m in enumerate(EXPECTED_METHODS):
        for j, ds in enumerate(datasets):
            sub = df[(df["method"] == m) & (df["dataset"] == ds)]
            if not sub.empty:
                grid[i, j] = len(sub)

    cmap = plt.cm.Greens.copy()
    cmap.set_bad("#f2f2f2")
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, aspect="auto", vmin=0, vmax=max(1, np.nanmax(grid)))
    ax.set_xticks(range(len(datasets)), datasets, rotation=30, ha="right")
    ax.set_yticks(range(len(EXPECTED_METHODS)), EXPECTED_METHODS)
    for i in range(len(EXPECTED_METHODS)):
        for j in range(len(datasets)):
            val = grid[i, j]
            txt = "—" if np.isnan(val) else f"{int(val)} runs"
            ax.text(j, i, txt, ha="center", va="center",
                    color="#666" if np.isnan(val) else "black", fontsize=9)
    ax.set_title("Coverage: result files per method × dataset  (— = the method wrote "
                 "nothing for that dataset in this run)")
    fig.tight_layout()
    _save(fig, out)


def plot_tradeoff(df: pd.DataFrame, method: str, out: Path, metrics: list[str], logx: bool) -> None:
    """metric (rows) × dataset (cols) grid; one line per series; x = ε_workload."""
    sub = df[df["method"] == method]
    if sub.empty:
        return
    sub = sub.assign(series=series_labels(sub))
    datasets = sorted(sub["dataset"].unique())
    # Colour and marker are assigned once for the whole method, so a series keeps its
    # identity across the dataset columns instead of being recoloured per panel.
    series = sorted(sub["series"].unique())
    nrows, ncols = len(metrics), len(datasets)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols + 1, 2.6 * nrows + 1),
                             squeeze=False, sharex=True)
    for r, metric in enumerate(metrics):
        for c, ds in enumerate(datasets):
            ax = axes[r][c]
            dsub = sub[sub["dataset"] == ds]
            any_pt = False
            for mi, label in enumerate(series):
                msub = dsub[dsub["series"] == label].sort_values("epsilon")
                if msub.empty or msub[metric].isna().all():
                    continue
                any_pt = True
                ax.plot(msub["epsilon"], msub[metric], marker=MODEL_MARKERS[mi % len(MODEL_MARKERS)],
                        label=label, color=f"C{mi % 10}")
            if logx and any_pt:
                ax.set_xscale("log")
            if r == 0:
                ax.set_title(ds, fontsize=10)
            if c == 0:
                ax.set_ylabel(metric, fontsize=9)
            if r == nrows - 1:
                ax.set_xlabel(EPSILON_AXIS_LABEL, fontsize=9)
            ax.grid(alpha=0.3)
            if metric in HEATMAP_COLS:
                ax.set_ylim(-0.02, 1.02)
            if not any_pt:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="#999")
    # Gathered across every panel, not just the first: a series absent from the leftmost
    # dataset was drawn in the grid and missing from the legend.
    seen: dict[str, object] = {}
    for row in axes:
        for ax in row:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(label, handle)
    labels, handles = list(seen), list(seen.values())
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=9,
                   bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{method} — privacy/utility tradeoff (utility vs ε_workload)",
                 y=1.04, fontsize=12)
    fig.tight_layout()
    _save(fig, out)


def _eps_range(values) -> str:
    """The span of shared-axis epsilons a bar was averaged over, as an axis annotation."""
    vals = sorted({round(float(v), 3) for v in values if not np.isnan(v)})
    if not vals:
        return "—"
    return f"{vals[0]:g}" if len(vals) == 1 else f"{vals[0]:g}–{vals[-1]:g}"


def plot_dp_synrag_bars(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["method"] == "dp_synrag"]
    if sub.empty:
        return
    metrics = ["contains_accuracy", "rouge_l", "f1_score", "bert_score_f1", "hit_rate"]
    datasets = sorted(sub["dataset"].unique())
    fig, ax = plt.subplots(figsize=(2.4 * len(datasets) + 3, 5))
    x = np.arange(len(datasets))
    w = 0.16
    # One bar per (dataset, metric), over the distinct operating points only. The method
    # is flat in Q, so it files the same measurement at every workload size; averaging
    # the copies in would leave the height unchanged and the count wrong.
    sub = sub.drop_duplicates(subset=["dataset", "model", "epsilon"])
    for mi, metric in enumerate(metrics):
        vals = [sub[sub["dataset"] == ds][metric].mean() for ds in datasets]
        vals = [0 if (v is None or np.isnan(v)) else v for v in vals]
        bars = ax.bar(x + (mi - 2) * w, vals, w, label=metric, color=f"C{mi}")
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(x, [
        f"{ds}\n(ε_workload {_eps_range(sub[sub['dataset'] == ds]['epsilon'])})"
        for ds in datasets
    ])
    ax.set_ylabel("metric value")
    ax.set_ylim(0, 1.0)
    ax.set_title("dp_synrag — utility per dataset, averaged over the ε_workload sweep\n"
                 "medical 0% = task artifact (fictional names); movielens hit_rate=0 is a title-format false negative")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out)


def plot_metric_heatmap(df: pd.DataFrame, out: Path) -> None:
    # Q is part of a row's identity, not decoration: a method priced over one question
    # and the same method priced over a hundred are two rows whose labels were otherwise
    # identical, and a reader could not tell which was which.
    labels = [
        f"{r.method}/{r.dataset}/{r.model_short}/ε_w={r.epsilon:.1f}"
        + (f"/{_q_label(r.workload_queries)}" if _q_label(r.workload_queries) else "")
        for r in df.itertuples()
    ]
    mat = df[HEATMAP_COLS].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.1 * len(HEATMAP_COLS) + 5, 0.34 * len(df) + 1.5))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#d9d9d9")  # NaN cells = light gray
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(HEATMAP_COLS)), HEATMAP_COLS, rotation=35, ha="right")
    ax.set_yticks(range(len(df)), labels, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "NaN", ha="center", va="center", fontsize=6, color="#444")
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="value (mia: 0.5 = no leak)")
    ax.set_title("All configs × metrics (gray = NaN). Spot zeros / NaN / saturation at a glance.")
    fig.tight_layout()
    _save(fig, out)


def plot_mia(df: pd.DataFrame, out: Path) -> None:
    """AUC-MIA against the shared axis, with the cells that measured nothing marked.

    A cell whose mechanism emitted text carrying no trace of any document scores every
    member and non-member alike, and the AUC lands at 0.5 — the same place a cell that
    genuinely protected its corpus lands. Drawing both as one dot claims the first is
    evidence of privacy. Informative cells are filled; non-informative ones are hollow
    and must be read as "no measurement", not as "no leakage". Cells from records
    written before the diagnostic existed carry no flag and are drawn hollow too, since
    what they measured is exactly what is unknown.
    """
    scored = df.dropna(subset=["s2mia_auc"]) if "s2mia_auc" in df.columns else df.iloc[:0]
    if scored.empty:
        # A utility run carries no attack. Drawing an empty log axis raises rather than
        # producing a blank figure, which took the whole plotting pass down with it.
        print(f"  [skip] {out.name}: no cell carries an AUC-MIA (this is a utility run)")
        return
    df = scored

    fig, ax = plt.subplots(figsize=(8.5, 5))
    informative = df.get("s2mia_informative")
    is_informative = (
        pd.Series(False, index=df.index) if informative is None else informative.fillna(0) > 0.5
    )

    for method in sorted(df["method"].unique()):
        sel = df["method"] == method
        colour = METHOD_COLORS.get(method, "gray")
        good = df[sel & is_informative]
        weak = df[sel & ~is_informative]
        if not good.empty:
            ax.scatter(good["epsilon"], good["s2mia_auc"], label=method,
                       color=colour, alpha=0.85, s=48)
        if not weak.empty:
            ax.scatter(weak["epsilon"], weak["s2mia_auc"],
                       label=None if not good.empty else f"{method} (no signal)",
                       facecolors="none", edgecolors=colour, alpha=0.55, s=48, linewidths=1.2)

    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(f"{EPSILON_AXIS_LABEL}, log")
    ax.set_ylabel("AUC-MIA (S2MIA)")
    n_weak = int((~is_informative).sum())
    ax.set_title(
        "Membership-inference leakage vs ε_workload  (0.5 = no leakage)\n"
        f"hollow = attack had nothing to rank, not evidence of privacy "
        f"({n_weak}/{len(df)} cells)",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out)


def plot_perplexity(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    mi = 0
    for method in sorted(df["method"].unique()):
        msub = df[df["method"] == method]
        msub = msub.assign(series=series_labels(msub))
        # Grouped on the series, so a line joins points that differ in epsilon alone —
        # the workload size is part of the series, never a step along the x axis.
        for (ds, label), sub in msub.groupby(["dataset", "series"]):
            sub = sub.sort_values("epsilon")
            if sub["perplexity"].isna().all():
                continue
            ax.plot(sub["epsilon"], sub["perplexity"], marker="o", ms=4,
                    label=f"{method}/{ds}/{label}", color=f"C{mi % 10}", alpha=0.8)
            mi += 1
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{EPSILON_AXIS_LABEL}, log")
    ax.set_ylabel("perplexity (log, lower = more fluent)")
    ax.set_title("Generated-text perplexity vs ε_workload")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    _save(fig, out)


def plot_sparse_diagnostics(df: pd.DataFrame, out: Path) -> None:
    """Empty-prediction + abstention rate for dpsparsevoterag — the BERTScore-NaN cause."""
    sub = df[(df["method"] == "dpsparsevoterag") & df["empty_pred_rate"].notna()]
    if sub.empty:
        return
    sub = sub.assign(cfg=sub["dataset"] + "/" + sub["model_short"])
    cfgs = sorted(sub["cfg"].unique())
    eps_vals = sorted(sub["epsilon"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(6 * 2 + 1, 0.4 * len(cfgs) + 3), sharey=True)
    for ax, col, title in zip(
        axes,
        ["empty_pred_rate", "abstain_rate"],
        ["Empty predictions (\"\") → BERTScore=NaN", "Abstention rate (exit_status=2, budget spent)"],
    ):
        y = np.arange(len(cfgs))
        h = 0.8 / max(1, len(eps_vals))
        for ei, eps in enumerate(eps_vals):
            vals = [sub[(sub["cfg"] == c) & (sub["epsilon"] == eps)][col].mean() for c in cfgs]
            vals = [0 if np.isnan(v) else v for v in vals]
            ax.barh(y + (ei - len(eps_vals) / 2) * h, vals, h, label=f"ε={eps:g}", color=f"C{ei}")
        ax.set_yticks(y, cfgs, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of 40 queries")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend(fontsize=8, title="ε", loc="lower right")
    fig.suptitle("dpsparsevoterag — why utility is unreliable (SVT halts before emitting tokens)", y=1.02)
    fig.tight_layout()
    _save(fig, out)


def plot_workload_crossings(df: pd.DataFrame, out: Path, metric: str = "contains_accuracy") -> None:
    """Utility against the workload size, one panel per target on the shared axis.

    The figure the benchmark exists to produce. Every other plot here puts epsilon on x
    and treats Q as a facet, which answers "who wins at this budget" — a question whose
    answer depends entirely on a workload size the reader cannot see. Here Q is the axis
    and the budget is held fixed, so the three regimes separate by shape rather than by
    value: a per-query method spends its budget again on every question and decays, a
    method whose guarantee already covers the stream is flat, and where the two meet is
    the workload size above which paying per query stops being worth it.

    Points are drawn as well as lines because a flat method has one measured cell that
    the analysis replicates across Q — a line through it is an extrapolation of its own
    guarantee, not four measurements.
    """
    if metric not in df.columns or df["workload_queries"].nunique(dropna=True) < 2:
        print(f"  [skip] {out.name}: needs >= 2 workload sizes and a {metric} column")
        return

    targets = sorted(t for t in df["epsilon_target"].dropna().unique()) if \
        "epsilon_target" in df.columns else []
    if not targets:
        print(f"  [skip] {out.name}: no epsilon_target column to hold fixed")
        return

    # Only the datasets every DP method reached. A partially finished sweep leaves one
    # method averaging six datasets and another four, and the gap between their curves
    # is then partly the difference between two corpora rather than between two
    # mechanisms — the comparison this figure exists to make, silently broken.
    swept = df[df["epsilon_target"].notna()]
    per_method = swept.groupby("method")["dataset"].apply(set)
    common = set.intersection(*per_method) if len(per_method) else set()
    if not common:
        print(f"  [skip] {out.name}: no dataset is shared by every method")
        return
    dropped = sorted(set(swept["dataset"]) - common)
    df = df[df["dataset"].isin(common)]

    # The non-private condition, as a reference line. It carries no epsilon, so it is
    # absent from every panel otherwise, and without it the panels show methods ranked
    # against each other with no indication of how far below the ceiling all of them sit.
    ref = df[(df["method"] == "vanilla") & df[metric].notna()][metric]
    reference = float(ref.mean()) if not ref.empty else None
    # The no-retrieval condition, as the floor. A method sitting on it answered from the
    # generator's own knowledge, so whatever its budget bought, it did not buy the
    # corpus — and read without this line a curve near the ceiling and a curve that never
    # used the corpus look equally successful.
    zero = df[(df["method"] == "no_rag") & df[metric].notna()][metric]
    floor = float(zero.mean()) if not zero.empty else None

    ncols = min(len(targets), 3)
    nrows = math.ceil(len(targets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.1 * nrows),
                             squeeze=False, sharey=True)

    for ax, target in zip(axes.flat, targets):
        panel = df[df["epsilon_target"] == target]
        if reference is not None:
            ax.axhline(reference, ls=":", color="black", lw=1.3,
                       label="vanilla (non-private)")
        if floor is not None:
            ax.axhline(floor, ls=":", color=METHOD_COLORS["no_rag"], lw=1.3,
                       label="no_rag (no retrieval)")
        for method in sorted(panel["method"].unique()):
            sub = (panel[panel["method"] == method]
                   .groupby("workload_queries", as_index=False)[metric].mean()
                   .sort_values("workload_queries")
                   .dropna(subset=[metric]))
            if sub.empty:
                continue
            ax.plot(sub["workload_queries"], sub[metric], marker="o", ms=5,
                    color=METHOD_COLORS.get(method, "gray"), label=method, lw=1.6)
        ax.set_xscale("log")
        ax.set_xlabel("Q (questions the guarantee covers), log")
        ax.set_title(f"ε_workload = {target:g}", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(targets):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(metric)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 8), fontsize=8)
    subtitle = f"{len(common)} dataset(s) every method reached: {', '.join(sorted(common))}"
    if dropped:
        subtitle += f"   |   excluded, not run by every method yet: {', '.join(dropped)}"
    fig.suptitle(
        "Utility against workload size at a fixed shared-axis budget — "
        "flat lines pay once for the stream, decaying lines pay per question\n"
        + subtitle,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    _save(fig, out)


def plot_released_documents(df: pd.DataFrame, out: Path) -> None:
    """How many documents the retrieval mechanism released per question, by dataset.

    The empirical check on DP-RAG's top-p recalibration. The released set grows with the
    corpus, so the full 21M-passage DPR datasets run at a `top_p` three orders of
    magnitude below the one the 20k corpora use; whether that held the set at a
    comparable size is a measurement, not an assumption, and this is the measurement.
    """
    sub = df.dropna(subset=["n_docs_per_query"])
    sub = sub[sub["n_docs_per_query"] > 0]
    if sub.empty:
        print(f"  [skip] {out.name}: no cell recorded n_docs_per_query")
        return

    datasets = sorted(sub["dataset"].unique())
    fig, ax = plt.subplots(figsize=(max(6.5, 1.3 * len(datasets) + 3), 4.6))
    ax.boxplot([sub[sub["dataset"] == d]["n_docs_per_query"].values for d in datasets],
               labels=datasets, showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("documents released per question, log")
    ax.set_title("Size of the released set per dataset  (DP-RAG top-p)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, out)


def plot_utility_by_llm(df: pd.DataFrame, out: Path, metric: str = "contains_accuracy") -> None:
    """Utility per generator, with the cells that never ran marked as absent.

    A generator that OOMs is skipped by the runner and leaves no cell. Averaged away,
    its absence reads as poor performance on the datasets it never reached — Phi-3.5
    lost 145 cells that way, all on the two long-document datasets. An empty slot here
    is labelled, so a missing measurement cannot be mistaken for a low one.
    """
    if metric not in df.columns or "model_short" not in df.columns:
        print(f"  [skip] {out.name}: needs {metric} and model_short")
        return
    pivot = df.pivot_table(index="model_short", columns="dataset", values=metric, aggfunc="mean")
    counts = df.pivot_table(index="model_short", columns="dataset", values=metric, aggfunc="count")
    if pivot.empty:
        print(f"  [skip] {out.name}: nothing to pivot")
        return

    fig, ax = plt.subplots(figsize=(1.5 * len(pivot.columns) + 4, 0.7 * len(pivot) + 3))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            n = counts.values[i, j] if counts is not None else 0
            if pd.isna(v):
                ax.text(j, i, "no run", ha="center", va="center", fontsize=8, color="crimson")
            else:
                ax.text(j, i, f"{v:.2f}\nn={int(n)}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.55 else "black")
    fig.colorbar(im, ax=ax, label=metric)
    ax.set_title(f"{metric} by generator and dataset  ('no run' = cells the runner never produced)",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, out)


def _save(fig, out: Path) -> None:
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def latest_run(results: Path) -> Path:
    runs = sorted(results.glob("run_*"))
    if not runs:
        raise SystemExit(f"no run_* directories under {results}")
    return runs[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="run directory (default: latest results/run_*)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    run_dir = args.run_dir or latest_run(repo / "results")
    if not run_dir.is_absolute():
        run_dir = (repo / run_dir).resolve()
    print(f"[plot_run] run dir: {run_dir}")

    df = load_run(run_dir)
    if df.empty:
        raise SystemExit("no result JSONs found under the run directory")
    print(f"[plot_run] loaded {len(df)} result files "
          f"({df['method'].nunique()} methods, {df['dataset'].nunique()} datasets)")

    out_dir = run_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    # The headline figure first: it is the one the benchmark is for, and putting it at
    # 00 means a truncated run still writes it.
    plot_workload_crossings(df, out_dir / "00_workload_crossings.png")
    plot_coverage(df, out_dir / "01_coverage.png")
    plot_metric_heatmap(df, out_dir / "02_metric_heatmap.png")

    # One trade-off panel per method that produced cells, rather than the two that were
    # named here by hand: a method absent from this list is a method whose epsilon-utility
    # curve nobody ever looks at.
    for method in sorted(df["method"].unique()):
        plot_tradeoff(
            df, method, out_dir / f"03_tradeoff_{method}.png",
            metrics=["contains_accuracy", "rouge_l", "f1_score", "bert_score_f1"],
            logx=True,
        )

    plot_dp_synrag_bars(df, out_dir / "04_dp_synrag_utility.png")
    plot_mia(df, out_dir / "05_mia_vs_epsilon.png")
    plot_perplexity(df, out_dir / "06_perplexity_vs_epsilon.png")
    plot_sparse_diagnostics(df, out_dir / "07_dpsparsevoterag_diagnostics.png")
    plot_released_documents(df, out_dir / "09_released_documents.png")
    plot_utility_by_llm(df, out_dir / "10_utility_by_llm.png")

    # Tidy CSV alongside the plots for quick inspection.
    df.drop(columns=["file"]).to_csv(out_dir / "run_table.csv", index=False)
    print(f"  wrote run_table.csv")
    print(f"[plot_run] done → {out_dir}")


if __name__ == "__main__":
    main()
