"""Generate diagnostic plots for a single benchmark run.

Reads every result JSON under a run directory, normalises them into a tidy
DataFrame (headline ``epsilon_total`` + the ``metrics`` dict per file), and
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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / server-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Metrics that live in [0, 1] (higher = better, except mia where 0.5 = no leak).
UTILITY_01 = ["match_accuracy", "contains_accuracy", "rouge_l", "f1_score", "bert_score_f1", "hit_rate"]
PRIVACY_01 = ["s2mia_auc"]
HEATMAP_COLS = UTILITY_01 + PRIVACY_01

METHOD_COLORS = {
    "dp_rag": "#1f77b4",
    "dpvoterag": "#9467bd",
    "dpsparsevoterag": "#2ca02c",
    "dp_synrag": "#d62728",
    "dpksa": "#ff7f0e",
}
MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
# Methods we expect a runner to attempt, for the coverage panel.
EXPECTED_METHODS = ["dp_rag", "dpvoterag", "dpsparsevoterag", "dp_synrag", "dpksa"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def short_model(model: str | None) -> str:
    if not model:
        return "—"
    name = model.split("__")[-1].split("/")[-1]
    return name.replace("-instruct", "").replace("-Instruct", "")


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
            "method": method,
            "dataset": dataset,
            "model": model,
            "model_short": short_model(model),
            "epsilon": _to_float(d.get("epsilon_total")),
            # Which repetition this row came from; None outside a seed sweep. Without
            # it the repetitions of one cell are indistinguishable and any aggregation
            # silently averages them as if they were separate measurements.
            "seed": seed,
            "file": str(path),
        }
        for k in HEATMAP_COLS + ["perplexity"]:
            row[k] = _to_float(metrics.get(k))
        # AUC-MIA lives in `metrics` for --mia-only runs; older scored files
        # mirror it as a flat top-level `mia_auc`, so fall back to that.
        if math.isnan(row["s2mia_auc"]):
            row["s2mia_auc"] = _to_float(d.get("mia_auc"))

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
    return df.sort_values(["method", "dataset", "model_short", "epsilon"]).reset_index(drop=True)


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
    ax.set_title("Coverage: result files per method × dataset\n(— = not run; dpksa never ran, dpvoterag wrote no files this run)")
    fig.tight_layout()
    _save(fig, out)


def plot_tradeoff(df: pd.DataFrame, method: str, out: Path, metrics: list[str], logx: bool) -> None:
    """metric (rows) × dataset (cols) grid; one line per model; x = epsilon."""
    sub = df[df["method"] == method]
    if sub.empty:
        return
    datasets = sorted(sub["dataset"].unique())
    models = sorted(sub["model_short"].unique())
    nrows, ncols = len(metrics), len(datasets)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols + 1, 2.6 * nrows + 1),
                             squeeze=False, sharex=True)
    for r, metric in enumerate(metrics):
        for c, ds in enumerate(datasets):
            ax = axes[r][c]
            dsub = sub[sub["dataset"] == ds]
            any_pt = False
            for mi, model in enumerate(models):
                msub = dsub[dsub["model_short"] == model].sort_values("epsilon")
                if msub.empty or msub[metric].isna().all():
                    continue
                any_pt = True
                ax.plot(msub["epsilon"], msub[metric], marker=MODEL_MARKERS[mi % len(MODEL_MARKERS)],
                        label=model, color=f"C{mi}")
            if logx and any_pt:
                ax.set_xscale("log")
            if r == 0:
                ax.set_title(ds, fontsize=10)
            if c == 0:
                ax.set_ylabel(metric, fontsize=9)
            if r == nrows - 1:
                ax.set_xlabel("ε (epsilon_total)", fontsize=9)
            ax.grid(alpha=0.3)
            if metric in HEATMAP_COLS:
                ax.set_ylim(-0.02, 1.02)
            if not any_pt:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="#999")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=9,
                   bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{method} — privacy/utility tradeoff (utility vs ε)", y=1.04, fontsize=12)
    fig.tight_layout()
    _save(fig, out)


def plot_dp_synrag_bars(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["method"] == "dp_synrag"]
    if sub.empty:
        return
    metrics = ["contains_accuracy", "rouge_l", "f1_score", "bert_score_f1", "hit_rate"]
    datasets = sorted(sub["dataset"].unique())
    fig, ax = plt.subplots(figsize=(2.4 * len(datasets) + 3, 5))
    x = np.arange(len(datasets))
    w = 0.16
    for mi, metric in enumerate(metrics):
        vals = [sub[sub["dataset"] == ds][metric].mean() for ds in datasets]
        vals = [0 if (v is None or np.isnan(v)) else v for v in vals]
        bars = ax.bar(x + (mi - 2) * w, vals, w, label=metric, color=f"C{mi}")
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(x, [f"{ds}\n(ε≈{sub[sub['dataset']==ds]['epsilon'].mean():.0f})" for ds in datasets])
    ax.set_ylabel("metric value")
    ax.set_ylim(0, 1.0)
    ax.set_title("dp_synrag — utility per dataset (single, honest-but-huge ε)\n"
                 "medical 0% = task artifact (fictional names); movielens hit_rate=0 is a title-format false negative")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out)


def plot_metric_heatmap(df: pd.DataFrame, out: Path) -> None:
    labels = [f"{r.method}/{r.dataset}/{r.model_short}/ε={r.epsilon:.1f}" for r in df.itertuples()]
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
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].sort_values("epsilon")
        ax.scatter(sub["epsilon"], sub["s2mia_auc"], label=method,
                   color=METHOD_COLORS.get(method, "gray"), alpha=0.8, s=45)
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="0.5 (no leakage)")
    ax.set_xscale("log")
    ax.set_xlabel("ε (epsilon_total, log)")
    ax.set_ylabel("AUC-MIA (S2MIA)")
    ax.set_title("Membership-inference leakage vs ε  (closer to 0.5 = more private)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out)


def plot_perplexity(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    mi = 0
    for method in sorted(df["method"].unique()):
        for (ds, model), sub in df[df["method"] == method].groupby(["dataset", "model_short"]):
            sub = sub.sort_values("epsilon")
            if sub["perplexity"].isna().all():
                continue
            ax.plot(sub["epsilon"], sub["perplexity"], marker="o", ms=4,
                    label=f"{method}/{ds}/{model}", color=f"C{mi % 10}", alpha=0.8)
            mi += 1
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ε (epsilon_total, log)")
    ax.set_ylabel("perplexity (log, lower = more fluent)")
    ax.set_title("Generated-text perplexity vs ε")
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

    plot_coverage(df, out_dir / "00_coverage.png")
    plot_metric_heatmap(df, out_dir / "01_metric_heatmap.png")
    plot_tradeoff(df, "dpsparsevoterag", out_dir / "02_tradeoff_dpsparsevoterag.png",
                  metrics=["contains_accuracy", "rouge_l", "f1_score"], logx=True)
    plot_tradeoff(df, "dp_rag", out_dir / "03_tradeoff_dp_rag.png",
                  metrics=["contains_accuracy", "rouge_l", "f1_score", "bert_score_f1"], logx=False)
    plot_dp_synrag_bars(df, out_dir / "04_dp_synrag_utility.png")
    plot_mia(df, out_dir / "05_mia_vs_epsilon.png")
    plot_perplexity(df, out_dir / "06_perplexity_vs_epsilon.png")
    plot_sparse_diagnostics(df, out_dir / "07_dpsparsevoterag_diagnostics.png")

    # Tidy CSV alongside the plots for quick inspection.
    df.drop(columns=["file"]).to_csv(out_dir / "run_table.csv", index=False)
    print(f"  wrote run_table.csv")
    print(f"[plot_run] done → {out_dir}")


if __name__ == "__main__":
    main()
