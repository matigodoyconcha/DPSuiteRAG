"""The results figures of the benchmark, for any run and any set of methods.

Three figures, each with a CSV of the values it draws:

* ``reference_range``  the no-retrieval floor and the non-private ceiling of every
  corpus. The gap between them is all a private method can win on that corpus.
* ``tradeoff_q<Q>``    utility against the shared privacy budget at one workload size,
  one panel per corpus, one line per DP method.
* ``workload_axis``    utility against the workload size Q at fixed budgets, one row
  per budget and one column per corpus (or group of corpora).

Only what the run contains is drawn: a method with no cells is left out of the panels
and the legend, and a corpus enters the method figures only when it has both reference
conditions and its ceiling lies above its floor (``--all-datasets`` keeps every one).
Every value is a mean over the generators that answered.

Methods whose guarantee covers a whole stream of queries (``corpus_total`` in
``metrics.workload_accounting.METHOD_REGIMES``: MuRAG and MuRAG-Ada) answer one stream,
and their point at workload Q is the utility over the first Q answers of the longest
stream the run holds for that cell. A Q longer than the stream is left out, never
padded. The per-query methods are read at each Q as the runner priced them.

Usage::

    python -m analysis.plot_results --run-dir results/run_<stamp>
    # several directories are merged (e.g. a separate 100-query MuRAG stream run)
    python -m analysis.plot_results --run-dir results/run_a --run-dir results/stream100
    # redraw from the CSV a previous call wrote, without the result JSONs
    python -m analysis.plot_results --from-csv results/figures/results_cells.csv

Figures go to ``--out`` (default ``<first run dir>/figures``) as PDF and PNG.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.load_results import _VARIANT_METHODS, method_of_payload  # noqa: E402
from analysis.plot_run import load_run  # noqa: E402
from analysis.style import (CEILING_LABEL, CEILING_METHOD, EPSILON_AXIS,  # noqa: E402
                            FLOOR_LABEL, FLOOR_METHOD, INK, METHOD_STYLE, MUTED, REF,
                            REFERENCE_STYLE, WORKLOAD_AXIS, apply, dataset_label,
                            half_up, line_kwargs, style_axes)
from metrics.workload_accounting import METHOD_REGIMES  # noqa: E402

#: Metrics recomputed from a stream prefix. Others are read from the payload only.
STREAM_METRICS = ("contains_accuracy",)
#: Columns of the tidy cell table every figure is drawn from.
CELL_COLUMNS = ["method", "dataset", "model", "epsilon_target", "workload_queries"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _stream_methods() -> set[str]:
    return {m for m, regime in METHOD_REGIMES.items() if regime == "corpus_total"}


def _payload_stream(path: Path) -> tuple[list, list]:
    payload = json.load(open(path, encoding="utf-8"))
    results = payload.get("results") or []
    preds = [r.get("prediction") for r in results]
    refs = [r.get("answers") or r.get("reference") for r in results]
    return preds, refs


def stream_cells(run_dirs: list[Path], workloads: list[int], metric: str) -> pd.DataFrame:
    """Utility over the first Q answers of each stream-priced cell (see module doc)."""
    if metric not in STREAM_METRICS:
        raise SystemExit(f"stream methods can only be rescored on {STREAM_METRICS}, "
                         f"not {metric}")
    from metrics.utility import contains_accuracy

    streams = _stream_methods()
    # Directories a stream method can be written under: its own name, or the runner
    # whose payload variant names it (MuRAG-Ada is written by the MuRAG runner).
    dirs = streams | {d for (d, _), m in _VARIANT_METHODS.items() if m in streams}
    longest: dict[tuple, tuple[int, Path]] = {}
    for run_dir in run_dirs:
        for path in sorted(run_dir.glob("*/**/*.json")):
            rel = path.relative_to(run_dir).parts
            if rel[0] not in dirs or len(rel) < 4:
                continue
            try:
                payload = json.load(open(path, encoding="utf-8"))
            except (OSError, ValueError):
                continue
            method = method_of_payload(rel[0], payload)
            if method not in streams:
                continue
            n = len(payload.get("results") or [])
            if n == 0 or payload.get("epsilon_target") is None:
                continue
            key = (method, payload.get("dataset") or rel[-3], rel[-2],
                   float(payload.get("epsilon_target")))
            if n > longest.get(key, (0, None))[0]:
                longest[key] = (n, path)

    rows, short = [], set()
    for (method, dataset, model, eps), (n, path) in sorted(longest.items()):
        preds, refs = _payload_stream(path)
        for q in workloads:
            if q > n:
                short.add((method, dataset, n, q))
                continue
            rows.append({"method": method, "dataset": dataset, "model": model,
                         "epsilon_target": eps, "workload_queries": float(q),
                         metric: contains_accuracy(preds[:q], refs[:q])})
    for method, dataset, n, q in sorted(short):
        print(f"  [note] {method} on {dataset}: stream of {n} answers, Q={q} left out")
    return pd.DataFrame(rows, columns=CELL_COLUMNS + [metric])


def load_cells(run_dirs: list[Path], metric: str, workloads: list[int]) -> pd.DataFrame:
    """One row per (method, dataset, model, epsilon_target, Q), later dirs win ties."""
    frames = []
    for order, run_dir in enumerate(run_dirs):
        df = load_run(run_dir)
        if df.empty:
            print(f"  [skip] no result JSONs under {run_dir}")
            continue
        frames.append(df.assign(_order=order))
    if not frames:
        raise SystemExit("no result JSONs found under the run directories")
    df = pd.concat(frames, ignore_index=True)
    if metric not in df.columns:
        raise SystemExit(f"metric {metric!r} is not in the results")

    streams = _stream_methods()
    df = df[~df.method.isin(streams)]
    df = (df.sort_values("_order")
            .drop_duplicates(subset=CELL_COLUMNS, keep="last")
            [CELL_COLUMNS + [metric]])
    present_streams = stream_cells(run_dirs, workloads, metric)
    return pd.concat([df, present_streams], ignore_index=True)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def references(cells: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Floor and ceiling per corpus (mean over generators), with the gap between them."""
    ref = cells[cells.method.isin([FLOOR_METHOD, CEILING_METHOD])]
    # The reference conditions have no budget and are written once per workload size by
    # some runners: one value per generator first, so a generator is not counted twice.
    per_model = ref.groupby(["dataset", "method", "model"])[metric].mean()
    t = per_model.groupby(["dataset", "method"]).mean().unstack("method")
    t = t.rename(columns={FLOOR_METHOD: "floor", CEILING_METHOD: "ceiling"})
    for col in ("floor", "ceiling"):
        if col not in t:
            t[col] = math.nan
    t = t[["floor", "ceiling"]].dropna()
    t["gap"] = t.ceiling - t.floor
    n = per_model.groupby(["dataset", "method"]).size().unstack("method")
    uneven = n[n.nunique(axis=1) > 1]
    if not uneven.empty:
        print("  [note] floor and ceiling averaged over different generators:\n"
              f"{uneven.to_string()}")
    return t.sort_values("gap")


def method_means(cells: pd.DataFrame, metric: str) -> pd.DataFrame:
    dp = cells[cells.method.isin(METHOD_STYLE) & cells.epsilon_target.notna()]
    return (dp.groupby(["dataset", "method", "epsilon_target", "workload_queries"])
              [metric].agg(["mean", "count"]).reset_index()
              .rename(columns={"mean": metric, "count": "n_models"}))


def panel_datasets(ref: pd.DataFrame, means: pd.DataFrame, all_datasets: bool) -> list[str]:
    have = set(means.dataset)
    keep = [d for d in ref.index if d in have and (all_datasets or ref.gap[d] > 0)]
    dropped = sorted((have & set(ref.index)) - set(keep))
    if dropped:
        print(f"  [note] ceiling not above floor, left out of the method figures: {dropped}"
              " (use --all-datasets to keep them)")
    missing = sorted(have - set(ref.index))
    if missing:
        print(f"  [note] no floor/ceiling for {missing}, left out of the method figures")
    # Largest gap first: the corpora where the comparison has the most room lead.
    return sorted(keep, key=lambda d: -ref.gap[d])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _legend_handles(methods: list[str]) -> list:
    handles = [plt.Line2D([], [], label=METHOD_STYLE[key][0], **line_kwargs(key))
               for key in METHOD_STYLE if key in methods]
    handles += [plt.Line2D([], [], color=REF, lw=1.2, ls=(0, (4, 3)), label=FLOOR_LABEL),
                plt.Line2D([], [], color=INK, lw=1.2, label=CEILING_LABEL)]
    return handles


def _reference_lines(ax, floor: float, ceiling: float) -> None:
    ax.axhline(ceiling, color=INK, lw=1.2, zorder=1)
    ax.axhline(floor, color=REF, lw=1.2, ls=(0, (4, 3)), zorder=1)
    for value, colour in ((ceiling, INK), (floor, REF)):
        ax.text(1.015, value, half_up(value), va="center", ha="left", fontsize=9,
                color=colour, clip_on=False, transform=ax.get_yaxis_transform())


def _log_ticks(ax, values: list[float]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(values)
    ax.set_xticklabels([f"{v:g}" for v in values])
    ax.minorticks_off()
    ax.set_xlim(min(values) * 0.85, max(values) * 1.15)


def _grid(n: int, max_cols: int = 3) -> tuple[int, int]:
    ncols = 2 if n == 4 else min(n, max_cols)
    return math.ceil(n / ncols), ncols


def _save(fig, out_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight",
                    facecolor="white" if ext == "png" else "none")
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


def figure_reference_range(ref: pd.DataFrame, metric: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 0.5 * len(ref) + 1.2))
    floor_c, ceil_c = "#D55E00", "#0072B2"
    for i, (name, row) in enumerate(ref.iterrows()):
        gap = max(row.gap, 0.0)
        if gap > 0.005:
            ax.plot([row.floor, row.ceiling], [i, i], color="#d9d9d9", lw=11,
                    solid_capstyle="round", zorder=1)
        inside = gap > 0.12
        ax.text((row.floor + row.ceiling) / 2 if inside else max(row.floor, row.ceiling) + 0.03,
                i, half_up(gap, 3), va="center", ha="center" if inside else "left",
                fontsize=9.5, fontweight="bold", color="#404040" if inside else MUTED, zorder=4)
        # The floor is a larger hollow square, so it stays visible where the two meet.
        ax.plot(row.floor, i, "s", ms=12, mfc="white", mec=floor_c, mew=2.2, zorder=3)
        ax.plot(row.ceiling, i, "o", ms=8, color=ceil_c, mec="white", mew=1.0, zorder=4)
    ax.set_yticks(range(len(ref)), [dataset_label(d) for d in ref.index])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.7, len(ref) - 0.3)
    ax.set_xlabel(f"{_metric_label(metric)}, mean over generators")
    ax.plot([], [], "s", ms=10, mfc="white", mec=floor_c, mew=2.0, label=FLOOR_LABEL)
    ax.plot([], [], "o", color=ceil_c, label=CEILING_LABEL)
    ax.legend(loc="lower right", frameon=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c9c9")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED)
    ax.xaxis.grid(True, color="#ececec", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out_dir, "reference_range")
    ref.to_csv(out_dir / "reference_range.csv")


def figure_tradeoff(means: pd.DataFrame, ref: pd.DataFrame, datasets: list[str],
                    workload: int, metric: str, out_dir: Path) -> None:
    at_q = means[means.workload_queries == workload]
    if at_q.empty or not datasets:
        print(f"  [skip] tradeoff_q{workload}: no DP cell at Q={workload}")
        return
    eps = sorted(at_q.epsilon_target.unique())
    nrows, ncols = _grid(len(datasets))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols + 0.6, 2.9 * nrows + 1.0),
                             sharex=True, sharey=True, squeeze=False)
    for ax, name in zip(axes.flat, datasets):
        _reference_lines(ax, ref.floor[name], ref.ceiling[name])
        sub = at_q[at_q.dataset == name]
        for key in METHOD_STYLE:
            s = sub[sub.method == key].sort_values("epsilon_target")
            if not s.empty:
                ax.plot(s.epsilon_target, s[metric], **line_kwargs(key))
        ax.set_title(dataset_label(name), fontweight="bold", loc="left", pad=4)
        _log_ticks(ax, eps)
        ax.set_ylim(-0.03, 1.0)
        style_axes(ax)
        ax.tick_params(labelleft=True, labelbottom=True)
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(f"{EPSILON_AXIS}   (Q = {workload})")
    fig.supylabel(_metric_label(metric), color=INK)
    _finish(fig, sorted(at_q.method.unique()))
    _save(fig, out_dir, f"tradeoff_q{workload}")
    table = at_q[at_q.dataset.isin(datasets)].pivot_table(
        index=["dataset", "method"], columns="epsilon_target", values=metric)
    table.to_csv(out_dir / f"tradeoff_q{workload}.csv")


def figure_workload(means: pd.DataFrame, ref: pd.DataFrame, cells: pd.DataFrame,
                    panels: list[tuple[str, list[str]]], epsilons: list[float],
                    metric: str, out_dir: Path) -> None:
    sel = means[means.epsilon_target.isin(epsilons)]
    if sel.workload_queries.nunique() < 2 or not panels:
        print("  [skip] workload_axis: needs at least two workload sizes")
        return
    qs = sorted(sel.workload_queries.unique())
    fig, axes = plt.subplots(len(epsilons), len(panels),
                             figsize=(3.3 * len(panels) + 0.8, 2.9 * len(epsilons) + 1.0),
                             sharex=True, sharey=True, squeeze=False)
    rows = []
    for c, (label, names) in enumerate(panels):
        floor = ref.floor[names].mean()
        ceiling = ref.ceiling[names].mean()
        for r, eps in enumerate(epsilons):
            ax = axes[r][c]
            _reference_lines(ax, floor, ceiling)
            sub = sel[(sel.epsilon_target == eps) & sel.dataset.isin(names)]
            for key in METHOD_STYLE:
                s = sub[sub.method == key]
                # A group of corpora is averaged only where every corpus has the cell,
                # so a point is never the mean of a different set of corpora.
                counts = s.groupby("workload_queries").dataset.nunique()
                full = counts[counts == len(names)].index
                s = s[s.workload_queries.isin(full)].groupby("workload_queries")[metric].mean()
                if s.empty:
                    continue
                ax.plot(s.index, s.values, **line_kwargs(key))
                rows += [{"panel": label, "epsilon_target": eps, "method": key,
                          "workload_queries": q, metric: v} for q, v in s.items()]
            if r == 0:
                ax.set_title(label, fontweight="bold", loc="left", pad=4)
            if c == 0:
                ax.set_ylabel(f"ε = {eps:g}", fontweight="bold")
            _log_ticks(ax, qs)
            ax.set_ylim(-0.03, 1.0)
            style_axes(ax)
            ax.tick_params(labelleft=True, labelbottom=True)
    for ax in axes[-1]:
        ax.set_xlabel(WORKLOAD_AXIS)
    fig.supylabel(_metric_label(metric), color=INK)
    _finish(fig, sorted(sel.method.unique()))
    _save(fig, out_dir, "workload_axis")
    pd.DataFrame(rows).to_csv(out_dir / "workload_axis.csv", index=False)


def _finish(fig, methods: list[str]) -> None:
    handles = _legend_handles(methods)
    fig.tight_layout(rect=(0.02, 0.0, 0.96, 1.0))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.5, columnspacing=1.4)


def _metric_label(metric: str) -> str:
    return metric.replace("_", " ").capitalize()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _parse_groups(specs: list[str]) -> list[tuple[str, list[str]]]:
    groups = []
    for spec in specs:
        label, _, names = spec.partition("=")
        if not names:
            raise SystemExit(f"--group expects LABEL=dataset1,dataset2, got {spec!r}")
        groups.append((label, [n.strip() for n in names.split(",") if n.strip()]))
    return groups


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, action="append",
                     help="run directory; repeat to merge several")
    src.add_argument("--from-csv", type=Path, help="cell table written by a previous call")
    ap.add_argument("--out", type=Path, default=None, help="output directory")
    ap.add_argument("--metric", default="contains_accuracy")
    ap.add_argument("--workload", type=int, default=1,
                    help="workload size of the trade-off figure (default 1)")
    ap.add_argument("--workloads", type=int, nargs="+", default=[1, 5, 25, 100],
                    help="workload sizes read from the streams (default 1 5 25 100)")
    ap.add_argument("--epsilons", type=float, nargs="+", default=None,
                    help="budgets of the workload figure (default: the two largest)")
    ap.add_argument("--datasets", nargs="+", default=None, help="only these corpora")
    ap.add_argument("--exclude-datasets", nargs="+", default=[])
    ap.add_argument("--all-datasets", action="store_true",
                    help="keep corpora whose ceiling is not above their floor")
    ap.add_argument("--group", action="append", default=[], metavar="LABEL=D1,D2",
                    help="average these corpora into one column of the workload figure")
    args = ap.parse_args(argv)

    apply()
    if args.from_csv:
        cells = pd.read_csv(args.from_csv)
        out_dir = args.out or args.from_csv.parent
    else:
        run_dirs = [d.resolve() for d in args.run_dir]
        cells = load_cells(run_dirs, args.metric, args.workloads)
        out_dir = args.out or run_dirs[0] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        cells = cells[cells.dataset.isin(args.datasets)]
    cells = cells[~cells.dataset.isin(args.exclude_datasets)]
    if not args.from_csv:
        cells.to_csv(out_dir / "results_cells.csv", index=False)
        print(f"  wrote results_cells.csv ({len(cells)} cells)")

    present = sorted(set(cells.method) & set(METHOD_STYLE))
    absent = [m for m in METHOD_STYLE if m not in present]
    print(f"[plot_results] methods: {present}" + (f"   (absent: {absent})" if absent else ""))
    for cond in (FLOOR_METHOD, CEILING_METHOD):
        if cond not in set(cells.method):
            print(f"  [note] no {cond} cells: {REFERENCE_STYLE[cond][0]} is missing")

    ref = references(cells, args.metric)
    if ref.empty:
        raise SystemExit("the method figures need both no_rag and vanilla cells")
    means = method_means(cells, args.metric)
    datasets = panel_datasets(ref, means, args.all_datasets)

    figure_reference_range(ref, args.metric, out_dir)
    figure_tradeoff(means, ref, datasets, args.workload, args.metric, out_dir)

    grouped = _parse_groups(args.group)
    in_group = {d for _, names in grouped for d in names}
    unknown = sorted(in_group - set(datasets))
    if unknown:
        raise SystemExit(f"--group names corpora not in the method figures: {unknown}")
    panels = [(dataset_label(d), [d]) for d in datasets if d not in in_group] + grouped
    epsilons = args.epsilons or sorted(means.epsilon_target.unique())[-2:]
    figure_workload(means, ref, cells, panels, epsilons, args.metric, out_dir)
    print(f"[plot_results] done -> {out_dir}")


if __name__ == "__main__":
    main()
