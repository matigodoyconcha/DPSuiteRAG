"""Orchestrating runner for BenchDP-RAG experiments.

Runs one or more DP methods sequentially via subprocess (each method gets a
fresh process so GPU memory is fully released between methods).

Usage:
    # Paper-faithful configs (each method's paper LLM(s) + datasets only)
    python experiments/runner.py --paper-faithful

    # Only specific methods
    python experiments/runner.py --paper-faithful --methods dp_rag dp_synrag

    # Full configs (the broad ≤4B-LLM × 6-dataset sweep; this is the default)
    python experiments/runner.py --methods dp_rag dpvoterag dp_synrag

Sweeping multiple LLMs is configured per-method in the YAML configs, the same
way datasets are: add a top-level ``llms:`` list (each entry merged over the
singular ``llm:`` block) and every method runner repeats its grid for each LLM,
writing results under ``<output_dir>/<dataset>/<model_label>/``. See
experiments/configs/full/dp_rag.yaml for an example.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))
from bench_datasets.dpr_corpus import corpus_available  # noqa: E402
from experiments.common import dataset_names  # noqa: E402
from experiments.run_logging import log, setup_summary_logging  # noqa: E402

METHODS: dict[str, dict[str, str]] = {
    "vanilla": {
        "script": "experiments/run_vanilla.py",
        "paper_faithful_config": "experiments/configs/full/vanilla.yaml",
        "full_config": "experiments/configs/full/vanilla.yaml",
    },
    "dp_rag": {
        "script": "experiments/run_dp_rag.py",
        "paper_faithful_config": "experiments/configs/paper_faithful/dp_rag.yaml",
        "full_config": "experiments/configs/full/dp_rag.yaml",
    },
    "dpvoterag": {
        "script": "experiments/run_dpvoterag.py",
        "paper_faithful_config": "experiments/configs/paper_faithful/dpvoterag.yaml",
        "full_config": "experiments/configs/full/dpvoterag.yaml",
    },
    "dpsparsevoterag": {
        "script": "experiments/run_dpsparsevoterag.py",
        "paper_faithful_config": "experiments/configs/paper_faithful/dpsparsevoterag.yaml",
        "full_config": "experiments/configs/full/dpsparsevoterag.yaml",
    },
    "dp_synrag": {
        "script": "experiments/run_dp_synrag.py",
        "paper_faithful_config": "experiments/configs/paper_faithful/dp_synrag.yaml",
        "full_config": "experiments/configs/full/dp_synrag.yaml",
    },
    "dpksa": {
        "script": "experiments/run_dpksa.py",
        "paper_faithful_config": "experiments/configs/paper_faithful/dpksa.yaml",
        "full_config": "experiments/configs/full/dpksa.yaml",
    },
    "private_rag": {
        "script": "experiments/run_private_rag.py",
        "paper_faithful_config": "experiments/configs/paper_faithful/private_rag.yaml",
        "full_config": "experiments/configs/full/private_rag.yaml",
    },
    "private_rag_ada": {
        "script": "experiments/run_private_rag.py",
        "paper_faithful_config": "experiments/configs/paper_faithful/private_rag_ada.yaml",
        "full_config": "experiments/configs/full/private_rag_ada.yaml",
    },
}

# vanilla leads the sweep: it is the non-private reference condition every DP curve
# is read against, and it is the cheapest run, so a broken cell surfaces immediately
# rather than after the DP methods have spent the GPU.
DEFAULT_METHODS = [
    "vanilla",
    "dp_rag",
    "dpvoterag",
    "dpsparsevoterag",
    "dp_synrag",
    "dpksa",
    "private_rag",
    "private_rag_ada",
]

# The standalone MIA run uses the same set: vanilla is there as the empirical
# leakage ceiling / protection floor.
MIA_DEFAULT_METHODS = list(DEFAULT_METHODS)


def _hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# Substrings that mark an error/warning/skip worth surfacing in the general log.
_PROBLEM_MARKERS = (
    "[WARN]",
    "OOM",
    "does not fit",
    "SKIPPED",
    "Traceback (most recent call last):",
)


def _method_log_files(log_dir: str | None, since: float) -> list[Path]:
    """The per-method (and per-LLM child) log files written during this method.

    Methods run sequentially into the shared run log dir, so selecting by mtime
    window avoids depending on the log filename (private_rag_ada logs as
    ``private_rag``) and also picks up the per-LLM isolation subprocess logs.
    """
    if not log_dir:
        return []
    d = Path(log_dir)
    if not d.is_dir():
        return []
    files: list[Path] = []
    for p in d.glob("*.log"):
        if p.name.startswith("summary_"):
            continue
        try:
            if p.stat().st_mtime >= since - 2:
                files.append(p)
        except OSError:
            continue
    return sorted(files, key=lambda p: p.stat().st_mtime)


def _grep_problems(paths: list[Path], max_lines: int = 40) -> list[str]:
    """Deduplicated lines matching :data:`_PROBLEM_MARKERS`, streamed (bounded)."""
    hits: list[str] = []
    seen: set[str] = set()
    for p in paths:
        try:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if any(m in line for m in _PROBLEM_MARKERS):
                        s = line.strip()
                        if s and s not in seen:
                            seen.add(s)
                            hits.append(s)
                            if len(hits) >= max_lines:
                                return hits
        except OSError:
            continue
    return hits


def _tail(path: Path, n: int = 60, max_bytes: int = 262144) -> list[str]:
    """Last ``n`` lines of ``path`` (reads only the trailing ``max_bytes``)."""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read()
    except OSError:
        return []
    return data.decode("utf-8", errors="replace").splitlines()[-n:]


def _report_method_problems(name: str, log_dir: str | None, since: float, ok: bool) -> None:
    """Echo a method's warnings/skips (and, on failure, its crash tail) into the
    general summary log, so problems are visible without opening the method log."""
    files = _method_log_files(log_dir, since)
    if not files:
        return
    problems = _grep_problems(files)
    if problems:
        log(f"PROBLEMS in {name} (warnings / skips / errors — see per-method log):")
        for line in problems:
            print(f"    {line}")
    if not ok:
        newest = files[-1]
        log(f"FAILURE tail — {name} ({newest.name}):")
        for line in _tail(newest):
            print(f"    {line}")


def run_method(
    name: str,
    script: str,
    config: str,
    output_dir: str | None = None,
    log_dir: str | None = None,
    only_dataset: str | None = None,
    index_manifest: str | None = None,
    mia_only: bool = False,
    run_seed: int | None = None,
) -> dict:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / script),
        "--config",
        str(PROJECT_ROOT / config),
    ]

    # Each method runs in a fresh subprocess; pass the per-run output directory via
    # BENCHDP_OUTPUT_DIR, which the method runners apply as an output_dir override.
    env = os.environ.copy()
    # Standalone MIA run: the runner indexes only a member subset and issues the
    # attack probes instead of the utility loop (see experiments/mia_probe.py).
    if mia_only:
        env["BENCHDP_MIA_ONLY"] = "1"
    if output_dir is not None:
        env["BENCHDP_OUTPUT_DIR"] = output_dir
    # Vary only the mechanisms' noise across repetitions; the corpus and the questions
    # follow data_seed, so the index is built once and reused by every seed.
    if run_seed is not None:
        env["BENCHDP_RUN_SEED"] = str(run_seed)
    # Share the run's log directory so each method tees its own <method>_<stamp>.log
    # alongside the orchestrator's summary log.
    if log_dir is not None:
        env["BENCHDP_LOG_DIR"] = log_dir
    # Defer metric computation: method runners only save raw predictions/references;
    # score_results.py computes all metrics afterwards in a clean process. This keeps
    # the model-based metrics (BERTScore/perplexity) from loading on top of the
    # method's generator (avoids OOM).
    env["BENCHDP_DEFER_METRICS"] = "1"
    # Restrict this invocation to one dataset and record any index it builds in the
    # orchestrator's per-dataset manifest, so the index is built once, shared by
    # every method/LLM for that dataset, and deleted before the next dataset.
    if only_dataset is not None:
        env["BENCHDP_ONLY_DATASET"] = only_dataset
    if index_manifest is not None:
        env["BENCHDP_INDEX_MANIFEST"] = index_manifest

    label = f"{name} [{only_dataset}]" if only_dataset else name
    sep = "=" * 64
    print(f"\n{sep}")
    log(f"METHOD START : {label}")
    print(f"  CONFIG : {config}")
    if output_dir is not None:
        print(f"  OUTPUT : {output_dir}")
    print(f"  CMD    : {' '.join(cmd)}")
    print(f"{sep}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    log(f"METHOD END   : {label} — {status} — {_hms(elapsed)}")

    # Surface this method's errors/warnings/skips (and, on failure, its crash tail)
    # into the general summary log too — not only the per-method log file.
    _report_method_problems(label, log_dir, t0, ok=result.returncode == 0)

    return {
        "method": name,
        "dataset": only_dataset,
        "config": config,
        "returncode": result.returncode,
        "elapsed": elapsed,
        "ok": result.returncode == 0,
    }


def _cleanup_indices(manifest_path: str) -> None:
    """Delete every index directory listed in *manifest_path*, then the manifest.

    Run after all of a dataset's methods finish, so at most one dataset's indices
    occupy disk at a time. Missing / already-removed dirs are ignored and
    duplicate lines are deduplicated.
    """
    try:
        with open(manifest_path, encoding="utf-8") as f:
            dirs = {line.strip() for line in f if line.strip()}
    except OSError:
        dirs = set()
    for d in sorted(dirs):
        shutil.rmtree(d, ignore_errors=True)
        log(f"  removed index {d}")
    try:
        os.remove(manifest_path)
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BenchDP-RAG orchestrating runner — runs methods sequentially."
    )
    parser.add_argument(
        "--paper-faithful",
        action="store_true",
        help=(
            "Use paper_faithful/ configs (each method's paper LLM(s) + "
            "datasets only). Default is the full ≤4B × 6-dataset sweep."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS.keys()),
        default=None,
        metavar="METHOD",
        help=(
            f"Methods to run (default: {DEFAULT_METHODS}; the MIA run adds vanilla). "
            f"Choices: {list(METHODS.keys())}"
        ),
    )
    parser.add_argument(
        "--mia-only",
        action="store_true",
        help=(
            "Run the standalone membership inference attack (S2MIA + Anderson Yes/No) "
            "instead of the utility benchmark: each method indexes only a member "
            "subset of the corpus and is probed on members vs held-out non-members. "
            "Results are written under results/mia_run_<stamp>/."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help=(
            "Repeat every cell once per seed, varying only the DP mechanisms' noise. "
            "The corpus and the test questions stay fixed (they follow data_seed), so "
            "the index is built once and the spread across seeds measures the "
            "mechanism rather than the sample. Results land in <method>/seed_<n>/."
        ),
    )
    args = parser.parse_args()
    if args.methods is None:
        args.methods = MIA_DEFAULT_METHODS if args.mia_only else DEFAULT_METHODS

    config_key = "paper_faithful_config" if args.paper_faithful else "full_config"

    # `None` means "one run, seeded by the config" — the layout stays flat, exactly as
    # it was before repetitions existed.
    seeds: list[int | None] = list(args.seeds) if args.seeds else [None]

    # One timestamped directory per run so each invocation's results are isolated.
    # The MIA run uses a separate root so it never overwrites a utility run.
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = "mia_run" if args.mia_only else "run"
    run_root = PROJECT_ROOT / "results" / f"{run_prefix}_{run_stamp}"
    # Scope this run's cached indices so a concurrent run with identical settings
    # never shares — and therefore never deletes — them (subprocesses read it via
    # experiments.index_cache.index_cache_dir).
    os.environ["BENCHDP_RUN_ID"] = run_stamp

    # One log directory per run, mirroring the results layout; the orchestrator
    # tees its own output into summary_<stamp>.log and shares the directory with
    # every method subprocess.
    log_root = PROJECT_ROOT / "logs" / f"run_{run_stamp}"
    setup_summary_logging(log_root)

    print("\n" + "=" * 64)
    log("BenchDP-RAG Runner START")
    mode = "PAPER-FAITHFUL" if args.paper_faithful else "FULL"
    if args.mia_only:
        mode += " · MIA-ONLY (S2MIA + Anderson Yes/No)"
    print(f"  Mode   : {mode}")
    print(f"  Methods: {args.methods}")
    print(f"  Results: {run_root}")
    print("=" * 64)

    # Load each selected method's config so the run can iterate DATASET-major:
    # build a dataset's index once, reuse it across every method and LLM, then
    # delete it before the next dataset (NQ-scale indices are tens of GB).
    method_cfgs: dict[str, dict] = {}
    for name in args.methods:
        cfg_path = PROJECT_ROOT / METHODS[name][config_key]
        with open(cfg_path, encoding="utf-8") as f:
            method_cfgs[name] = yaml.safe_load(f) or {}

    # Ordered unique datasets across the selected methods, and the methods that
    # evaluate each one (kept in the requested method order).
    dataset_order: list[str] = []
    methods_for: dict[str, list[str]] = {}
    unscheduled: list[str] = []  # methods whose config lists no named dataset
    for name in args.methods:
        names = dataset_names(method_cfgs[name])
        if not names:
            unscheduled.append(name)
            continue
        for d in names:
            if d not in methods_for:
                dataset_order.append(d)
                methods_for[d] = []
            if name not in methods_for[d]:
                methods_for[d].append(name)

    log(f"Datasets ({len(dataset_order)}): {dataset_order}")
    for d in dataset_order:
        print(f"    {d}: {methods_for[d]}")
    if unscheduled:
        log(f"Methods without a dataset list (run once, unrestricted): {unscheduled}")

    # Group datasets that share an on-disk index so it survives until every dataset
    # in the group has run. Natural Questions and TriviaQA retrieve from the same
    # DPR corpus, so when it is present they build one ~34 GB index and reuse it
    # instead of building it twice; every other dataset stays its own singleton
    # group. At most one group's indices ever live on disk (freed together after
    # the group finishes), preserving the dataset-major memory bound.
    # In MIA mode every dataset indexes its own small member subset (the index
    # cache is bypassed), so the DPR-corpus sharing optimisation does not apply.
    shared_corpus = {"natural_questions", "triviaqa"}
    merge_shared = (
        not args.mia_only and shared_corpus.issubset(dataset_order) and corpus_available()
    )
    groups: list[list[str]] = []
    merged_placed = False
    for dataset in dataset_order:
        if merge_shared and dataset in shared_corpus:
            if not merged_placed:
                # Place the merged group where its first member appears and process
                # its members adjacently, so the shared index is built exactly once.
                groups.append([d for d in dataset_order if d in shared_corpus])
                merged_placed = True
            continue  # the other member is folded into the merged group above
        groups.append([dataset])

    wall_start = time.time()
    rows: list[dict] = []

    for group in groups:
        # A fresh manifest per group: every index built while running the group's
        # datasets is recorded here and deleted once they all finish, so datasets
        # sharing a corpus keep their common index alive across the whole group.
        fd, manifest_path = tempfile.mkstemp(prefix=f"benchdp_idx_{group[0]}_", suffix=".manifest")
        os.close(fd)
        try:
            for dataset in group:
                print("\n" + "#" * 64)
                log(f"DATASET START : {dataset} — methods: {methods_for[dataset]}")
                print("#" * 64)
                for name in methods_for[dataset]:
                    info = METHODS[name]
                    for run_seed in seeds:
                        # Seeds sit under the method so every repetition of a cell
                        # stays together and the index is shared across all of them.
                        out = run_root / name
                        if run_seed is not None:
                            out = out / f"seed_{run_seed}"
                        rows.append(run_method(
                            name,
                            info["script"],
                            info[config_key],
                            output_dir=str(out),
                            log_dir=str(log_root),
                            only_dataset=dataset,
                            index_manifest=manifest_path,
                            mia_only=args.mia_only,
                            run_seed=run_seed,
                        ))
                log(f"DATASET END   : {dataset}")
        finally:
            # Always free the group's indices — including on an interrupt or a crash
            # mid-group — so a killed run never orphans tens of GB in tmp.
            _cleanup_indices(manifest_path)
            log(f"GROUP END     : freed indices for {group}")

    for name in unscheduled:
        info = METHODS[name]
        for run_seed in seeds:
            # Repeated on the same seed grid as the scheduled methods; running these
            # once would leave part of the run without repetitions and mix two output
            # layouts under one root.
            out = run_root / name
            if run_seed is not None:
                out = out / f"seed_{run_seed}"
            rows.append(run_method(
                name,
                info["script"],
                info[config_key],
                output_dir=str(out),
                log_dir=str(log_root),
                mia_only=args.mia_only,
                run_seed=run_seed,
            ))

    wall_total = time.time() - wall_start

    # Score all results offline, in this fresh process (no generator loaded), so
    # the model-based metrics never stack on top of a method's generator.
    print("\n" + "=" * 64)
    log("SCORING START (offline metrics)")
    print("=" * 64)
    score_cmd = [sys.executable, str(PROJECT_ROOT / "experiments" / "score_results.py"), str(run_root)]
    score_rc = subprocess.run(score_cmd, cwd=str(PROJECT_ROOT)).returncode
    log(f"SCORING END — {'OK' if score_rc == 0 else f'FAILED (rc={score_rc})'}")
    if score_rc != 0:
        print(f"  WARNING: scoring failed (rc={score_rc}); raw predictions are saved, re-run "
              f"`python experiments/score_results.py {run_root}` to compute metrics.")

    print("\n" + "=" * 64)
    log(f"SUMMARY — wall time {_hms(wall_total)}")
    print("=" * 64)
    header = f"  {'Dataset':<24} {'Method':<18} {'Time':>10}  Status"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        status = "OK" if row["ok"] else "FAILED"
        ds = row.get("dataset") or "(all)"
        print(f"  {ds:<24} {row['method']:<18} {_hms(row['elapsed']):>10}  {status}")
    print("  " + "-" * (len(header) - 2))
    print(f"  {'TOTAL':<24} {'':<18} {_hms(wall_total):>10}")
    print("=" * 64)
    print(f"  Results saved under: {run_root}")
    print(f"  Logs saved under   : {log_root}")
    print("=" * 64 + "\n")

    failed = [f"{r['method']}[{r.get('dataset') or 'all'}]" for r in rows if not r["ok"]]
    if failed:
        print(f"FAILED runs: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
