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
experiments/configs/dp_rag.yaml for an example.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))
from experiments.run_logging import log, setup_summary_logging  # noqa: E402

METHODS: dict[str, dict[str, str]] = {
    "dp_rag": {
        "script": "experiments/run_dp_rag.py",
        "paper_faithful_config": "experiments/configs/dp_rag_paper_faithful.yaml",
        "full_config": "experiments/configs/dp_rag.yaml",
    },
    "dpvoterag": {
        "script": "experiments/run_dpvoterag.py",
        "paper_faithful_config": "experiments/configs/dpvoterag_paper_faithful.yaml",
        "full_config": "experiments/configs/dpvoterag.yaml",
    },
    "dpsparsevoterag": {
        "script": "experiments/run_dpsparsevoterag.py",
        "paper_faithful_config": "experiments/configs/dpsparsevoterag_paper_faithful.yaml",
        "full_config": "experiments/configs/dpsparsevoterag.yaml",
    },
    "dp_synrag": {
        "script": "experiments/run_dp_synrag.py",
        "paper_faithful_config": "experiments/configs/dp_synrag_paper_faithful.yaml",
        "full_config": "experiments/configs/dp_synrag.yaml",
    },
    "dpksa": {
        "script": "experiments/run_dpksa.py",
        "paper_faithful_config": "experiments/configs/dpksa_paper_faithful.yaml",
        "full_config": "experiments/configs/dpksa.yaml",
    },
    "private_rag": {
        "script": "experiments/run_private_rag.py",
        "paper_faithful_config": "experiments/configs/private_rag_paper_faithful.yaml",
        "full_config": "experiments/configs/private_rag.yaml",
    },
    "private_rag_ada": {
        "script": "experiments/run_private_rag.py",
        "paper_faithful_config": "experiments/configs/private_rag_ada_paper_faithful.yaml",
        "full_config": "experiments/configs/private_rag_ada.yaml",
    },
}

DEFAULT_METHODS = [
    "dp_rag",
    "dpvoterag",
    "dpsparsevoterag",
    "dp_synrag",
    "dpksa",
    "private_rag",
    "private_rag_ada",
]


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
    if output_dir is not None:
        env["BENCHDP_OUTPUT_DIR"] = output_dir
    # Share the run's log directory so each method tees its own <method>_<stamp>.log
    # alongside the orchestrator's summary log.
    if log_dir is not None:
        env["BENCHDP_LOG_DIR"] = log_dir
    # Defer metric computation: method runners only save raw predictions/references;
    # score_results.py computes all metrics afterwards in a clean process. This keeps
    # the model-based metrics (BERTScore/perplexity) from loading on top of the
    # method's generator (avoids OOM).
    env["BENCHDP_DEFER_METRICS"] = "1"

    sep = "=" * 64
    print(f"\n{sep}")
    log(f"METHOD START : {name}")
    print(f"  CONFIG : {config}")
    if output_dir is not None:
        print(f"  OUTPUT : {output_dir}")
    print(f"  CMD    : {' '.join(cmd)}")
    print(f"{sep}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    log(f"METHOD END   : {name} — {status} — {_hms(elapsed)}")

    # Surface this method's errors/warnings/skips (and, on failure, its crash tail)
    # into the general summary log too — not only the per-method log file.
    _report_method_problems(name, log_dir, t0, ok=result.returncode == 0)

    return {
        "method": name,
        "config": config,
        "returncode": result.returncode,
        "elapsed": elapsed,
        "ok": result.returncode == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BenchDP-RAG orchestrating runner — runs methods sequentially."
    )
    parser.add_argument(
        "--paper-faithful",
        action="store_true",
        help=(
            "Use *_paper_faithful.yaml configs (each method's paper LLM(s) + "
            "datasets only). Default is the full ≤4B × 6-dataset sweep."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS.keys()),
        default=DEFAULT_METHODS,
        metavar="METHOD",
        help=(
            f"Methods to run (default: {DEFAULT_METHODS}). "
            f"Choices: {list(METHODS.keys())}"
        ),
    )
    args = parser.parse_args()

    config_key = "paper_faithful_config" if args.paper_faithful else "full_config"

    # One timestamped directory per run so each invocation's results are isolated:
    # results/run_<YYYYmmdd_HHMMSS>/<method>/<dataset>/...
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = PROJECT_ROOT / "results" / f"run_{run_stamp}"

    # One log directory per run, mirroring the results layout; the orchestrator
    # tees its own output into summary_<stamp>.log and shares the directory with
    # every method subprocess.
    log_root = PROJECT_ROOT / "logs" / f"run_{run_stamp}"
    setup_summary_logging(log_root)

    print("\n" + "=" * 64)
    log("BenchDP-RAG Runner START")
    print(f"  Mode   : {'PAPER-FAITHFUL' if args.paper_faithful else 'FULL'}")
    print(f"  Methods: {args.methods}")
    print(f"  Results: {run_root}")
    print("=" * 64)

    wall_start = time.time()
    rows: list[dict] = []

    for name in args.methods:
        info = METHODS[name]
        config = info[config_key]
        row = run_method(
            name,
            info["script"],
            config,
            output_dir=str(run_root / name),
            log_dir=str(log_root),
        )
        rows.append(row)

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
    header = f"  {'Method':<20} {'Config':<30} {'Time':>10}  Status"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        cfg_short = Path(row["config"]).name
        status = "OK" if row["ok"] else "FAILED"
        print(f"  {row['method']:<20} {cfg_short:<30} {_hms(row['elapsed']):>10}  {status}")
    print("  " + "-" * (len(header) - 2))
    print(f"  {'TOTAL':<20} {'':<30} {_hms(wall_total):>10}")
    print("=" * 64)
    print(f"  Results saved under: {run_root}")
    print(f"  Logs saved under   : {log_root}")
    print("=" * 64 + "\n")

    failed = [r["method"] for r in rows if not r["ok"]]
    if failed:
        print(f"FAILED methods: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
