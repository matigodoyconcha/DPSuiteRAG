"""Orchestrating runner for BenchDP-RAG experiments.

Runs one or more DP methods sequentially via subprocess (each method gets a
fresh process so GPU memory is fully released between methods).

Usage:
    # Quick smoke test (default: all 3 methods, *_quick.yaml configs)
    python experiments/runner.py --quick

    # Only specific methods
    python experiments/runner.py --quick --methods dp_rag dp_synrag

    # Full configs
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
from experiments.run_logging import setup_summary_logging  # noqa: E402

METHODS: dict[str, dict[str, str]] = {
    "dp_rag": {
        "script": "experiments/run_dp_rag.py",
        "quick_config": "experiments/configs/dp_rag_quick.yaml",
        "full_config": "experiments/configs/dp_rag.yaml",
    },
    "dpvoterag": {
        "script": "experiments/run_dpvoterag.py",
        "quick_config": "experiments/configs/dpvoterag_quick.yaml",
        "full_config": "experiments/configs/dpvoterag.yaml",
    },
    "dpsparsevoterag": {
        "script": "experiments/run_dpsparsevoterag.py",
        "quick_config": "experiments/configs/dpsparsevoterag_quick.yaml",
        "full_config": "experiments/configs/dpsparsevoterag.yaml",
    },
    "dp_synrag": {
        "script": "experiments/run_dp_synrag.py",
        "quick_config": "experiments/configs/dp_synrag_quick.yaml",
        "full_config": "experiments/configs/dp_synrag.yaml",
    },
    "dpksa": {
        "script": "experiments/run_dpksa.py",
        "quick_config": "experiments/configs/dpksa_quick.yaml",
        "full_config": "experiments/configs/dpksa.yaml",
    },
}

DEFAULT_METHODS = ["dp_rag", "dpvoterag", "dpsparsevoterag", "dp_synrag", "dpksa"]


def _hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


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
    print(f"  METHOD : {name}")
    print(f"  CONFIG : {config}")
    if output_dir is not None:
        print(f"  OUTPUT : {output_dir}")
    print(f"  CMD    : {' '.join(cmd)}")
    print(f"{sep}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n[{name}] {status} — {_hms(elapsed)}")

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
        "--quick",
        action="store_true",
        help="Use *_quick.yaml configs (smoke test).",
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

    config_key = "quick_config" if args.quick else "full_config"

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
    print("  BenchDP-RAG Runner")
    print(f"  Mode   : {'QUICK (smoke test)' if args.quick else 'FULL'}")
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
    print("  SCORING (offline metrics)")
    print("=" * 64)
    score_cmd = [sys.executable, str(PROJECT_ROOT / "experiments" / "score_results.py"), str(run_root)]
    score_rc = subprocess.run(score_cmd, cwd=str(PROJECT_ROOT)).returncode
    if score_rc != 0:
        print(f"  WARNING: scoring failed (rc={score_rc}); raw predictions are saved, re-run "
              f"`python experiments/score_results.py {run_root}` to compute metrics.")

    print("\n" + "=" * 64)
    print("  SUMMARY")
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
