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
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

DEFAULT_METHODS = ["dp_rag", "dpvoterag", "dp_synrag"]


def _hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def run_method(name: str, script: str, config: str) -> dict:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / script),
        "--config",
        str(PROJECT_ROOT / config),
    ]

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  METHOD : {name}")
    print(f"  CONFIG : {config}")
    print(f"  CMD    : {' '.join(cmd)}")
    print(f"{sep}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
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

    print("\n" + "=" * 64)
    print("  BenchDP-RAG Runner")
    print(f"  Mode   : {'QUICK (smoke test)' if args.quick else 'FULL'}")
    print(f"  Methods: {args.methods}")
    print("=" * 64)

    wall_start = time.time()
    rows: list[dict] = []

    for name in args.methods:
        info = METHODS[name]
        config = info[config_key]
        row = run_method(name, info["script"], config)
        rows.append(row)

    wall_total = time.time() - wall_start

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
    print("=" * 64 + "\n")

    failed = [r["method"] for r in rows if not r["ok"]]
    if failed:
        print(f"FAILED methods: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
