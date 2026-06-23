"""Run logging for BenchDP-RAG experiments.

Every method runner tees its stdout/stderr — so prints, warnings and verbose
output land in a file as well as on the console — into a per-run log directory:

    logs/run_<YYYYmmdd_HHMMSS>/<method>_<YYYYmmdd_HHMMSS>.log

The orchestrating runner creates the run directory once and shares it with each
method subprocess via the ``BENCHDP_LOG_DIR`` environment variable; it also
writes a top-level ``summary_<stamp>.log`` next to the per-method files. Running
a method standalone creates its own ``logs/run_<stamp>/`` so it is self-contained.
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import IO

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Tee:
    """Write to several streams at once (e.g. the console and a log file)."""

    def __init__(self, *streams: IO[str]) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def run_log_dir() -> Path:
    """The per-run log directory, shared across method subprocesses.

    Uses ``BENCHDP_LOG_DIR`` when set by the orchestrating runner; otherwise
    creates a fresh ``logs/run_<timestamp>/`` so a standalone method run is
    self-contained.
    """
    shared = os.environ.get("BENCHDP_LOG_DIR")
    if shared:
        log_dir = Path(shared)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = PROJECT_ROOT / "logs" / f"run_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _tee_to(log_path: Path) -> IO[str]:
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    # Route stdlib logging and the warnings module into the same tee'd stream so
    # library warnings/log records are captured in the file too.
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    logging.captureWarnings(True)
    return log_file


def setup_method_logging(method: str) -> Path:
    """Tee a method runner's output into ``<run_log_dir>/<method>_<stamp>.log``."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = run_log_dir() / f"{method}_{stamp}.log"
    _tee_to(log_path)
    print(f"[log] {method} → {log_path}")
    return log_path


def setup_summary_logging(log_dir: Path) -> Path:
    """Tee the orchestrator's output into ``<log_dir>/summary_<stamp>.log``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"summary_{stamp}.log"
    _tee_to(log_path)
    print(f"[log] summary → {log_path}")
    return log_path
