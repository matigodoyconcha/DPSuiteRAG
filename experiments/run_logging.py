"""Run logging for BenchDP-RAG experiments.

Every method runner tees its stdout/stderr — so prints, warnings and verbose
output land in a file as well as on the console — into a per-run log directory:

    logs/run_<YYYYmmdd_HHMMSS>/<method>_<YYYYmmdd_HHMMSS>.log

The orchestrating runner creates the run directory once and shares it with each
method subprocess via the ``BENCHDP_LOG_DIR`` environment variable; it also
writes a top-level ``summary_<stamp>.log`` next to the per-method files. Running
a method standalone creates its own ``logs/run_<stamp>/`` so it is self-contained.

A method that sweeps several LLMs runs each in its own isolation subprocess
(:mod:`llm_isolation`); they all append to the SAME ``<method>_<stamp>.log`` via
the ``BENCHDP_METHOD_LOG`` variable (see :func:`setup_method_logging`), so a
multi-LLM method produces one log file, not one per LLM.
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

    def isatty(self) -> bool:
        # A multiplexed stream is not a terminal. Returning False keeps libraries
        # that probe ``sys.stdout.isatty()`` (e.g. transformers' colored model-load
        # report) working and keeps ANSI escapes out of the log file. Without this
        # method those libraries raise AttributeError and abort the whole run.
        return False


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


# Milestone logger. Routing milestones through the stdlib logger (instead of raw
# print) is precisely what gives them a wall-clock timestamp: the Formatter
# configured in _tee_to prepends %(asctime)s. A bare print() is NOT timestamped —
# the _Tee only duplicates the stream, it does not format it. Per-token verbose
# output stays on print() by design (one timestamp per token would be noise).
_LOGGER = logging.getLogger("benchdp")

# Full local datetime, e.g. "[2026-06-26 14:23:07] ...". Local time so it matches
# the run_<YYYYmmdd_HHMMSS> directory stamps and the operator's wall clock.
_LOG_FORMAT = "[%(asctime)s] %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def log(message: str) -> None:
    """Emit a timestamped milestone line (method/dataset/epsilon start & end,
    scoring, per-query heartbeats, run summaries).

    Goes through the stdlib logger so the Formatter adds the wall-clock timestamp
    automatically; the tee then mirrors it to both console and the log file. Use
    this for the handful of important events you want to locate in time — keep
    high-frequency per-token mechanism traces on plain ``print``.
    """
    _LOGGER.info(message)


def _tee_to(log_path: Path) -> IO[str]:
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    # Route stdlib logging and the warnings module into the same tee'd stream so
    # library warnings/log records are captured in the file too, and give every
    # record a wall-clock timestamp via the formatter. Sending a message through
    # this logger (see log()) is the only way print()-style output gets a stamp.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATEFMT,
        force=True,
    )
    logging.captureWarnings(True)
    return log_file


def setup_method_logging(method: str) -> Path:
    """Tee a method runner's output into ``<run_log_dir>/<method>_<stamp>.log``.

    All of a method's per-LLM isolation subprocesses (see :mod:`llm_isolation`)
    share ONE log file instead of each creating its own. The first process to run
    — the method's LLM orchestrator — picks the ``<method>_<stamp>.log`` path and
    exports it as ``BENCHDP_METHOD_LOG``; because ``resolve_llm_execution`` fans
    out with ``os.environ.copy()``, every per-LLM child inherits that path and
    appends to the same file (the children run sequentially, so appends never
    interleave). Each *method* still gets its own file — runner.py launches every
    method in a fresh process without the variable set, so e.g. private_rag and
    private_rag_ada stay separate via their differing start-time stamps.
    """
    shared = os.environ.get("BENCHDP_METHOD_LOG")
    if shared:
        log_path = Path(shared)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include the dataset the orchestrator restricted this run to, so each
        # per-(method, dataset) invocation is identifiable by its filename.
        dataset = os.environ.get("BENCHDP_ONLY_DATASET")
        base = f"{method}_{dataset}_{stamp}" if dataset else f"{method}_{stamp}"
        log_path = run_log_dir() / f"{base}.log"
        # Exported so the per-LLM children (spawned via os.environ.copy) reuse it.
        os.environ["BENCHDP_METHOD_LOG"] = str(log_path)
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
