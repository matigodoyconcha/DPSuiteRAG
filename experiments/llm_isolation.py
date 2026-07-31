"""Run each swept LLM in its own subprocess so GPU memory is fully reclaimed.

A method runner's ``main()`` sweeps every LLM in the config inside ONE process.
Even with ``free_memory()`` between LLMs, CUDA fragmentation or a lingering
reference can leave a model's VRAM unreclaimed, so a later/larger LLM OOMs.
Giving each LLM its own process guarantees 100% reclaim — the OS frees all of a
process' GPU memory when it exits — which is the same reason ``runner.py`` already
runs each *method* in its own subprocess.

Integration (uniform across the sweep runners)::

    llm_configs = _iter_llm_configs(config)
    multi_llm = len(llm_configs) > 1
    todo = resolve_llm_execution(llm_configs, method_name="dpvoterag")
    if todo is None:
        return                       # orchestrator: already fanned out to children
    for _idx, llm_cfg in todo:
        ...existing per-LLM body...

Env contract:
    ``BENCHDP_ONLY_LLM_INDEX``    set by the orchestrator on each child; the child
                                  runs only that one LLM (in-process).
    ``BENCHDP_NO_LLM_ISOLATION=1``  run every LLM in-process (old behaviour; escape
                                  hatch for debugging or single-process profiling).
    ``BENCHDP_METHOD_LOG``        the method's log-file path, set by
                                  :func:`run_logging.setup_method_logging` before
                                  the fan-out below and carried to each child by the
                                  ``os.environ.copy()`` here, so all LLMs of a method
                                  append to ONE log file instead of one file each.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from experiments.run_logging import log


def resolve_llm_execution(
    llm_configs: list[dict[str, Any]], *, method_name: str
) -> list[tuple[int, dict[str, Any]]] | None:
    """Decide which LLMs THIS process runs; fan out to per-LLM children if orchestrating.

    Returns the list of ``(index, llm_cfg)`` for the caller to process in-process,
    or ``None`` when this process was the orchestrator and has already run every
    LLM in its own subprocess (the caller should then ``return`` immediately).

    A child's OOM/crash must NOT abort the remaining LLMs (mirrors the in-process
    OOM-skip), so per-child failures are logged and the sweep continues; each next
    LLM still gets a clean process.
    """
    only = os.environ.get("BENCHDP_ONLY_LLM_INDEX")
    if only is not None:
        # Worker child: run exactly the one LLM the orchestrator assigned.
        i = int(only)
        return [(i, llm_configs[i])]

    if len(llm_configs) <= 1 or os.environ.get("BENCHDP_NO_LLM_ISOLATION") == "1":
        # Nothing to isolate (single LLM) or isolation disabled: old in-process loop.
        return list(enumerate(llm_configs))

    # Orchestrator: one fresh subprocess per LLM.
    n = len(llm_configs)
    script = os.path.abspath(sys.argv[0])
    failures = 0
    for i, llm_cfg in enumerate(llm_configs):
        name = str(llm_cfg.get("model_name", f"llm[{i}]"))
        env = os.environ.copy()
        env["BENCHDP_ONLY_LLM_INDEX"] = str(i)
        log(f"[{method_name}] LLM {i + 1}/{n} START (isolated subprocess): {name}")
        rc = subprocess.run([sys.executable, script, *sys.argv[1:]], env=env).returncode
        if rc == 0:
            log(f"[{method_name}] LLM {i + 1}/{n} END   : {name} — OK")
        else:
            # Keep going: the next LLM gets its own clean process regardless.
            print(
                f"[WARN] LLM subprocess failed (rc={rc}) for {name} — "
                f"skipping, continuing the sweep."
            )
            log(f"[{method_name}] LLM {i + 1}/{n} END   : {name} — FAILED (rc={rc})")
            failures += 1

    if failures == n:
        # Every child died, so this method produced nothing. Returning normally made
        # the parent exit 0 and the orchestrator record the cell as OK, so a sweep
        # where all seven generators OOM'd looked like a successful run with no
        # results in it.
        raise RuntimeError(
            f"all {n} LLM subprocesses failed for {method_name}; no results were "
            "produced (see the [WARN] lines above for each one)"
        )
    if failures:
        print(
            f"[WARN] {failures} of {n} LLM subprocesses failed for {method_name}; "
            "the results directory is incomplete."
        )
    return None
