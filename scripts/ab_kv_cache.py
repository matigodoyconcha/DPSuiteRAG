#!/usr/bin/env python
"""Answer one question before the fleet runs: does the KV cache change the science?

The per-token generation loops re-encode the whole prompt at every step. Caching the
attention keys and values instead is the same computation on paper, but not the same
computation in floating point: the reduction order changes, logits move by ~1e-6, and
these methods take an argmax over a noisy vote histogram. A pair of candidates within
1e-6 of each other can therefore swap, the answer text changes, and everything
downstream of it changes with it. Bit-identical output is not on offer, so "identical"
is the wrong question. The question is whether what moves is confined to near-ties and
is unbiased -- noise -- or whether it points somewhere, which would mean the two paths
are not computing the same thing.

    python scripts/ab_kv_cache.py --dry-run          # print the plan, run nothing
    nohup python scripts/ab_kv_cache.py > ab.log 2>&1 &

The harness runs the same cells twice -- same config, same ``data_seed``, same
``run_seed`` -- once on the cached path and once with ``BENCHDP_DISABLE_KV_CACHE=1``
forcing the old one, then compares the two result trees.


The verdict
-----------

Three families of check, in the order they are allowed to matter.

**1. Epsilon is exact or it is a bug.** The budget a method is charged is a function of
its config and its accounting, and of nothing a generator computes. ``epsilon``,
``epsilon_workload``, ``epsilon_workload_basic``, ``epsilon_workload_by_q``,
``workload_queries``, ``workload_regime`` must be equal to the last bit. A cache that
moves any of them has reached something it must never reach, and no amount of utility
agreement redeems that. Any difference fails the run outright.

``epsilon_workload_spent`` and ``budget_utilisation`` are deliberately NOT in that set:
they count releases actually made, so a single flipped argmax that ends a generation one
token earlier moves them legitimately. They are reported, not gated.

**2. What moved must be unbiased.** This is the check that carries the argument, and it
is paired: both arms answer the same questions in the same order, so the comparison is
per query rather than between two means. Among the queries whose answer text changed,
per-query correctness is compared with an exact two-sided McNemar (binomial on the
discordant pairs). Under the null -- the cache only resolves near-ties, and a near-tie
resolves either way with equal probability -- gains and losses are exchangeable and the
test passes. A cache that systematically helps or systematically hurts fails it, even if
the two means happen to land close together. That is the failure mode a mean-vs-mean
comparison cannot see, and it is the one that would quietly change a thesis figure.

The flip rate itself is capped (``--max-flip-rate``, default 0.25). The near-tie story
only holds if near-ties are rare: these histograms are perturbed by DP noise whose scale
is many orders of magnitude above 1e-6, so the argmax should almost never sit inside the
float-reordering band. If a quarter of the answers move, the explanation has stopped
being "floating point" and the two paths are doing different arithmetic.

**3. The aggregate metrics must stay inside the noise the repo already lives with.**
This is a backstop, and the honest thing to say about its threshold is that the repo has
never run repeated seeds, so there is no measured seed-to-seed number to quote. What
exists is 23 configurations that were executed twice across separate runs (``dpksa`` in
``results/run_20260702_114726`` vs ``run_20260712_094517``, n=40 questions; ``dp_synrag``
in ``run_20260710_100921`` vs ``run_20260712_224716_maquinatoro``, n=102-200). Those
pairs differ by run-time randomness *and* by whatever drifted between the two dates, so
they bound the noise from above rather than measure it. The cleanest of them are the
n=40 ``dpksa`` cells: |delta| <= 0.05 on match_accuracy, <= 0.03 on rouge_l / f1_score,
<= 0.03 on bert_score_f1. Those are the defaults in :data:`METRIC_TOLERANCE`. They are
loose on purpose -- at n questions every proportion metric is quantised at 1/n, so a
single flipped answer already moves match_accuracy by 1/n and a tighter band would fail
on arithmetic rather than on science. Check 2 is what makes the verdict sharp; check 3
only catches a shift too large to be a handful of ties.

Verdicts: ``PASS``, ``FAIL``, or ``INCONCLUSIVE``. Inconclusive is reserved for the case
where nothing measurable happened -- most importantly, where the cached arm was no
faster than the uncached one, which means the switch was almost certainly not honoured
and the two arms ran the same code.


What the generator side must expose
-----------------------------------

This script sets exactly one environment variable, following the repo's existing
``BENCHDP_*`` convention (``BENCHDP_ONLY_LLM_INDEX``, ``BENCHDP_RUN_SEED``, ...):

    BENCHDP_DISABLE_KV_CACHE=1   ->  take the old, re-encode-every-token path

Unset (or any other value) means the cached path. Nothing else is passed. Until some
file in the tree reads that name the harness refuses to run, because two arms that both
take the cached path agree perfectly and prove nothing; ``--allow-unwired`` overrides
the refusal for a rehearsal.
"""

from __future__ import annotations

import argparse
import json
import math
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import dataset_names  # noqa: E402
from experiments.runner import METHODS, _hms, run_method  # noqa: E402
from metrics.utility import contains_accuracy, match_accuracy  # noqa: E402

#: The environment variable the generators must honour. Named here once so the
#: preflight, the runs and the report can never drift apart.
DISABLE_ENV = "BENCHDP_DISABLE_KV_CACHE"

#: Methods whose generation is a hand-written per-token loop, and which therefore have
#: a cache to gain. ``vanilla`` is not one of them -- it calls ``generate()``, which has
#: cached for years -- and is included as a control: if the reference condition stops
#: reproducing itself between two arms, the harness is measuring itself and not the
#: cache.
DEFAULT_METHODS = [
    "vanilla",
    "dp_rag",
    "dpvoterag",
    "dpsparsevoterag",
    "private_rag",
    "private_rag_ada",
]

#: Fields fixed by the config and the accountant, never by an argmax. Any difference is
#: a bug in the cache, not noise. Compared for exact equality, JSON value against JSON
#: value, so a dict-valued curve is compared entry by entry.
BUDGET_FIELDS = (
    "epsilon",
    "epsilon_total",
    "epsilon_workload",
    "epsilon_workload_basic",
    "epsilon_workload_by_q",
    "workload_queries",
    "workload_regime",
    "workload_error",
    "delta",
)

#: Reported, never gated: these count releases actually made, so one flipped argmax that
#: halts a generation early moves them for a legitimate reason.
SPENT_FIELDS = ("epsilon_workload_spent", "budget_utilisation")

#: Absolute tolerance per metric, from the repeated-configuration spread documented in
#: the module docstring. Metrics absent here (perplexity, anything added later) are
#: reported but do not gate the verdict, because they have no bounded scale to threshold.
METRIC_TOLERANCE = {
    "match_accuracy": 0.05,
    "contains_accuracy": 0.05,
    "answer_rate": 0.05,
    "hit_rate": 0.05,
    "retrieval_precision_at_k": 0.05,
    "f1_score": 0.03,
    "rouge_l": 0.03,
    "rouge_1": 0.03,
    "levenshtein": 0.03,
    "bert_score_f1": 0.03,
}

#: Below this the cached arm is not measurably faster, which means the switch probably
#: did nothing and the comparison is between a path and itself.
MIN_SPEEDUP = 1.05


# --------------------------------------------------------------------------- preflight


def _switch_is_wired() -> list[str]:
    """Files that read :data:`DISABLE_ENV`, so the harness can tell a real A/B from
    two runs of the same code path."""
    hits: list[str] = []
    for sub in ("core", "dp_methods", "experiments", "vanilla", "metrics"):
        root = PROJECT_ROOT / sub
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            try:
                if DISABLE_ENV in path.read_text(encoding="utf-8", errors="replace"):
                    hits.append(str(path.relative_to(PROJECT_ROOT)))
            except OSError:
                continue
    return sorted(h for h in hits if h != "scripts/ab_kv_cache.py")


# ------------------------------------------------------------------------------ config


def _set_queries(cfg: dict, queries: int | None, index_docs: int | None, model: str | None) -> dict:
    """A copy of *cfg* resized for the A/B, leaving every unspecified axis alone.

    The smoke configs ask four questions, which is the right size for "does it run" and
    the wrong size for "did the answers move": at n=4 a single flipped answer moves every
    proportion metric by 0.25 and no tolerance can separate a tie from a regression.
    Only the axes passed on the command line are touched, so a method that was given its
    own corpus size on purpose (dp_synrag) keeps it.
    """
    out = json.loads(json.dumps(cfg))  # deep copy through plain JSON types
    cells = out.get("datasets")
    cells = cells if isinstance(cells, list) and cells else []
    blocks = [*cells, out["dataset"]] if isinstance(out.get("dataset"), dict) else list(cells)
    for block in blocks:
        if queries is not None:
            block["max_test_queries"] = queries
        if index_docs is not None:
            block["max_index_docs"] = index_docs
    if queries is not None:
        out["max_queries"] = queries
    if model is not None:
        out["llms"] = [{"model_name": model}]
        if isinstance(out.get("llm"), dict):
            out["llm"] = {**out["llm"], "model_name": model}
    return out


def _materialise_configs(
    methods: list[str],
    config_key: str,
    dest: Path,
    queries: int | None,
    index_docs: int | None,
    model: str | None,
) -> dict[str, tuple[str, dict]]:
    """Write the A/B's configs and return ``{method: (path, parsed)}``.

    Both arms read the same file, so the two runs cannot differ by a config edit made
    between them.
    """
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, tuple[str, dict]] = {}
    for name in methods:
        src = PROJECT_ROOT / METHODS[name][config_key]
        with open(src, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        if queries is None and index_docs is None and model is None:
            out[name] = (str(src), cfg)
            continue
        cfg = _set_queries(cfg, queries, index_docs, model)
        path = dest / f"{name}.yaml"
        header = (
            f"# GENERATED by scripts/ab_kv_cache.py from {src.relative_to(PROJECT_ROOT)}\n"
            f"# Both arms of the KV-cache A/B read this file.\n"
        )
        path.write_text(
            header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        out[name] = (str(path), cfg)
    return out


# -------------------------------------------------------------------------------- runs


def _run_arm(
    label: str,
    disable_cache: bool,
    methods: list[str],
    configs: dict[str, tuple[str, dict]],
    out_root: Path,
    log_root: Path,
    manifest: str,
    seed: int,
) -> dict:
    """Run every method once, dataset-major, with the cache switch held at one value.

    The switch travels through the process environment because that is how the whole
    orchestration layer already talks to its method subprocesses; ``run_method`` copies
    ``os.environ``, so setting it here reaches every child.
    """
    if disable_cache:
        os.environ[DISABLE_ENV] = "1"
    else:
        os.environ.pop(DISABLE_ENV, None)

    print("\n" + "#" * 64)
    print(f"ARM {label}: {DISABLE_ENV}={os.environ.get(DISABLE_ENV, '<unset>')}")
    print("#" * 64)

    order: list[str] = []
    methods_for: dict[str, list[str]] = {}
    for name in methods:
        for ds in dataset_names(configs[name][1]):
            if ds not in methods_for:
                order.append(ds)
                methods_for[ds] = []
            methods_for[ds].append(name)

    rows: list[dict] = []
    t0 = time.time()
    for ds in order:
        for name in methods_for[ds]:
            rows.append(
                run_method(
                    name,
                    METHODS[name]["script"],
                    configs[name][0],
                    output_dir=str(out_root / name / f"seed_{seed}"),
                    log_dir=str(log_root),
                    only_dataset=ds,
                    index_manifest=manifest,
                    run_seed=seed,
                )
            )
    wall = time.time() - t0

    os.environ.pop(DISABLE_ENV, None)
    return {"label": label, "disable_cache": disable_cache, "wall_sec": wall, "methods": rows}


def _score(root: Path) -> int:
    """Compute the metrics for an arm in a fresh process, exactly as the runner does."""
    cmd = [sys.executable, str(PROJECT_ROOT / "experiments" / "score_results.py"), str(root)]
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


# --------------------------------------------------------------------------- comparing


def _payloads(root: Path) -> dict[str, dict]:
    """Result payloads keyed by their path relative to *root*.

    Both arms write the identical directory layout, so the relative path is the cell
    identity and matching on it needs no heuristics.
    """
    out: dict[str, dict] = {}
    for path in sorted(root.rglob("*.json")):
        if path.name in {"metrics_summary.csv", "last_stats.json"}:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and (data.get("results") or data.get("predictions")):
            out[str(path.relative_to(root))] = data
    return out


def _items(data: dict) -> list[dict]:
    return data.get("results") or []


def _predictions(data: dict) -> list[str]:
    preds = data.get("predictions")
    if isinstance(preds, list):
        return [str(p) for p in preds]
    return [str(it.get("prediction", "")) for it in _items(data)]


def _references(data: dict) -> list[str]:
    refs = data.get("references")
    if isinstance(refs, list):
        return [str(r) for r in refs]
    return [str(it.get("reference", "")) for it in _items(data)]


def _golds(data: dict) -> list:
    """Per-query acceptable answers: the alias list when the runner saved one, else the
    single reference. Mirrors what ``score_results`` feeds the overlap metrics, so the
    per-query correctness computed here agrees with the aggregate it is compared to."""
    refs = _references(data)
    items = _items(data)
    if len(items) != len(refs):
        return list(refs)
    out = []
    for it, ref in zip(items, refs):
        aliases = it.get("answers")
        out.append([str(a) for a in aliases] if isinstance(aliases, list) and aliases else ref)
    return out


def _query_ids(data: dict) -> list[str]:
    return [str(it.get("query_id", "")) for it in _items(data)]


def _mcnemar_p(gained: int, lost: int) -> float | None:
    """Exact two-sided McNemar: the probability of a split at least this lopsided if a
    flipped answer were equally likely to help or hurt.

    ``None`` when nothing is discordant -- there is no imbalance to test, which is the
    strongest possible result rather than a missing one.
    """
    n = gained + lost
    if n == 0:
        return None
    k = min(gained, lost)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def _compare_cell(rel: str, on: dict, off: dict) -> dict:
    """One cell's worth of comparison: structure, budget, flips, metrics, speed."""
    cell: dict = {"file": rel, "errors": [], "warnings": []}

    p_on, p_off = _predictions(on), _predictions(off)
    ids_on, ids_off = _query_ids(on), _query_ids(off)
    if ids_on and ids_off and ids_on != ids_off:
        cell["errors"].append("the two arms answered different questions")
    if len(p_on) != len(p_off):
        cell["errors"].append(f"different answer counts: {len(p_on)} vs {len(p_off)}")

    n = min(len(p_on), len(p_off))
    cell["n_queries"] = n
    identical = sum(1 for a, b in zip(p_on[:n], p_off[:n]) if a == b)
    cell["identical_answers"] = identical
    cell["identical_fraction"] = identical / n if n else None
    cell["flipped_indices"] = [i for i in range(n) if p_on[i] != p_off[i]]

    # Budget: exact, or it is a bug.
    budget: dict = {}
    for key in BUDGET_FIELDS:
        a, b = on.get(key), off.get(key)
        if a is None and b is None:
            continue
        budget[key] = {"cache_on": a, "cache_off": b, "equal": a == b}
        if a != b:
            cell["errors"].append(f"epsilon field {key} moved: {a!r} (on) vs {b!r} (off)")
    cell["budget"] = budget

    spent: dict = {}
    for key in SPENT_FIELDS:
        a, b = on.get(key), off.get(key)
        if a is None and b is None:
            continue
        spent[key] = {"cache_on": a, "cache_off": b, "equal": a == b}
    cell["spent"] = spent

    # Paired correctness over the queries whose text moved. Computed on single-item
    # lists so it goes through the repo's own normalisation (SQuAD normalisation,
    # alias max) and cannot disagree with the aggregate it sits next to.
    golds = _golds(on)
    flips: dict[str, dict[str, int]] = {}
    for name, fn in (("match_accuracy", match_accuracy), ("contains_accuracy", contains_accuracy)):
        gained = lost = same = 0
        for i in cell["flipped_indices"]:
            if i >= len(golds):
                continue
            a = fn([p_on[i]], [golds[i]])
            b = fn([p_off[i]], [golds[i]])
            if a > b:
                gained += 1
            elif a < b:
                lost += 1
            else:
                same += 1
        p = _mcnemar_p(gained, lost)
        flips[name] = {
            "cache_gained": gained,
            "cache_lost": lost,
            "flipped_but_equally_correct": same,
            "mcnemar_p": p,
        }
    cell["flip_balance"] = flips

    # Aggregate metrics, for the cells score_results reached.
    m_on = on.get("metrics") or {}
    m_off = off.get("metrics") or {}
    deltas: dict = {}
    for key in sorted(set(m_on) | set(m_off)):
        a, b = m_on.get(key), m_off.get(key)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        if isinstance(a, float) and math.isnan(a) or isinstance(b, float) and math.isnan(b):
            continue
        tol = METRIC_TOLERANCE.get(key)
        deltas[key] = {
            "cache_on": a,
            "cache_off": b,
            "delta": a - b,
            "abs_delta": abs(a - b),
            "tolerance": tol,
            "within": None if tol is None else abs(a - b) <= tol,
        }
    cell["metrics"] = deltas

    t_on, t_off = on.get("time_per_query_sec"), off.get("time_per_query_sec")
    if isinstance(t_on, (int, float)) and isinstance(t_off, (int, float)) and t_on > 0:
        cell["time_per_query_sec"] = {"cache_on": t_on, "cache_off": t_off, "speedup": t_off / t_on}
    return cell


def compare(on_root: Path, off_root: Path, max_flip_rate: float) -> dict:
    """The whole comparison, plus the verdict it implies."""
    on, off = _payloads(on_root), _payloads(off_root)
    only_on = sorted(set(on) - set(off))
    only_off = sorted(set(off) - set(on))
    shared = sorted(set(on) & set(off))

    cells = [_compare_cell(rel, on[rel], off[rel]) for rel in shared]

    errors: list[str] = []
    warnings: list[str] = []
    if only_on or only_off:
        errors.append(
            f"the arms produced different cells: {len(only_on)} only with the cache, "
            f"{len(only_off)} only without"
        )
    if not shared:
        errors.append("no cell was produced by both arms")

    for cell in cells:
        errors.extend(f"{cell['file']}: {e}" for e in cell["errors"])
        frac = cell["identical_fraction"]
        if frac is not None and (1.0 - frac) > max_flip_rate:
            errors.append(
                f"{cell['file']}: {1.0 - frac:.0%} of answers changed (cap {max_flip_rate:.0%}) "
                "— too many to be near-ties resolved differently"
            )
        for name, bal in cell["flip_balance"].items():
            p = bal["mcnemar_p"]
            if p is not None and p < 0.05:
                errors.append(
                    f"{cell['file']}: {name} moved one way — the cache gained "
                    f"{bal['cache_gained']} and lost {bal['cache_lost']} (McNemar p={p:.4f})"
                )
        for name, d in cell["metrics"].items():
            if d["within"] is False:
                errors.append(
                    f"{cell['file']}: {name} moved {d['abs_delta']:.4f} "
                    f"(tolerance {d['tolerance']})"
                )

    speedups = [c["time_per_query_sec"]["speedup"] for c in cells if "time_per_query_sec" in c]
    speedups.sort()
    median_speedup = speedups[len(speedups) // 2] if speedups else None

    inconclusive: list[str] = []
    if not speedups:
        inconclusive.append(
            "no cell recorded time_per_query_sec, so there is no evidence the cached path "
            "ran at all"
        )
    elif median_speedup is not None and median_speedup < MIN_SPEEDUP:
        inconclusive.append(
            f"median per-query speedup {median_speedup:.3f}x is below {MIN_SPEEDUP}x: the "
            f"cached arm was not faster, so {DISABLE_ENV} was probably not honoured and "
            "both arms ran the same path"
        )
    if all((c["identical_fraction"] or 0.0) >= 1.0 for c in cells) and cells and not speedups:
        warnings.append("every answer is byte-identical and no timing was recorded")

    if errors:
        verdict = "FAIL"
    elif inconclusive:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS"

    total_q = sum(c["n_queries"] for c in cells)
    total_same = sum(c["identical_answers"] for c in cells)
    return {
        "verdict": verdict,
        "errors": errors,
        "inconclusive_reasons": inconclusive,
        "warnings": warnings,
        "cells_compared": len(cells),
        "cells_only_cache_on": only_on,
        "cells_only_cache_off": only_off,
        "queries_compared": total_q,
        "identical_answer_fraction": (total_same / total_q) if total_q else None,
        "median_per_query_speedup": median_speedup,
        "criterion": {
            "budget_fields_exact": list(BUDGET_FIELDS),
            "metric_tolerance": METRIC_TOLERANCE,
            "max_flip_rate": max_flip_rate,
            "mcnemar_alpha": 0.05,
            "min_speedup": MIN_SPEEDUP,
        },
        "cells": cells,
    }


# ----------------------------------------------------------------------------- reports


def render(summary: dict, arms: list[dict] | None) -> str:
    """The human-readable half of the output. Same content as the JSON, ordered so the
    verdict and the reason for it are the first two things on screen."""
    L: list[str] = []
    L.append("=" * 72)
    L.append(f"KV-CACHE A/B — VERDICT: {summary['verdict']}")
    L.append("=" * 72)
    for line in summary["errors"]:
        L.append(f"  FAIL  {line}")
    for line in summary["inconclusive_reasons"]:
        L.append(f"  ????  {line}")
    for line in summary["warnings"]:
        L.append(f"  warn  {line}")
    if summary["verdict"] == "PASS":
        L.append("  Epsilon identical everywhere; what moved is confined to near-ties and")
        L.append("  is unbiased; every gated metric is inside the repeated-run band.")
    L.append("")

    frac = summary["identical_answer_fraction"]
    L.append(f"Cells compared        : {summary['cells_compared']}")
    L.append(f"Queries compared      : {summary['queries_compared']}")
    L.append(
        "Byte-identical answers: "
        + (f"{frac:.4f}" if frac is not None else "n/a")
        + (f"  ({summary['queries_compared']} queries)" if frac is not None else "")
    )
    sp = summary["median_per_query_speedup"]
    L.append(f"Median speedup /query : {f'{sp:.3f}x' if sp is not None else 'n/a'}")
    if arms:
        for arm in arms:
            L.append(f"Arm {arm['label']:<10} wall : {_hms(arm['wall_sec'])}")
        L.append(
            "  (the first arm pays for building the index; both arms share it, so the "
            "wall-clock\n   gap is not the speedup — read time_per_query_sec instead)"
        )
    L.append("")

    L.append("Per cell")
    L.append("-" * 72)
    for cell in summary["cells"]:
        f = cell["identical_fraction"]
        L.append(f"  {cell['file']}")
        L.append(
            f"    identical {cell['identical_answers']}/{cell['n_queries']}"
            + (f" ({f:.2f})" if f is not None else "")
            + (
                f"   speedup {cell['time_per_query_sec']['speedup']:.3f}x"
                if "time_per_query_sec" in cell
                else ""
            )
        )
        bad_budget = [k for k, v in cell["budget"].items() if not v["equal"]]
        L.append(
            "    epsilon: EXACT"
            if not bad_budget
            else f"    epsilon: DIFFERS on {bad_budget}   <-- bug, not noise"
        )
        for k, v in cell["spent"].items():
            if not v["equal"]:
                L.append(f"    spent {k}: {v['cache_on']} vs {v['cache_off']} (informational)")
        for name, bal in cell["flip_balance"].items():
            p = bal["mcnemar_p"]
            L.append(
                f"    {name}: +{bal['cache_gained']} / -{bal['cache_lost']} on flips, "
                f"McNemar p={'n/a' if p is None else f'{p:.4f}'}"
            )
        for name, d in sorted(cell["metrics"].items()):
            mark = {True: "ok", False: "OVER", None: "--"}[d["within"]]
            L.append(
                f"    {name:<26} {d['cache_on']:>10.4f}  {d['cache_off']:>10.4f}  "
                f"d={d['delta']:+.4f}  [{mark}]"
            )
        L.append("")
    return "\n".join(L)


def describe_plan(plan: dict) -> str:
    L = ["=" * 72, "KV-CACHE A/B — PLAN", "=" * 72]
    for key in ("config_set", "methods", "seed", "queries", "index_docs", "model", "output"):
        L.append(f"  {key:<12}: {plan[key]}")
    L.append("")
    L.append("  Arm 1  cache OFF : " + f"{DISABLE_ENV}=1")
    L.append("  Arm 2  cache ON  : " + f"{DISABLE_ENV} unset")
    L.append("  Both arms: same config file, same data_seed, same BENCHDP_RUN_SEED, one")
    L.append("  shared index (BENCHDP_RUN_ID), scored afterwards by score_results.py.")
    L.append("")
    L.append("  Cells per arm:")
    for name, datasets in plan["cells"].items():
        L.append(f"    {name:<18} {datasets}")
    L.append("")
    L.append(f"  Switch wired in : {plan['switch_wired_in'] or 'NOWHERE YET'}")
    return "\n".join(L)


# -------------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=DEFAULT_METHODS,
                        metavar="METHOD", help=f"default: {DEFAULT_METHODS}")
    parser.add_argument("--config-set", choices=["smoke", "full", "paper_faithful"],
                        default="smoke", help="which config directory the cells come from")
    parser.add_argument("--seed", type=int, default=1,
                        help="the single run_seed both arms use (default: 1)")
    parser.add_argument("--queries", type=int, default=50,
                        help="questions per cell; the smoke set's four cannot resolve a "
                             "flipped answer from a regression (default: 50, 0 = leave the "
                             "config alone)")
    parser.add_argument("--index-docs", type=int, default=None,
                        help="override every cell's corpus size (default: the config's)")
    parser.add_argument("--model", type=str, default=None,
                        help="override the generator (default: the config's)")
    parser.add_argument("--max-flip-rate", type=float, default=0.25,
                        help="fail above this fraction of changed answers (default: 0.25)")
    parser.add_argument("--out", type=str, default=None,
                        help="artifact directory (default: results/ab_kv_cache_<stamp>)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit; runs nothing, loads no model")
    parser.add_argument("--allow-unwired", action="store_true",
                        help=f"run even though nothing reads {DISABLE_ENV} yet")
    parser.add_argument("--compare-only", nargs=2, metavar=("CACHE_ON_DIR", "CACHE_OFF_DIR"),
                        help="skip both runs and re-compare two existing result trees")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) if args.out else PROJECT_ROOT / "results" / f"ab_kv_cache_{stamp}"

    if args.compare_only:
        on_root, off_root = (Path(p) for p in args.compare_only)
        for p in (on_root, off_root):
            if not p.is_dir():
                print(f"not a directory: {p}")
                return 2
        summary = compare(on_root, off_root, args.max_flip_rate)
        report = render(summary, arms=None)
        print(report)
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "ab_kv_cache_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_root / "ab_kv_cache_report.txt").write_text(report, encoding="utf-8")
        return 0 if summary["verdict"] == "PASS" else 1

    config_key = {"smoke": "smoke_config", "full": "full_config",
                  "paper_faithful": "paper_faithful_config"}[args.config_set]
    queries = args.queries or None
    configs = _materialise_configs(
        args.methods, config_key, out_root / "configs", queries, args.index_docs, args.model
    )
    wired = _switch_is_wired()
    plan = {
        "config_set": args.config_set,
        "methods": args.methods,
        "seed": args.seed,
        "queries": queries or "(config)",
        "index_docs": args.index_docs or "(config)",
        "model": args.model or "(config)",
        "output": str(out_root),
        "cells": {n: dataset_names(cfg) for n, (_, cfg) in configs.items()},
        "switch_wired_in": wired,
        "disable_env": DISABLE_ENV,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(describe_plan(plan))

    if args.dry_run:
        print("\n--dry-run: nothing was executed.")
        return 0

    if not wired and not args.allow_unwired:
        print(
            f"\nREFUSING TO RUN: no file under core/, dp_methods/, experiments/, vanilla/ or\n"
            f"metrics/ reads {DISABLE_ENV}. Both arms would take the identical code path and\n"
            f"agree perfectly, which proves nothing about the cache.\n\n"
            f"The generator side must read {DISABLE_ENV} from the environment and, when it is\n"
            f'set to "1", take the re-encode-every-token path. Re-run once it does, or pass\n'
            f"--allow-unwired to rehearse the harness."
        )
        return 2

    log_root = PROJECT_ROOT / "logs" / f"ab_kv_cache_{stamp}"
    log_root.mkdir(parents=True, exist_ok=True)

    # One run id across both arms, so the index is built once and the second arm reuses
    # it. That guarantees the two arms retrieve from the identical corpus -- the point of
    # the exercise is the generator, and a rebuilt index would put a second difference in
    # the comparison.
    os.environ["BENCHDP_RUN_ID"] = f"abkv_{stamp}"
    fd, manifest = tempfile.mkstemp(prefix=f"benchdp_abkv_{stamp}_", suffix=".manifest")
    os.close(fd)

    arms: list[dict] = []
    try:
        # The uncached arm runs first: it is the current behaviour, so if the machine
        # cannot finish even that, nothing about the cache has been risked.
        arms.append(_run_arm("cache_off", True, args.methods, configs,
                             out_root / "cache_off", log_root, manifest, args.seed))
        arms.append(_run_arm("cache_on", False, args.methods, configs,
                             out_root / "cache_on", log_root, manifest, args.seed))
    finally:
        try:
            for d in {ln.strip() for ln in open(manifest, encoding="utf-8") if ln.strip()}:
                shutil.rmtree(d, ignore_errors=True)
            os.remove(manifest)
        except OSError:
            pass

    failed = [f"{a['label']}/{r['method']}" for a in arms for r in a["methods"] if not r["ok"]]

    print("\n" + "=" * 64)
    print("SCORING both arms (offline metrics)")
    print("=" * 64)
    for arm in arms:
        rc = _score(out_root / arm["label"])
        if rc != 0:
            print(f"  WARNING: scoring {arm['label']} failed (rc={rc})")

    summary = compare(out_root / "cache_on", out_root / "cache_off", args.max_flip_rate)
    summary["arms"] = [{k: v for k, v in a.items() if k != "methods"} for a in arms]
    summary["method_runs"] = [
        {"arm": a["label"], **{k: r[k] for k in ("method", "dataset", "elapsed", "ok")}}
        for a in arms
        for r in a["methods"]
    ]
    if failed:
        summary["errors"].insert(0, f"method runs failed: {failed}")
        summary["verdict"] = "FAIL"

    report = render(summary, arms)
    print("\n" + report)
    (out_root / "ab_kv_cache_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_root / "ab_kv_cache_report.txt").write_text(report, encoding="utf-8")
    print(f"Summary : {out_root / 'ab_kv_cache_summary.json'}")
    print(f"Report  : {out_root / 'ab_kv_cache_report.txt'}")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
