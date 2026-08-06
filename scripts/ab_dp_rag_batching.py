#!/usr/bin/env python
"""Answer one question before the fleet runs: does batching DP-RAG's documents change
the science?

DP-RAG scores one prompt per retrieved document at every generated token. The shipped
top-p retriever selects 43-129 documents, so at 30 tokens that is up to ~3,900 model
calls per query when each document gets its own forward — measured at ~62 s/query, which
prices the whole grid at hundreds of GPU-hours. Grouping the prompts into sub-batches
turns those into ~270 calls. Every row is an independent causal forward, so the grouped
result is the same computation — but not the same floating point: the reduction order
changes and the last-token logits move by ~1e-6.

DP-RAG samples the next token from an exponential mechanism over those logits, so a
1e-6 perturbation can change a sampled token, and every token after it is then generated
from a different prefix. Bit-identical output is not on offer. The question is whether
what moves is confined to near-ties and is unbiased — noise — or whether it points
somewhere, which would mean the two paths are not computing the same thing.

    python scripts/ab_dp_rag_batching.py --dry-run       # print the plan, run nothing
    nohup python scripts/ab_dp_rag_batching.py > ab.log 2>&1 &

Both arms run the same cells with the same ``data_seed`` and ``run_seed`` off the same
config file and the same index; only ``BENCHDP_DISABLE_DOC_BATCHING`` differs.


Why this is not scripts/ab_kv_cache.py
--------------------------------------

That script's budget and metric criteria are exactly right and are reused here verbatim.
Its third criterion is not: it caps the fraction of *answers* that changed at 25%, and on
that basis it failed DP-RAG for changing 100% of them. That was a false positive.
Answer-level identity is the wrong statistic for autoregressive sampling — one flipped
token at step 3 rewrites steps 4..30 by construction, so a single near-tie resolved the
other way shows up as a wholly different answer. Measuring the amplification tells you
nothing about the perturbation.

The statistic here is per *token*. For each pair of answers, the tokens up to the first
disagreement are steps where both arms sampled the same token from the same seed; the
first disagreement is one flip; everything after it is downstream of that flip and is
discarded as evidence rather than counted. That estimates the per-step probability that
the two paths' logits differed enough to move the exponential mechanism's draw, which is
the quantity the near-tie story is actually about. At the shipped temperature it should
be a percent or two: the sampling draw is fixed by the seed, so a relative 1e-6 shift in
the categorical weights only flips a step whose cumulative mass crosses the draw inside
that band.


The verdict
-----------

**1. Epsilon is exact or it is a bug.** :data:`~scripts.ab_kv_cache.BUDGET_FIELDS` must
be equal to the last bit; the budget is a function of the config and the accountant and
of nothing a forward pass computes. ``epsilon_workload_spent`` / ``budget_utilisation``
count releases actually made and are reported, not gated: a flipped token that draws EOS
one step earlier moves them legitimately.

**2. What moved must be unbiased**, tested pairwise. Both arms answer the same questions
in the same order, so among the queries whose text changed, per-query correctness is
compared with an exact two-sided McNemar. Under the null — the batching only resolves
near-ties, and a near-tie resolves either way with equal probability — gains and losses
are exchangeable. A path that systematically helps or hurts fails, even when the two
means land close together.

**3. The per-token flip rate must stay small** (``--max-token-flip-rate``), and **the
aggregate metrics must stay inside the repeated-run band** the repo already lives with
(:data:`~scripts.ab_kv_cache.METRIC_TOLERANCE`).

Verdicts: ``PASS``, ``FAIL``, ``INCONCLUSIVE``. Inconclusive is reserved for the case
where the switch demonstrably did nothing — here that is read off the exact model-call
count each arm records (``n_forwards_per_query``), not off a wall-clock, so it is
reproducible on any machine.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import dataset_names  # noqa: E402
from experiments.runner import METHODS, _hms, run_method  # noqa: E402
from metrics.utility import contains_accuracy, match_accuracy  # noqa: E402
from scripts.ab_kv_cache import (  # noqa: E402
    BUDGET_FIELDS,
    METRIC_TOLERANCE,
    SPENT_FIELDS,
    _golds,
    _materialise_configs,
    _mcnemar_p,
    _payloads,
    _predictions,
    _query_ids,
    _score,
)

#: The environment variable the generator must honour, named once so the preflight, the
#: runs and the report cannot drift apart. Mirrors ``BENCHDP_DISABLE_KV_CACHE``.
DISABLE_ENV = "BENCHDP_DISABLE_DOC_BATCHING"

#: The only method with a per-document forward loop over a top-p document set.
METHOD = "dp_rag"

#: Exact model calls per query, as each arm records it. The ratio between the arms is
#: the whole effect of the change and is what decides whether the grid fits; it is also
#: how the harness tells a real A/B from two runs of the same code path, because it is
#: integer-exact where a wall-clock is not.
FORWARDS_FIELD = "n_forwards_per_query"

#: Below this forward-count ratio the sequential arm did not actually run sequentially,
#: so the two arms took the same path and the comparison proves nothing. The shipped
#: sub-batch is 16 and the shipped k_docs is 43-129, so the real ratio is ~14x; anything
#: near 1 means the switch was not honoured.
MIN_FORWARD_RATIO = 1.5

#: Fail above this fraction of *decoding steps* that sampled a different token. See the
#: module docstring: at the shipped temperature a relative 1e-6 shift in the categorical
#: weights should move only a percent or two of the seeded draws, so a tenth of every
#: step flipping means the two paths are doing different arithmetic, not the same
#: arithmetic in a different order.
MAX_TOKEN_FLIP_RATE = 0.10

#: Word-and-punctuation runs. The comparison only needs a consistent, model-independent
#: segmentation of two answer strings — loading the run's tokenizer to score a text file
#: would tie the report to a checkpoint it does not otherwise need.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _first_divergence(a: str, b: str) -> tuple[int, int]:
    """``(agreeing_steps, flips)`` for one pair of answers.

    Tokens before the first disagreement are steps where both arms sampled the same
    token from the same seed, and count as evidence of agreement. The first disagreement
    is one flip. Everything after it was generated from a prefix the two arms no longer
    share, so it is not independent evidence of anything and is discarded — which is
    exactly what an answer-level identity check gets wrong.
    """
    ta, tb = _tokens(a), _tokens(b)
    n = min(len(ta), len(tb))
    shared = 0
    while shared < n and ta[shared] == tb[shared]:
        shared += 1
    if shared == len(ta) == len(tb):
        return shared, 0
    return shared, 1


# --------------------------------------------------------------------------- preflight


def _switch_is_wired() -> list[str]:
    """Files that read :data:`DISABLE_ENV`, so the harness can refuse a sham A/B."""
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
    return sorted(hits)


# -------------------------------------------------------------------------------- runs


def _run_arm(
    label: str,
    sequential: bool,
    config_path: str,
    datasets: list[str],
    out_root: Path,
    log_root: Path,
    manifest: str,
    seed: int,
) -> dict:
    """Run every dataset once with the batching switch held at one value.

    The switch travels through the process environment because that is how the whole
    orchestration layer already talks to its method subprocesses; ``run_method`` copies
    ``os.environ``, so setting it here reaches every child.
    """
    if sequential:
        os.environ[DISABLE_ENV] = "1"
    else:
        os.environ.pop(DISABLE_ENV, None)

    print("\n" + "#" * 64)
    print(f"ARM {label}: {DISABLE_ENV}={os.environ.get(DISABLE_ENV, '<unset>')}")
    print("#" * 64)

    rows: list[dict] = []
    t0 = time.time()
    for ds in datasets:
        rows.append(
            run_method(
                METHOD,
                METHODS[METHOD]["script"],
                config_path,
                output_dir=str(out_root / METHOD / f"seed_{seed}"),
                log_dir=str(log_root),
                only_dataset=ds,
                index_manifest=manifest,
                run_seed=seed,
            )
        )
    wall = time.time() - t0

    os.environ.pop(DISABLE_ENV, None)
    return {"label": label, "sequential": sequential, "wall_sec": wall, "methods": rows}


# --------------------------------------------------------------------------- comparing


def _compare_cell(rel: str, batched: dict, sequential: dict) -> dict:
    """One cell's worth of comparison: structure, budget, token flips, metrics, cost."""
    cell: dict = {"file": rel, "errors": [], "warnings": []}

    p_bat, p_seq = _predictions(batched), _predictions(sequential)
    ids_bat, ids_seq = _query_ids(batched), _query_ids(sequential)
    if ids_bat and ids_seq and ids_bat != ids_seq:
        cell["errors"].append("the two arms answered different questions")
    if len(p_bat) != len(p_seq):
        cell["errors"].append(f"different answer counts: {len(p_bat)} vs {len(p_seq)}")

    n = min(len(p_bat), len(p_seq))
    cell["n_queries"] = n
    identical = sum(1 for a, b in zip(p_bat[:n], p_seq[:n]) if a == b)
    cell["identical_answers"] = identical
    # Reported for orientation only. One flipped token rewrites the rest of the answer,
    # so this number measures the amplification, not the perturbation, and nothing is
    # gated on it.
    cell["identical_answer_fraction"] = identical / n if n else None
    cell["flipped_indices"] = [i for i in range(n) if p_bat[i] != p_seq[i]]

    # The statistic that is gated: per decoding step.
    agree = flips = 0
    for i in range(n):
        a, f = _first_divergence(p_bat[i], p_seq[i])
        agree += a
        flips += f
    steps = agree + flips
    cell["token_steps_compared"] = steps
    cell["token_flips"] = flips
    cell["token_flip_rate"] = (flips / steps) if steps else None

    # Budget: exact, or it is a bug.
    budget: dict = {}
    for key in BUDGET_FIELDS:
        a, b = batched.get(key), sequential.get(key)
        if a is None and b is None:
            continue
        budget[key] = {"batched": a, "sequential": b, "equal": a == b}
        if a != b:
            cell["errors"].append(
                f"epsilon field {key} moved: {a!r} (batched) vs {b!r} (sequential)"
            )
    cell["budget"] = budget

    spent: dict = {}
    for key in SPENT_FIELDS:
        a, b = batched.get(key), sequential.get(key)
        if a is None and b is None:
            continue
        spent[key] = {"batched": a, "sequential": b, "equal": a == b}
    cell["spent"] = spent

    # Paired correctness over the queries whose text moved. Computed on single-item lists
    # so it goes through the repo's own normalisation (SQuAD normalisation, alias max) and
    # cannot disagree with the aggregate it sits next to.
    golds = _golds(batched)
    flip_balance: dict[str, dict] = {}
    for name, fn in (("match_accuracy", match_accuracy), ("contains_accuracy", contains_accuracy)):
        gained = lost = same = 0
        for i in cell["flipped_indices"]:
            if i >= len(golds):
                continue
            a = fn([p_bat[i]], [golds[i]])
            b = fn([p_seq[i]], [golds[i]])
            if a > b:
                gained += 1
            elif a < b:
                lost += 1
            else:
                same += 1
        flip_balance[name] = {
            "batched_gained": gained,
            "batched_lost": lost,
            "flipped_but_equally_correct": same,
            "mcnemar_p": _mcnemar_p(gained, lost),
        }
    cell["flip_balance"] = flip_balance

    m_bat = batched.get("metrics") or {}
    m_seq = sequential.get("metrics") or {}
    deltas: dict = {}
    for key in sorted(set(m_bat) | set(m_seq)):
        a, b = m_bat.get(key), m_seq.get(key)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        if a != a or b != b:  # NaN
            continue
        tol = METRIC_TOLERANCE.get(key)
        deltas[key] = {
            "batched": a,
            "sequential": b,
            "delta": a - b,
            "abs_delta": abs(a - b),
            "tolerance": tol,
            "within": None if tol is None else abs(a - b) <= tol,
        }
    cell["metrics"] = deltas

    f_bat, f_seq = batched.get(FORWARDS_FIELD), sequential.get(FORWARDS_FIELD)
    if isinstance(f_bat, (int, float)) and isinstance(f_seq, (int, float)) and f_bat > 0:
        cell["forwards_per_query"] = {
            "batched": f_bat,
            "sequential": f_seq,
            "ratio": f_seq / f_bat,
            "docs_per_query": batched.get("n_docs_per_query"),
        }
    t_bat, t_seq = batched.get("time_per_query_sec"), sequential.get("time_per_query_sec")
    if isinstance(t_bat, (int, float)) and isinstance(t_seq, (int, float)) and t_bat > 0:
        cell["time_per_query_sec"] = {
            "batched": t_bat,
            "sequential": t_seq,
            "speedup": t_seq / t_bat,
        }
    return cell


def compare(batched_root: Path, sequential_root: Path, max_token_flip_rate: float) -> dict:
    """The whole comparison, plus the verdict it implies."""
    bat, seq = _payloads(batched_root), _payloads(sequential_root)
    only_bat = sorted(set(bat) - set(seq))
    only_seq = sorted(set(seq) - set(bat))
    shared = sorted(set(bat) & set(seq))

    cells = [_compare_cell(rel, bat[rel], seq[rel]) for rel in shared]

    errors: list[str] = []
    warnings: list[str] = []
    if only_bat or only_seq:
        errors.append(
            f"the arms produced different cells: {len(only_bat)} only batched, "
            f"{len(only_seq)} only sequential"
        )
    if not shared:
        errors.append("no cell was produced by both arms")

    for cell in cells:
        errors.extend(f"{cell['file']}: {e}" for e in cell["errors"])
        rate = cell["token_flip_rate"]
        if rate is not None and rate > max_token_flip_rate:
            errors.append(
                f"{cell['file']}: {rate:.1%} of decoding steps sampled a different token "
                f"(cap {max_token_flip_rate:.1%}) — too many to be near-ties resolved "
                "differently"
            )
        for name, bal in cell["flip_balance"].items():
            p = bal["mcnemar_p"]
            if p is not None and p < 0.05:
                errors.append(
                    f"{cell['file']}: {name} moved one way — batching gained "
                    f"{bal['batched_gained']} and lost {bal['batched_lost']} "
                    f"(McNemar p={p:.4f})"
                )
        for name, d in cell["metrics"].items():
            if d["within"] is False:
                errors.append(
                    f"{cell['file']}: {name} moved {d['abs_delta']:.4f} "
                    f"(tolerance {d['tolerance']})"
                )

    ratios = sorted(c["forwards_per_query"]["ratio"] for c in cells if "forwards_per_query" in c)
    median_ratio = ratios[len(ratios) // 2] if ratios else None
    speedups = sorted(c["time_per_query_sec"]["speedup"] for c in cells if "time_per_query_sec" in c)
    median_speedup = speedups[len(speedups) // 2] if speedups else None

    inconclusive: list[str] = []
    if not ratios:
        inconclusive.append(
            f"no cell recorded {FORWARDS_FIELD}, so there is no evidence the two arms took "
            "different paths"
        )
    elif median_ratio is not None and median_ratio < MIN_FORWARD_RATIO:
        inconclusive.append(
            f"median forward-count ratio {median_ratio:.2f}x is below {MIN_FORWARD_RATIO}x: "
            f"the sequential arm made about as many model calls as the batched one, so "
            f"{DISABLE_ENV} was probably not honoured and both arms ran the same path"
        )
    total_steps = sum(c["token_steps_compared"] for c in cells)
    if total_steps < 100:
        warnings.append(
            f"only {total_steps} decoding steps compared: the flip rate has no resolution"
        )

    verdict = "FAIL" if errors else ("INCONCLUSIVE" if inconclusive else "PASS")

    total_q = sum(c["n_queries"] for c in cells)
    total_flips = sum(c["token_flips"] for c in cells)
    return {
        "verdict": verdict,
        "errors": errors,
        "inconclusive_reasons": inconclusive,
        "warnings": warnings,
        "cells_compared": len(cells),
        "cells_only_batched": only_bat,
        "cells_only_sequential": only_seq,
        "queries_compared": total_q,
        "token_steps_compared": total_steps,
        "token_flips": total_flips,
        "token_flip_rate": (total_flips / total_steps) if total_steps else None,
        "median_forward_ratio": median_ratio,
        "median_per_query_speedup": median_speedup,
        "criterion": {
            "budget_fields_exact": list(BUDGET_FIELDS),
            "metric_tolerance": METRIC_TOLERANCE,
            "max_token_flip_rate": max_token_flip_rate,
            "mcnemar_alpha": 0.05,
            "min_forward_ratio": MIN_FORWARD_RATIO,
        },
        "cells": cells,
    }


# ----------------------------------------------------------------------------- reports


def render(summary: dict, arms: list[dict] | None) -> str:
    """The human-readable half of the output, verdict first."""
    L: list[str] = []
    L.append("=" * 72)
    L.append(f"DP-RAG DOCUMENT-BATCHING A/B — VERDICT: {summary['verdict']}")
    L.append("=" * 72)
    for line in summary["errors"]:
        L.append(f"  FAIL  {line}")
    for line in summary["inconclusive_reasons"]:
        L.append(f"  ????  {line}")
    for line in summary["warnings"]:
        L.append(f"  warn  {line}")
    if summary["verdict"] == "PASS":
        L.append("  Epsilon identical everywhere; the per-token flip rate is inside the")
        L.append("  near-tie band and unbiased; every gated metric is inside the")
        L.append("  repeated-run band.")
    L.append("")

    rate = summary["token_flip_rate"]
    L.append(f"Cells compared        : {summary['cells_compared']}")
    L.append(f"Queries compared      : {summary['queries_compared']}")
    L.append(
        "Token flip rate       : "
        + (f"{rate:.4f}" if rate is not None else "n/a")
        + f"  ({summary['token_flips']}/{summary['token_steps_compared']} decoding steps)"
    )
    ratio = summary["median_forward_ratio"]
    L.append(f"Median forwards saved : {f'{ratio:.2f}x' if ratio is not None else 'n/a'}")
    sp = summary["median_per_query_speedup"]
    L.append(f"Median speedup /query : {f'{sp:.3f}x' if sp is not None else 'n/a'}")
    if arms:
        for arm in arms:
            L.append(f"Arm {arm['label']:<12} wall: {_hms(arm['wall_sec'])}")
        L.append(
            "  (the first arm pays for building the index; both arms share it, so the "
            "wall-clock\n   gap is not the speedup — read time_per_query_sec instead)"
        )
    L.append("")

    L.append("Per cell")
    L.append("-" * 72)
    for cell in summary["cells"]:
        L.append(f"  {cell['file']}")
        parts = [
            f"    steps {cell['token_steps_compared']}, flips {cell['token_flips']}"
            + (
                f" ({cell['token_flip_rate']:.4f})"
                if cell["token_flip_rate"] is not None
                else ""
            )
        ]
        if "forwards_per_query" in cell:
            fw = cell["forwards_per_query"]
            parts.append(
                f"   forwards/query {fw['sequential']:.0f} -> {fw['batched']:.0f} "
                f"({fw['ratio']:.2f}x, {fw['docs_per_query'] or 0:.0f} docs)"
            )
        if "time_per_query_sec" in cell:
            parts.append(f"   speedup {cell['time_per_query_sec']['speedup']:.3f}x")
        L.append("".join(parts))
        L.append(
            f"    whole answers identical {cell['identical_answers']}/{cell['n_queries']} "
            "(informational — one flip rewrites the rest)"
        )
        bad_budget = [k for k, v in cell["budget"].items() if not v["equal"]]
        L.append(
            "    epsilon: EXACT"
            if not bad_budget
            else f"    epsilon: DIFFERS on {bad_budget}   <-- bug, not noise"
        )
        for k, v in cell["spent"].items():
            if not v["equal"]:
                L.append(f"    spent {k}: {v['batched']} vs {v['sequential']} (informational)")
        for name, bal in cell["flip_balance"].items():
            p = bal["mcnemar_p"]
            L.append(
                f"    {name}: +{bal['batched_gained']} / -{bal['batched_lost']} on flips, "
                f"McNemar p={'n/a' if p is None else f'{p:.4f}'}"
            )
        for name, d in sorted(cell["metrics"].items()):
            mark = {True: "ok", False: "OVER", None: "--"}[d["within"]]
            L.append(
                f"    {name:<26} {d['batched']:>10.4f}  {d['sequential']:>10.4f}  "
                f"d={d['delta']:+.4f}  [{mark}]"
            )
        L.append("")
    return "\n".join(L)


def describe_plan(plan: dict) -> str:
    L = ["=" * 72, "DP-RAG DOCUMENT-BATCHING A/B — PLAN", "=" * 72]
    for key in ("config_set", "seed", "queries", "index_docs", "model", "datasets", "output"):
        L.append(f"  {key:<12}: {plan[key]}")
    L.append("")
    L.append(f"  Arm 1  sequential : {DISABLE_ENV}=1   (one forward per document)")
    L.append(f"  Arm 2  batched    : {DISABLE_ENV} unset (DOC_SUB_BATCH prompts per forward)")
    L.append("  Both arms: same config file, same data_seed, same BENCHDP_RUN_SEED, one")
    L.append("  shared index (BENCHDP_RUN_ID), scored afterwards by score_results.py.")
    L.append("")
    L.append(f"  Switch wired in : {plan['switch_wired_in'] or 'NOWHERE YET'}")
    return "\n".join(L)


# -------------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-set", choices=["smoke", "full", "paper_faithful"],
                        default="full", help="which config directory the cells come from")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="restrict to these datasets (default: every one in the config)")
    parser.add_argument("--seed", type=int, default=1,
                        help="the single run_seed both arms use (default: 1)")
    parser.add_argument("--queries", type=int, default=50,
                        help="questions per cell (default: 50, 0 = leave the config alone)")
    parser.add_argument("--index-docs", type=int, default=None,
                        help="override every cell's corpus size (default: the config's)")
    parser.add_argument("--model", type=str, default=None,
                        help="override the generator (default: the config's)")
    parser.add_argument("--max-token-flip-rate", type=float, default=MAX_TOKEN_FLIP_RATE,
                        help=f"fail above this fraction of flipped decoding steps "
                             f"(default: {MAX_TOKEN_FLIP_RATE})")
    parser.add_argument("--out", type=str, default=None,
                        help="artifact directory (default: results/ab_dp_rag_batching_<stamp>)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit; runs nothing, loads no model")
    parser.add_argument("--allow-unwired", action="store_true",
                        help=f"run even though nothing reads {DISABLE_ENV}")
    parser.add_argument("--compare-only", nargs=2, metavar=("BATCHED_DIR", "SEQUENTIAL_DIR"),
                        help="skip both runs and re-compare two existing result trees")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (
        Path(args.out) if args.out
        else PROJECT_ROOT / "results" / f"ab_dp_rag_batching_{stamp}"
    )

    if args.compare_only:
        bat_root, seq_root = (Path(p) for p in args.compare_only)
        for p in (bat_root, seq_root):
            if not p.is_dir():
                print(f"not a directory: {p}")
                return 2
        summary = compare(bat_root, seq_root, args.max_token_flip_rate)
        report = render(summary, arms=None)
        print(report)
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "ab_dp_rag_batching_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_root / "ab_dp_rag_batching_report.txt").write_text(report, encoding="utf-8")
        return 0 if summary["verdict"] == "PASS" else 1

    config_key = {"smoke": "smoke_config", "full": "full_config",
                  "paper_faithful": "paper_faithful_config"}[args.config_set]
    queries = args.queries or None
    configs = _materialise_configs(
        [METHOD], config_key, out_root / "configs", queries, args.index_docs, args.model
    )
    config_path, config = configs[METHOD]
    datasets = args.datasets or dataset_names(config)
    wired = _switch_is_wired()
    plan = {
        "config_set": args.config_set,
        "seed": args.seed,
        "queries": queries or "(config)",
        "index_docs": args.index_docs or "(config)",
        "model": args.model or "(config)",
        "datasets": datasets,
        "output": str(out_root),
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
            f"agree perfectly, which proves nothing.\n\n"
            f"Pass --allow-unwired to rehearse the harness."
        )
        return 2

    log_root = PROJECT_ROOT / "logs" / f"ab_dp_rag_batching_{stamp}"
    log_root.mkdir(parents=True, exist_ok=True)

    # One run id across both arms, so the index is built once and the second arm reuses
    # it. That guarantees the two arms retrieve from the identical corpus -- the point of
    # the exercise is the generation loop, and a rebuilt index would put a second
    # difference in the comparison.
    os.environ["BENCHDP_RUN_ID"] = f"abdocbatch_{stamp}"
    fd, manifest = tempfile.mkstemp(prefix=f"benchdp_abdocbatch_{stamp}_", suffix=".manifest")
    os.close(fd)

    arms: list[dict] = []
    try:
        # The sequential arm runs first: it is the current behaviour, so if the machine
        # cannot finish even that, nothing about the change has been risked.
        arms.append(_run_arm("sequential", True, config_path, datasets,
                             out_root / "sequential", log_root, manifest, args.seed))
        arms.append(_run_arm("batched", False, config_path, datasets,
                             out_root / "batched", log_root, manifest, args.seed))
    finally:
        try:
            for d in {ln.strip() for ln in open(manifest, encoding="utf-8") if ln.strip()}:
                shutil.rmtree(d, ignore_errors=True)
            os.remove(manifest)
        except OSError:
            pass

    failed = [f"{a['label']}/{r['dataset']}" for a in arms for r in a["methods"] if not r["ok"]]

    print("\n" + "=" * 64)
    print("SCORING both arms (offline metrics)")
    print("=" * 64)
    for arm in arms:
        rc = _score(out_root / arm["label"])
        if rc != 0:
            print(f"  WARNING: scoring {arm['label']} failed (rc={rc})")

    summary = compare(out_root / "batched", out_root / "sequential", args.max_token_flip_rate)
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
    (out_root / "ab_dp_rag_batching_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_root / "ab_dp_rag_batching_report.txt").write_text(report, encoding="utf-8")
    print(f"Summary : {out_root / 'ab_dp_rag_batching_summary.json'}")
    print(f"Report  : {out_root / 'ab_dp_rag_batching_report.txt'}")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
