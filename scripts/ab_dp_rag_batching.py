#!/usr/bin/env python
"""Answer one question before the fleet runs: does the fast DP-RAG inner loop change
the science?

DP-RAG scores one prompt per retrieved document at every generated token. The shipped
top-p retriever selects 43-129 documents, so at 30 tokens that is up to ~3,900 model
calls per query when each document gets its own uncached forward — measured at ~62
s/query, which prices the whole grid at hundreds of GPU-hours. Three changes to that
loop bring it to ~8 s/query in steady state: the per-document prompts are grouped into
sub-batches of ``DOC_SUB_BATCH``; the rows are sorted by length first so padding stops
dominating; and the KV-cache VRAM budget is split between sub-batches in proportion to
what each needs rather than equally, which admits nearly all of them instead of one.

Every row is an independent causal forward and every prompt grows by the same sampled
token, so all three leave the computation the same on paper — but not in floating point.
Grouping, padding and caching all change the order the reductions run in, and the
last-token logits move by ~1e-6. DP-RAG samples the next token from an exponential
mechanism over those logits, so a 1e-6 perturbation can change a sampled token, and
every token after it is then generated from a different prefix. Bit-identical output is
not on offer. The question is whether what moves is confined to near-ties and is
unbiased — noise — or whether it points somewhere, which would mean the two paths are
not computing the same thing.

    python scripts/ab_dp_rag_batching.py --dry-run       # print the plan, run nothing
    nohup python scripts/ab_dp_rag_batching.py > ab.log 2>&1 &


The arms
--------

**Everything off** — one uncached forward per document, rows in retrieval order — is
the path that produced every result currently in ``results/``. **Everything on** is what
the grid will run. That pair is the change a reader of the thesis is exposed to, so that
is what is validated; ``--isolate`` can hold a single knob instead, which is useful for
bisecting a failure but is not the comparison that needs to pass.

Both arms run the same cells off the same config file with the same ``data_seed`` and
``run_seed`` against one shared index. Only these differ:

    BENCHDP_DISABLE_DOC_BATCHING=1   one forward per document
    BENCHDP_DISABLE_LENGTH_SORT=1    rows in the order they arrive
    BENCHDP_DISABLE_KV_CACHE=1       re-encode the whole prompt at every step

A **control** method runs in both arms alongside DP-RAG. ``vanilla`` reads none of those
switches — it calls ``generate()`` — so its two arms must agree byte for byte. If they
do not, the harness is measuring itself (a non-deterministic machine, an index rebuilt
between arms, a seed that did not reach a subprocess) and nothing it says about DP-RAG
can be attributed to the change.


The verdict
-----------

**1. Epsilon is exact or it is a bug.** :data:`BUDGET_FIELDS` must be equal to the last
bit: the budget is a function of the config and the accountant and of nothing a forward
pass computes. ``epsilon_workload_spent`` / ``budget_utilisation`` count releases
actually made and are reported, not gated — a flipped token that draws EOS one step
earlier moves them legitimately.

**2. Retrieval must be untouched.** The change is to the generation loop; the retrieved
document ids are recorded per query and must match exactly. A cell where they moved is
not a generation A/B at all, and every downstream number in it is confounded.

**3. The control must reproduce itself**, byte for byte, in both arms.

**4. What moved must be unbiased.** This is the check that carries the argument, and it
is paired: both arms answer the same ``query_id``s in the same order, so the comparison
is per query rather than between two means. Among the queries whose answer text changed,
per-query correctness is compared with an exact two-sided McNemar (binomial on the
discordant pairs), pooled over cells because that is where the power is; per-cell tests
are reported and gated at a Bonferroni-corrected alpha so that running more cells cannot
manufacture a failure. Under the null — the fast path only resolves near-ties, and a
near-tie resolves either way with equal probability — gains and losses are exchangeable.
A path that systematically helps or hurts fails, even when the two means land close
together. That is the failure mode a mean-versus-mean comparison cannot see, and it is
the one that would quietly change a thesis figure.

A test with fewer than :data:`MIN_DISCORDANT` discordant pairs cannot reach p < 0.05 at
any split, so a pass from it would be unearned: that is reported as INCONCLUSIVE with
the remedy (more questions), not as a PASS.

**5. The two arms must not disagree on the very first token of essentially every
answer.** Note what is *not* gated here: the fraction of whole answers that changed.
``scripts/ab_kv_cache.py`` capped that at 25% and failed DP-RAG for changing 100% of
them — a false positive. Answer-level identity is the wrong statistic for autoregressive
sampling: one flipped token at step 3 rewrites steps 4..30 by construction, so a single
near-tie resolved the other way shows up as a wholly different answer. Measuring the
amplification says nothing about the perturbation. The statistic here is per decoding
step — tokens up to the first disagreement are steps where both arms sampled the same
token from the same seed, the first disagreement is one flip, and everything after it is
downstream of that flip and is discarded as evidence rather than counted.

That per-step flip rate is **reported, not gated by default**, and
:data:`MAX_FIRST_STEP_DIVERGENCE` is what is gated instead. The reason is in
"Which epsilon" below: the rate a correct implementation produces depends on where on
the epsilon axis the cell sits, nobody has ever measured it at either end, and a cap
guessed at from an intuition about one end is precisely the false positive this script
exists to avoid repeating. What *can* be argued without a measurement is the extreme.
Both arms draw the token from the same seeded uniform (``default_rng(seed).choice``
inverts the CDF), so the per-step probability that they disagree is the L1 distance
between the two CDFs, which scales linearly with the perturbation to the exponent. A
~1e-6 relative shift cannot move a CDF far enough to resample nearly every step at any
temperature: under a per-step hazard ``h`` the fraction of answers that differ at the
*first* token is ``h``, so first-token disagreement on essentially every query means
``h ≈ 1`` — the two paths are not perturbing a shared computation, they are computing
different things. That is regime-free, so it is the gate. Pass
``--max-token-flip-rate`` to gate the rate as well once this repo has a measured null
for it; the first run of this script is what produces that number.

Which epsilon
-------------

Both ends of the shipped axis, by default, because the sampling law is **not**
epsilon-invariant. The clipping bound is derived as ``C = token_epsilon * T / 2`` and is
also the bound ``_clip_logits`` applies, so with the clip binding the exponent is

    (theta / T) * logits_pub  +  (token_epsilon / 2) * SUM_j l_centered_j / max|l_centered_j|

and ``token_epsilon`` re-enters through the utility it was cancelled out of. The clip
goes inactive only at ``token_epsilon >= 1/T = 3.33`` (the shipped ``temperature: 0.3``,
``max|l_centered| <= 0.5``), and the shipped grid tops out at ``token_epsilon = 1.59``,
so it binds on every cell that will ever run. The distribution is correspondingly
near-uniform at the tight end and peaked at the loose one — a measured top-1 probability
of ~0.019 at ``token_epsilon = 0.0074`` against ~0.207 at 5.0. Two cells whose sampling
distributions differ that much are two different tests, and the verdict has to cover
both, so ``--epsilons`` defaults to the tightest and the loosest shipped target rather
than to one point in the middle.

**6. The aggregate metrics must stay inside the band the repo already lives with**
(:data:`METRIC_TOLERANCE`). Read :data:`NOISE_BAND_CAVEAT` before quoting that band: the
repo has never run repeated seeds, so there is no measured seed-to-seed figure, and what
the band is derived from bounds the noise from above rather than measuring it.

Verdicts: ``PASS``, ``FAIL``, ``INCONCLUSIVE``; the exit status is 0 only for ``PASS``.
Inconclusive is reserved for the case where the comparison could not have said anything
— most importantly, where the two arms made about the same number of model calls, which
means the switches were not honoured and both arms ran the same code. That is read off
the exact per-query forward count each arm records (``n_forwards_per_query``), not off a
wall clock, so it is reproducible on any machine. (Only the sub-batching changes that
count; under ``--isolate sort`` or ``--isolate kv_cache`` there is nothing but the clock
to go on, and the harness says so.)
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

import yaml

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
    _items,
    _mcnemar_p,
    _payloads,
    _predictions,
    _query_ids,
    _score,
    _set_queries,
)

#: The environment variables the fast path must honour, one per optimisation, named
#: once so the preflight, the runs and the report cannot drift apart. The baseline arm
#: sets all of them (or the one ``--isolate`` names); the optimised arm sets none.
SWITCHES = {
    "batching": "BENCHDP_DISABLE_DOC_BATCHING",
    "sort": "BENCHDP_DISABLE_LENGTH_SORT",
    "kv_cache": "BENCHDP_DISABLE_KV_CACHE",
}

#: The method under test: the only one with a per-document forward loop over a top-p
#: document set.
METHOD = "dp_rag"

#: The control. It generates with ``generate()`` and reads none of :data:`SWITCHES`, so
#: its two arms must agree byte for byte; anything else means the harness is measuring
#: itself rather than the change.
CONTROL_METHOD = "vanilla"

#: Exact model calls per query, as each arm records it. The ratio between the arms is
#: the whole effect of the change and is what decides whether the grid fits; it is also
#: how the harness tells a real A/B from two runs of the same code path, because it is
#: integer-exact where a wall clock is not.
FORWARDS_FIELD = "n_forwards_per_query"

#: Below this forward-count ratio the baseline arm did not actually run one forward per
#: document, so the two arms took the same path and the comparison proves nothing. The
#: shipped sub-batch is 16 and the shipped k_docs is 43-129, so the real ratio is ~14x;
#: anything near 1 means the switches were not honoured.
MIN_FORWARD_RATIO = 1.5

#: The same question for the knobs that do not change the *number* of model calls: the
#: length sort changes how much padding each call carries and the KV cache changes what
#: each call recomputes, so only the clock can tell whether they were honoured. A wall
#: clock is machine-dependent and shares the GPU with whatever else is running, hence a
#: threshold barely above 1 — it is asking "did anything happen at all", not measuring.
MIN_SPEEDUP = 1.05

#: Which measurement can show that the baseline arm really took the slow path, per
#: ``--isolate``. Turning the sub-batching off is the only change that alters the
#: forward count, and that count is integer-exact where a clock is not.
EVIDENCE_FOR_ISOLATE = {
    "all": "forwards",
    "batching": "forwards",
    "sort": "speed",
    "kv_cache": "speed",
}

#: Reported always; gated only when ``--max-token-flip-rate`` names a number. There is
#: no measured null for this rate anywhere in this repo, and the rate a *correct*
#: implementation produces varies along the epsilon axis (see the module docstring), so
#: a fixed cap would be a guess with the power to fail a correct implementation — the
#: exact false positive that made ``ab_kv_cache.py`` reject DP-RAG. The runs this script
#: performs are what will supply the number.
MAX_TOKEN_FLIP_RATE = None

#: Fail above this fraction of answers that disagree on their *first* token. Unlike the
#: rate above, this one can be argued without a measurement. Both arms invert the same
#: seeded uniform, so their per-step disagreement probability is the L1 distance between
#: the two CDFs, which is linear in a ~1e-6 perturbation and cannot approach 1 at any
#: temperature. Under a per-step hazard h the fraction of answers differing at the first
#: token is exactly h, so near-total first-token divergence means h ~ 1: not a perturbed
#: shared computation but two different ones.
MAX_FIRST_STEP_DIVERGENCE = 0.95

#: Below this many compared answers the first-token fraction is too coarse to separate
#: "nearly all" from "several in a row", so the gate above is not applied.
MIN_QUERIES_FOR_FIRST_STEP = 20

#: Significance level for the unbiasedness test. Applied to the pooled McNemar; the
#: per-cell tests divide it by the number of tests so that adding cells cannot
#: manufacture a failure.
MCNEMAR_ALPHA = 0.05

#: An exact two-sided McNemar on n discordant pairs cannot go below 2/2^n, so below six
#: pairs no split at all reaches 0.05. Passing on a test that could not have failed is
#: not a pass; the run is reported INCONCLUSIVE instead.
MIN_DISCORDANT = 6

#: Printed wherever :data:`METRIC_TOLERANCE` is used, because the band is weaker than it
#: looks and a reader should not have to dig for that.
NOISE_BAND_CAVEAT = (
    "The metric tolerances are NOT a measured seed-to-seed spread: this repo has never "
    "run repeated seeds, so no such figure exists. They come from 23 configurations "
    "that happened to be executed twice across separate runs (dpksa run_20260702_114726 "
    "vs run_20260712_094517, n=40; dp_synrag run_20260710_100921 vs "
    "run_20260712_224716_maquinatoro, n=102-200), which differ by run-time randomness "
    "AND by whatever drifted between the two dates — so they bound the noise from above "
    "rather than measure it. Treat the band as a backstop against a shift too large to "
    "be a handful of ties; the paired test above is what makes the verdict sharp."
)

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


def switches_are_wired() -> dict[str, list[str]]:
    """Files that read each switch, so the harness can refuse a sham A/B.

    Two arms that both take the fast path agree perfectly and prove nothing, and an
    unread environment variable is exactly how that happens silently.
    """
    sources: list[tuple[str, str]] = []
    for sub in ("core", "dp_methods", "experiments", "vanilla", "metrics"):
        root = PROJECT_ROOT / sub
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            try:
                sources.append((str(path.relative_to(PROJECT_ROOT)),
                                path.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                continue
    return {
        key: sorted(rel for rel, text in sources if env in text)
        for key, env in SWITCHES.items()
    }


# ------------------------------------------------------------------------------ config


def _restrict_grid(cfg: dict, epsilons: list[float] | None, workloads: list[int] | None) -> dict:
    """A copy of *cfg* with the shared axis cut down to the points the A/B needs.

    The full DP-RAG config sweeps four epsilon targets by four workload sizes, which is
    sixteen runs of a loop this script has to run twice. What may *not* be cut is the
    span of the epsilon axis: the clip binds on every shipped cell, so the exponential
    mechanism's sampling distribution goes from near-uniform at the tight end to peaked
    at the loose one (see "Which epsilon" in the module docstring), and a perturbation
    that is harmless in one regime has not been shown to be harmless in the other. Both
    ends run; the interior is what is dropped.

    Keys the config does not declare are left absent rather than invented: the control
    method has no privacy axis, and writing one into its config would make it sweep
    something it does not understand.
    """
    out = json.loads(json.dumps(cfg))
    if epsilons is not None and "epsilon_targets" in out:
        out["epsilon_targets"] = list(epsilons)
    if workloads is not None and "workload_queries" in out:
        out["workload_queries"] = [int(q) for q in workloads]
    return out


def _materialise_config(
    method: str,
    config_key: str,
    dest: Path,
    *,
    queries: int | None,
    index_docs: int | None,
    model: str | None,
    epsilons: list[float] | None,
    workloads: list[int] | None,
) -> tuple[str, dict]:
    """Write this A/B's config for *method* and return ``(path, parsed)``.

    Both arms read the same file, so the two runs cannot differ by a config edit made
    between them.
    """
    src = PROJECT_ROOT / METHODS[method][config_key]
    with open(src, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg = _set_queries(cfg, queries, index_docs, model)
    cfg = _restrict_grid(cfg, epsilons, workloads)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{method}.yaml"
    header = (
        f"# GENERATED by scripts/ab_dp_rag_batching.py from {src.relative_to(PROJECT_ROOT)}\n"
        f"# Both arms of the DP-RAG inner-loop A/B read this file.\n"
    )
    path.write_text(
        header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return str(path), cfg


# -------------------------------------------------------------------------------- runs


def _run_arm(
    label: str,
    disabled: list[str],
    jobs: list[tuple[str, str, list[str]]],
    out_root: Path,
    log_root: Path,
    manifest: str,
    seed: int,
) -> dict:
    """Run every job once with the optimisation switches held at one value.

    The switches travel through the process environment because that is how the whole
    orchestration layer already talks to its method subprocesses; ``run_method`` copies
    ``os.environ``, so setting them here reaches every child.
    """
    for key, env in SWITCHES.items():
        if key in disabled:
            os.environ[env] = "1"
        else:
            os.environ.pop(env, None)

    print("\n" + "#" * 64)
    print(f"ARM {label}")
    for key, env in SWITCHES.items():
        print(f"  {env}={os.environ.get(env, '<unset>')}")
    print("#" * 64)

    rows: list[dict] = []
    t0 = time.time()
    for method, config_path, datasets in jobs:
        for ds in datasets:
            rows.append(
                run_method(
                    method,
                    METHODS[method]["script"],
                    config_path,
                    output_dir=str(out_root / method / f"seed_{seed}"),
                    log_dir=str(log_root),
                    only_dataset=ds,
                    index_manifest=manifest,
                    run_seed=seed,
                )
            )
    wall = time.time() - t0

    for env in SWITCHES.values():
        os.environ.pop(env, None)
    return {"label": label, "disabled": list(disabled), "wall_sec": wall, "methods": rows}


# --------------------------------------------------------------------------- comparing


def _retrieved_ids(data: dict) -> list[list]:
    """Per-query retrieved document ids, empty when the method records none."""
    return [list(it.get("retrieved_ids") or []) for it in _items(data)]


def _is_control(rel: str, data: dict) -> bool:
    """Whether a cell belongs to the control method rather than to DP-RAG."""
    method = str(data.get("method") or "")
    return method == CONTROL_METHOD or rel.split("/")[0] == CONTROL_METHOD


def _compare_cell(rel: str, fast: dict, slow: dict) -> dict:
    """One cell's worth of comparison: structure, budget, retrieval, flips, metrics, cost."""
    cell: dict = {"file": rel, "errors": [], "warnings": [], "control": _is_control(rel, fast)}

    p_fast, p_slow = _predictions(fast), _predictions(slow)
    ids_fast, ids_slow = _query_ids(fast), _query_ids(slow)
    if ids_fast and ids_slow and ids_fast != ids_slow:
        cell["errors"].append("the two arms answered different questions")
    if len(p_fast) != len(p_slow):
        cell["errors"].append(f"different answer counts: {len(p_fast)} vs {len(p_slow)}")

    n = min(len(p_fast), len(p_slow))
    cell["n_queries"] = n
    identical = sum(1 for a, b in zip(p_fast[:n], p_slow[:n]) if a == b)
    cell["identical_answers"] = identical
    # Reported for orientation only, and gated only on the control. One flipped token
    # rewrites the rest of the answer, so for DP-RAG this measures the amplification,
    # not the perturbation.
    cell["identical_answer_fraction"] = identical / n if n else None
    cell["flipped_indices"] = [i for i in range(n) if p_fast[i] != p_slow[i]]

    if cell["control"] and cell["flipped_indices"]:
        cell["errors"].append(
            f"the control changed {len(cell['flipped_indices'])}/{n} answers — it reads "
            "none of the switches, so the two arms differ by something other than the "
            "change under test and nothing here is attributable"
        )

    # Per decoding step, and how many answers parted company at the very first one.
    agree = flips = first_step = 0
    for i in range(n):
        a, f = _first_divergence(p_fast[i], p_slow[i])
        agree += a
        flips += f
        if f and a == 0:
            first_step += 1
    steps = agree + flips
    cell["token_steps_compared"] = steps
    cell["token_flips"] = flips
    cell["token_flip_rate"] = (flips / steps) if steps else None
    cell["first_step_divergences"] = first_step
    cell["first_step_divergence_rate"] = (first_step / n) if n else None

    # Retrieval is upstream of the generation loop and must be untouched by it.
    r_fast, r_slow = _retrieved_ids(fast), _retrieved_ids(slow)
    if r_fast and r_slow:
        moved = sum(1 for a, b in zip(r_fast[:n], r_slow[:n]) if a != b)
        cell["retrieval_differs_on"] = moved
        if moved:
            cell["errors"].append(
                f"the retrieved document ids differ on {moved}/{n} queries — this is not "
                "a generation-only A/B and every number in this cell is confounded"
            )
    else:
        cell["retrieval_differs_on"] = None

    # Budget: exact, or it is a bug.
    budget: dict = {}
    for key in BUDGET_FIELDS:
        a, b = fast.get(key), slow.get(key)
        if a is None and b is None:
            continue
        budget[key] = {"optimised": a, "baseline": b, "equal": a == b}
        if a != b:
            cell["errors"].append(
                f"epsilon field {key} moved: {a!r} (optimised) vs {b!r} (baseline)"
            )
    cell["budget"] = budget

    spent: dict = {}
    for key in SPENT_FIELDS:
        a, b = fast.get(key), slow.get(key)
        if a is None and b is None:
            continue
        spent[key] = {"optimised": a, "baseline": b, "equal": a == b}
    cell["spent"] = spent

    # Paired correctness over the queries whose text moved. Computed on single-item lists
    # so it goes through the repo's own normalisation (SQuAD normalisation, alias max) and
    # cannot disagree with the aggregate it sits next to.
    golds = _golds(fast)
    flip_balance: dict[str, dict] = {}
    for name, fn in (("match_accuracy", match_accuracy), ("contains_accuracy", contains_accuracy)):
        gained = lost = same = 0
        for i in cell["flipped_indices"]:
            if i >= len(golds):
                continue
            a = fn([p_fast[i]], [golds[i]])
            b = fn([p_slow[i]], [golds[i]])
            if a > b:
                gained += 1
            elif a < b:
                lost += 1
            else:
                same += 1
        flip_balance[name] = {
            "optimised_gained": gained,
            "optimised_lost": lost,
            "flipped_but_equally_correct": same,
            "mcnemar_p": _mcnemar_p(gained, lost),
        }
    cell["flip_balance"] = flip_balance

    m_fast = fast.get("metrics") or {}
    m_slow = slow.get("metrics") or {}
    deltas: dict = {}
    for key in sorted(set(m_fast) | set(m_slow)):
        a, b = m_fast.get(key), m_slow.get(key)
        if not isinstance(a, (int, float)) or isinstance(a, bool):
            continue
        if not isinstance(b, (int, float)) or isinstance(b, bool):
            continue
        if a != a or b != b:  # NaN
            continue
        tol = METRIC_TOLERANCE.get(key)
        deltas[key] = {
            "optimised": a,
            "baseline": b,
            "delta": a - b,
            "abs_delta": abs(a - b),
            "tolerance": tol,
            "within": None if tol is None else abs(a - b) <= tol,
        }
    cell["metrics"] = deltas

    f_fast, f_slow = fast.get(FORWARDS_FIELD), slow.get(FORWARDS_FIELD)
    if isinstance(f_fast, (int, float)) and isinstance(f_slow, (int, float)) and f_fast > 0:
        cell["forwards_per_query"] = {
            "optimised": f_fast,
            "baseline": f_slow,
            "ratio": f_slow / f_fast,
            "docs_per_query": fast.get("n_docs_per_query"),
        }
    t_fast, t_slow = fast.get("time_per_query_sec"), slow.get("time_per_query_sec")
    if isinstance(t_fast, (int, float)) and isinstance(t_slow, (int, float)) and t_fast > 0:
        cell["time_per_query_sec"] = {
            "optimised": t_fast,
            "baseline": t_slow,
            "speedup": t_slow / t_fast,
        }
    return cell


def compare(
    optimised_root: Path,
    baseline_root: Path,
    max_token_flip_rate: float = MAX_TOKEN_FLIP_RATE,
    *,
    expect_control: bool = True,
    evidence: str = "forwards",
) -> dict:
    """The whole comparison, plus the verdict it implies."""
    fast, slow = _payloads(optimised_root), _payloads(baseline_root)
    only_fast = sorted(set(fast) - set(slow))
    only_slow = sorted(set(slow) - set(fast))
    shared = sorted(set(fast) & set(slow))

    cells = [_compare_cell(rel, fast[rel], slow[rel]) for rel in shared]
    treated = [c for c in cells if not c["control"]]
    controls = [c for c in cells if c["control"]]

    errors: list[str] = []
    warnings: list[str] = []
    if only_fast or only_slow:
        errors.append(
            f"the arms produced different cells: {len(only_fast)} only optimised, "
            f"{len(only_slow)} only baseline"
        )
    if not shared:
        errors.append("no cell was produced by both arms")
    if not treated and shared:
        errors.append(f"no {METHOD} cell was produced by both arms")

    for cell in cells:
        errors.extend(f"{cell['file']}: {e}" for e in cell["errors"])

    # The unbiasedness test, pooled over the treated cells because that is where the
    # power is. The per-cell tests below are Bonferroni-corrected against the number of
    # tests actually run, so adding cells cannot manufacture a failure.
    pooled: dict[str, dict] = {}
    for name in ("match_accuracy", "contains_accuracy"):
        gained = sum(c["flip_balance"][name]["optimised_gained"] for c in treated)
        lost = sum(c["flip_balance"][name]["optimised_lost"] for c in treated)
        pooled[name] = {
            "optimised_gained": gained,
            "optimised_lost": lost,
            "discordant": gained + lost,
            "mcnemar_p": _mcnemar_p(gained, lost),
        }
        p = pooled[name]["mcnemar_p"]
        if p is not None and p < MCNEMAR_ALPHA:
            errors.append(
                f"pooled {name} moved one way — the optimised path gained {gained} and "
                f"lost {lost} over the queries whose answer changed (McNemar "
                f"p={p:.4f} < {MCNEMAR_ALPHA})"
            )

    n_tests = sum(
        1
        for c in treated
        for bal in c["flip_balance"].values()
        if bal["mcnemar_p"] is not None
    )
    per_cell_alpha = MCNEMAR_ALPHA / n_tests if n_tests else MCNEMAR_ALPHA
    for cell in treated:
        for name, bal in cell["flip_balance"].items():
            p = bal["mcnemar_p"]
            if p is not None and p < per_cell_alpha:
                errors.append(
                    f"{cell['file']}: {name} moved one way — the optimised path gained "
                    f"{bal['optimised_gained']} and lost {bal['optimised_lost']} "
                    f"(McNemar p={p:.4f} < {per_cell_alpha:.4f}, Bonferroni over "
                    f"{n_tests} tests)"
                )
        rate = cell["token_flip_rate"]
        if max_token_flip_rate is not None and rate is not None and rate > max_token_flip_rate:
            errors.append(
                f"{cell['file']}: {rate:.1%} of decoding steps sampled a different token "
                f"(cap {max_token_flip_rate:.1%}, set explicitly) — too many to be "
                "near-ties resolved differently"
            )
        for name, d in cell["metrics"].items():
            if d["within"] is False:
                errors.append(
                    f"{cell['file']}: {name} moved {d['abs_delta']:.4f} "
                    f"(tolerance {d['tolerance']})"
                )

    # The regime-free gate: a hazard of ~1 is not a perturbation of a shared computation.
    first_step = sum(c["first_step_divergences"] for c in treated)
    first_step_of = sum(c["n_queries"] for c in treated)
    first_step_rate = (first_step / first_step_of) if first_step_of else None
    if (
        first_step_of >= MIN_QUERIES_FOR_FIRST_STEP
        and first_step_rate is not None
        and first_step_rate > MAX_FIRST_STEP_DIVERGENCE
    ):
        errors.append(
            f"{first_step}/{first_step_of} answers disagree on their very first token "
            f"({first_step_rate:.0%} > {MAX_FIRST_STEP_DIVERGENCE:.0%}). Both arms invert "
            "the same seeded uniform, so a ~1e-6 perturbation cannot move the sampled "
            "token at nearly every step at any temperature: the two paths are computing "
            "different things, not the same thing in a different order"
        )

    ratios = sorted(c["forwards_per_query"]["ratio"] for c in treated if "forwards_per_query" in c)
    median_ratio = ratios[len(ratios) // 2] if ratios else None
    speedups = sorted(
        c["time_per_query_sec"]["speedup"] for c in treated if "time_per_query_sec" in c
    )
    median_speedup = speedups[len(speedups) // 2] if speedups else None

    inconclusive: list[str] = []
    if evidence == "forwards":
        if not ratios:
            inconclusive.append(
                f"no cell recorded {FORWARDS_FIELD}, so there is no evidence the two arms "
                "took different paths"
            )
        elif median_ratio is not None and median_ratio < MIN_FORWARD_RATIO:
            inconclusive.append(
                f"median forward-count ratio {median_ratio:.2f}x is below "
                f"{MIN_FORWARD_RATIO}x: the baseline arm made about as many model calls as "
                "the optimised one, so the switches "
                f"({', '.join(SWITCHES.values())}) were probably not honoured and both "
                "arms ran the same path"
            )
    elif not speedups:
        inconclusive.append(
            "no cell recorded time_per_query_sec, and this knob does not change the "
            "forward count, so there is no evidence the two arms took different paths"
        )
    elif median_speedup is not None and median_speedup < MIN_SPEEDUP:
        inconclusive.append(
            f"median per-query speedup {median_speedup:.3f}x is below {MIN_SPEEDUP}x: this "
            "knob leaves the forward count alone, so the clock is the only evidence it was "
            "honoured, and it says the two arms ran the same path"
        )
    if expect_control and not controls:
        inconclusive.append(
            f"no {CONTROL_METHOD} cell was compared, so a harness measuring itself — a "
            "non-deterministic machine, a rebuilt index, a seed that did not reach a "
            "subprocess — would not have been detectable"
        )
    total_flipped = sum(len(c["flipped_indices"]) for c in treated)
    for name, agg in pooled.items():
        if total_flipped and agg["discordant"] < MIN_DISCORDANT:
            inconclusive.append(
                f"{name}: only {agg['discordant']} discordant pairs over {total_flipped} "
                f"changed answers — an exact McNemar needs {MIN_DISCORDANT} before any "
                "split can reach p<0.05, so this test could not have failed. Raise "
                "--queries (or add a dataset) before reading a pass from it"
            )

    total_steps = sum(c["token_steps_compared"] for c in treated)
    if total_steps < 100:
        warnings.append(
            f"only {total_steps} decoding steps compared: the flip rate has no resolution"
        )

    verdict = "FAIL" if errors else ("INCONCLUSIVE" if inconclusive else "PASS")

    total_q = sum(c["n_queries"] for c in treated)
    total_flips = sum(c["token_flips"] for c in treated)
    return {
        "verdict": verdict,
        "errors": errors,
        "inconclusive_reasons": inconclusive,
        "warnings": warnings,
        "cells_compared": len(cells),
        "control_cells_compared": len(controls),
        "cells_only_optimised": only_fast,
        "cells_only_baseline": only_slow,
        "queries_compared": total_q,
        "answers_changed": total_flipped,
        "token_steps_compared": total_steps,
        "token_flips": total_flips,
        "token_flip_rate": (total_flips / total_steps) if total_steps else None,
        "first_step_divergences": first_step,
        "first_step_divergence_rate": first_step_rate,
        "pooled_flip_balance": pooled,
        "median_forward_ratio": median_ratio,
        "median_per_query_speedup": median_speedup,
        "criterion": {
            "budget_fields_exact": list(BUDGET_FIELDS),
            "retrieved_ids_exact": True,
            "control_method_byte_identical": CONTROL_METHOD,
            "metric_tolerance": METRIC_TOLERANCE,
            "metric_tolerance_caveat": NOISE_BAND_CAVEAT,
            "max_token_flip_rate": max_token_flip_rate,
            "token_flip_rate_gated": max_token_flip_rate is not None,
            "max_first_step_divergence": MAX_FIRST_STEP_DIVERGENCE,
            "mcnemar_alpha": MCNEMAR_ALPHA,
            "mcnemar_alpha_per_cell": per_cell_alpha,
            "min_discordant_pairs": MIN_DISCORDANT,
            "switch_honoured_evidence": evidence,
            "min_forward_ratio": MIN_FORWARD_RATIO,
            "min_speedup": MIN_SPEEDUP,
        },
        "cells": cells,
    }


# ----------------------------------------------------------------------------- reports


def render(summary: dict, arms: list[dict] | None) -> str:
    """The human-readable half of the output, verdict first."""
    L: list[str] = []
    L.append("=" * 72)
    L.append(f"DP-RAG INNER-LOOP A/B — VERDICT: {summary['verdict']}")
    L.append("=" * 72)
    for line in summary["errors"]:
        L.append(f"  FAIL  {line}")
    for line in summary["inconclusive_reasons"]:
        L.append(f"  ????  {line}")
    for line in summary["warnings"]:
        L.append(f"  warn  {line}")
    if summary["verdict"] == "PASS":
        L.append("  Epsilon and retrieval identical everywhere; the control reproduced")
        L.append("  itself byte for byte; the arms still share the first sampled token on")
        L.append("  most answers and what moved is unbiased; every gated metric is inside")
        L.append("  the repeated-run band.")
    L.append("")

    rate = summary["token_flip_rate"]
    L.append(f"Cells compared        : {summary['cells_compared']} "
             f"({summary['control_cells_compared']} control)")
    L.append(f"Queries compared      : {summary['queries_compared']}")
    L.append(
        "Token flip rate       : "
        + (f"{rate:.4f}" if rate is not None else "n/a")
        + f"  ({summary['token_flips']}/{summary['token_steps_compared']} decoding steps)"
        + ("" if summary["criterion"]["token_flip_rate_gated"] else
           "  [reported, not gated — no measured null, and it varies along the epsilon axis]")
    )
    fsr = summary["first_step_divergence_rate"]
    L.append(
        "First-token disagree  : "
        + (f"{fsr:.4f}" if fsr is not None else "n/a")
        + f"  ({summary['first_step_divergences']}/{summary['queries_compared']} answers, "
        + f"cap {summary['criterion']['max_first_step_divergence']:.0%})"
    )
    L.append(
        f"Answers changed       : {summary['answers_changed']}/{summary['queries_compared']}"
        "  (informational — one flip rewrites the rest)"
    )
    for name, agg in summary["pooled_flip_balance"].items():
        p = agg["mcnemar_p"]
        L.append(
            f"Pooled {name:<17}: +{agg['optimised_gained']} / -{agg['optimised_lost']} "
            f"on changed answers, McNemar p={'n/a' if p is None else f'{p:.4f}'}"
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
        L.append(f"  {cell['file']}" + ("   [CONTROL]" if cell["control"] else ""))
        parts = [
            f"    steps {cell['token_steps_compared']}, flips {cell['token_flips']}"
            + (
                f" ({cell['token_flip_rate']:.4f})"
                if cell["token_flip_rate"] is not None
                else ""
            )
            + f", first-token {cell['first_step_divergences']}/{cell['n_queries']}"
        ]
        if "forwards_per_query" in cell:
            fw = cell["forwards_per_query"]
            parts.append(
                f"   forwards/query {fw['baseline']:.0f} -> {fw['optimised']:.0f} "
                f"({fw['ratio']:.2f}x, {fw['docs_per_query'] or 0:.0f} docs)"
            )
        if "time_per_query_sec" in cell:
            parts.append(f"   speedup {cell['time_per_query_sec']['speedup']:.3f}x")
        L.append("".join(parts))
        L.append(
            f"    whole answers identical {cell['identical_answers']}/{cell['n_queries']}"
            + ("" if cell["control"] else " (informational — one flip rewrites the rest)")
        )
        if cell["retrieval_differs_on"] is not None:
            L.append(
                "    retrieval: IDENTICAL"
                if not cell["retrieval_differs_on"]
                else f"    retrieval: DIFFERS on {cell['retrieval_differs_on']} queries"
                "   <-- confounded"
            )
        bad_budget = [k for k, v in cell["budget"].items() if not v["equal"]]
        L.append(
            "    epsilon: EXACT"
            if not bad_budget
            else f"    epsilon: DIFFERS on {bad_budget}   <-- bug, not noise"
        )
        for k, v in cell["spent"].items():
            if not v["equal"]:
                L.append(f"    spent {k}: {v['optimised']} vs {v['baseline']} (informational)")
        for name, bal in cell["flip_balance"].items():
            p = bal["mcnemar_p"]
            L.append(
                f"    {name}: +{bal['optimised_gained']} / -{bal['optimised_lost']} on "
                f"flips, McNemar p={'n/a' if p is None else f'{p:.4f}'}"
            )
        for name, d in sorted(cell["metrics"].items()):
            mark = {True: "ok", False: "OVER", None: "--"}[d["within"]]
            L.append(
                f"    {name:<26} {d['optimised']:>10.4f}  {d['baseline']:>10.4f}  "
                f"d={d['delta']:+.4f}  [{mark}]"
            )
        L.append("")

    L.append("On the metric tolerances")
    L.append("-" * 72)
    for line in _wrap(NOISE_BAND_CAVEAT, 72):
        L.append("  " + line)
    return "\n".join(L)


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap, so the caveat reads as prose in a fixed-width log."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def describe_plan(plan: dict) -> str:
    L = ["=" * 72, "DP-RAG INNER-LOOP A/B — PLAN", "=" * 72]
    for key in ("config_set", "seed", "queries", "index_docs", "model", "epsilon_targets",
                "workload_queries", "datasets", "control", "output"):
        L.append(f"  {key:<16}: {plan[key]}")
    L.append("")
    L.append("  Arm 1  baseline  (runs first — it is what results/ was produced with):")
    for env in SWITCHES.values():
        L.append(f"      {env}=" + ("1" if env in plan["baseline_env"] else "<unset>"))
    L.append("  Arm 2  optimised (what the grid will run):")
    for env in SWITCHES.values():
        L.append(f"      {env}=<unset>")
    L.append("")
    L.append("  Both arms: same config file, same data_seed, same BENCHDP_RUN_SEED, one")
    L.append("  shared index (BENCHDP_RUN_ID), scored afterwards by score_results.py.")
    L.append("")
    L.append("  Switch wired in:")
    for key, files in plan["switches_wired_in"].items():
        L.append(f"    {SWITCHES[key]:<32} {files or 'NOWHERE YET'}")
    L.append("")
    L.append("  Expected cost: the baseline arm is the slow one (~62 s/query measured on")
    L.append("  chatdoctor / facebook/opt-1.3b / 20k docs / RTX 3090), the optimised arm")
    L.append("  ~8 s/query, plus one index build shared by both. Multiply by the number")
    L.append(f"  of cells: {len(plan['epsilon_targets'])} epsilon target(s) x "
             f"{len(plan['workload_queries'])} workload(s) x the datasets above.")
    return "\n".join(L)


# -------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-set", choices=["smoke", "full", "paper_faithful"],
                        default="full", help="which config directory the cells come from")
    parser.add_argument("--datasets", nargs="+", default=["chatdoctor"],
                        help="datasets to run (default: chatdoctor, the cell the speedup "
                             "was measured on; 'all' takes every one in the config)")
    parser.add_argument("--seed", type=int, default=1,
                        help="the single run_seed both arms use (default: 1)")
    parser.add_argument("--queries", type=int, default=50,
                        help="questions per cell; the paired test needs enough changed "
                             "answers to have power (default: 50, 0 = leave the config alone)")
    parser.add_argument("--index-docs", type=int, default=None,
                        help="override every cell's corpus size (default: the config's)")
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b",
                        help="the single generator both arms use (default: facebook/opt-1.3b)")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1.0, 50.0],
                        help="epsilon targets to run. The sampling distribution is "
                             "near-uniform at the tight end and peaked at the loose one, "
                             "so both ends of the shipped axis run by default (1 and 50) "
                             "and neither is representative of the other")
    parser.add_argument("--workloads", nargs="+", type=int, default=[100],
                        help="workload sizes to run (default: 100)")
    parser.add_argument("--isolate", choices=["all", *SWITCHES], default="all",
                        help="which optimisation the baseline arm turns off; 'all' is the "
                             "comparison that needs to pass, the rest are for bisecting a "
                             "failure and are checked against the clock rather than the "
                             "forward count")
    parser.add_argument("--no-control", action="store_true",
                        help=f"skip the {CONTROL_METHOD} control arm (not recommended: it "
                             "is what detects a harness measuring itself)")
    parser.add_argument("--max-token-flip-rate", type=float, default=MAX_TOKEN_FLIP_RATE,
                        help="fail above this fraction of flipped decoding steps. Off by "
                             "default: the rate a correct implementation produces has "
                             "never been measured here and depends on where the cell sits "
                             "on the epsilon axis, so a guessed cap could only fail a "
                             "correct implementation. First-token disagreement is gated "
                             "instead")
    parser.add_argument("--out", type=str, default=None,
                        help="artifact directory (default: results/ab_dp_rag_batching_<stamp>)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit; runs nothing, loads no model")
    parser.add_argument("--allow-unwired", action="store_true",
                        help="run even though a switch is read by nothing")
    parser.add_argument("--compare-only", nargs=2, metavar=("OPTIMISED_DIR", "BASELINE_DIR"),
                        help="skip both runs and re-compare two existing result trees")
    args = parser.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (
        Path(args.out) if args.out
        else PROJECT_ROOT / "results" / f"ab_dp_rag_batching_{stamp}"
    )

    if args.compare_only:
        fast_root, slow_root = (Path(p) for p in args.compare_only)
        for p in (fast_root, slow_root):
            if not p.is_dir():
                print(f"not a directory: {p}")
                return 2
        summary = compare(
            fast_root, slow_root, args.max_token_flip_rate,
            expect_control=not args.no_control,
            evidence=EVIDENCE_FOR_ISOLATE[args.isolate],
        )
        report = render(summary, arms=None)
        print(report)
        out_root.mkdir(parents=True, exist_ok=True)
        _write_artifacts(out_root, summary, report)
        return 0 if summary["verdict"] == "PASS" else 1

    config_key = {"smoke": "smoke_config", "full": "full_config",
                  "paper_faithful": "paper_faithful_config"}[args.config_set]
    queries = args.queries or None
    methods = [METHOD] if args.no_control else [METHOD, CONTROL_METHOD]

    jobs: list[tuple[str, str, list[str]]] = []
    for method in methods:
        path, cfg = _materialise_config(
            method, config_key, out_root / "configs",
            queries=queries, index_docs=args.index_docs, model=args.model,
            epsilons=args.epsilons, workloads=args.workloads,
        )
        available = dataset_names(cfg)
        wanted = available if args.datasets == ["all"] else [
            d for d in args.datasets if d in available
        ]
        jobs.append((method, path, wanted))

    missing = [m for m, _, ds in jobs if not ds]
    baseline_env = (
        [SWITCHES[k] for k in SWITCHES] if args.isolate == "all" else [SWITCHES[args.isolate]]
    )
    wired = switches_are_wired()
    plan = {
        "config_set": args.config_set,
        "seed": args.seed,
        "queries": queries or "(config)",
        "index_docs": args.index_docs or "(config)",
        "model": args.model or "(config)",
        "epsilon_targets": args.epsilons,
        "workload_queries": args.workloads,
        "datasets": {m: ds for m, _, ds in jobs},
        "control": "(disabled)" if args.no_control else CONTROL_METHOD,
        "output": str(out_root),
        "isolate": args.isolate,
        "baseline_env": baseline_env,
        "switches_wired_in": wired,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(describe_plan(plan))
    if missing:
        print(f"\nno requested dataset is declared by: {missing}")
        return 2

    if args.dry_run:
        print("\n--dry-run: nothing was executed.")
        return 0

    unwired = [SWITCHES[k] for k in SWITCHES if not wired[k] and SWITCHES[k] in baseline_env]
    if unwired and not args.allow_unwired:
        print(
            f"\nREFUSING TO RUN: no file under core/, dp_methods/, experiments/, vanilla/ or\n"
            f"metrics/ reads {unwired}. Both arms would take the identical code path for\n"
            f"that optimisation and agree perfectly, which proves nothing.\n\n"
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
        # The baseline arm runs first: it is the current behaviour, so if the machine
        # cannot finish even that, nothing about the change has been risked.
        arms.append(_run_arm("baseline", list(SWITCHES) if args.isolate == "all"
                             else [args.isolate], jobs,
                             out_root / "baseline", log_root, manifest, args.seed))
        arms.append(_run_arm("optimised", [], jobs,
                             out_root / "optimised", log_root, manifest, args.seed))
    finally:
        try:
            with open(manifest, encoding="utf-8") as fh:
                for d in {ln.strip() for ln in fh if ln.strip()}:
                    shutil.rmtree(d, ignore_errors=True)
            os.remove(manifest)
        except OSError:
            pass

    failed = [
        f"{a['label']}/{r['method']}/{r['dataset']}"
        for a in arms
        for r in a["methods"]
        if not r["ok"]
    ]

    print("\n" + "=" * 64)
    print("SCORING both arms (offline metrics)")
    print("=" * 64)
    for arm in arms:
        rc = _score(out_root / arm["label"])
        if rc != 0:
            print(f"  WARNING: scoring {arm['label']} failed (rc={rc})")

    summary = compare(
        out_root / "optimised", out_root / "baseline", args.max_token_flip_rate,
        expect_control=not args.no_control,
        evidence=EVIDENCE_FOR_ISOLATE[args.isolate],
    )
    summary["plan"] = plan
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
    _write_artifacts(out_root, summary, report)
    return 0 if summary["verdict"] == "PASS" else 1


def _write_artifacts(out_root: Path, summary: dict, report: str) -> None:
    """Both halves of the output, machine-readable and human-readable, side by side."""
    summary_path = out_root / "ab_dp_rag_batching_summary.json"
    report_path = out_root / "ab_dp_rag_batching_report.txt"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    print(f"Summary : {summary_path}")
    print(f"Report  : {report_path}")


if __name__ == "__main__":
    sys.exit(main())
