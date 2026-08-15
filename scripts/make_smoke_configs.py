#!/usr/bin/env python
"""Derive ``experiments/configs/smoke/`` from ``experiments/configs/full/``.

The smoke set is the common protocol shrunk until a full eight-method pass finishes
in minutes: one dataset, a few hundred documents, four questions, a small generator,
two points on each budget knob. It exists to answer whether every method executes end
to end, writes the fields the analysis reads, and responds to the seeds the way it
claims -- questions about code paths, which do not need a realistic corpus.

What it deliberately cannot answer is how long the real grid takes or where a
retrieval threshold should sit. Both depend on corpus size, and measuring them here
would size the main grid from a fraction of its work.

Generating it rather than hand-writing it keeps the two sets from drifting: every
structural parameter (thresholds, voter counts, chunking policy, metric lists) comes
from the full config verbatim, and only the axes below are overridden.

    python scripts/make_smoke_configs.py
    python experiments/runner.py --smoke --seeds 1 2
    python scripts/verify_smoke.py results/run_<stamp>

The budget points are the method's operating region, never the tightest one on its
grid: at the low end several methods legitimately release nothing, and a smoke run
must not confuse an empty cell with a broken runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_DIR = PROJECT_ROOT / "experiments" / "configs" / "full"
SMOKE_DIR = PROJECT_ROOT / "experiments" / "configs" / "smoke"

# ChatDoctor: self-contained corpus, so the smoke run does not depend on the DPR
# passages being downloaded, and short documents keep the voting prompts inside
# OPT's 2048 positions.
DATASET = "chatdoctor"
# Small enough to run the whole protocol in minutes. This set answers "does every
# method execute end to end, write the fields the analysis reads, and respond to the
# seeds the way it claims", which needs coverage of the code paths, not a realistic
# corpus. Timing and any threshold calibration must be measured at the protocol's own
# size: at a fraction of the corpus the retrieval thresholds admit far fewer documents
# and the voting methods run on a fraction of the forward passes, so a per-query time
# measured here would undersize the main grid rather than size it.
MAX_INDEX_DOCS = 500
MAX_TEST_QUERIES = 4
# The smallest real generator that still produces text. Quality is not what this run
# measures, and a 82M model keeps a full eight-method pass inside a coffee break.
MODEL = "distilgpt2"

# Two points of the protocol's own shared axis (full/*.yaml `epsilon_targets`), taken
# rather than invented: a smoke cell then sits on a coordinate the main grid also runs,
# so a target the calibration cannot reach shows up here instead of at hour three of the
# real run. The loose end of it, because at epsilon=1 several methods legitimately
# release nothing and an empty cell must not be mistaken for a broken runner.
SHARED_AXIS_TARGETS = [10, 50]
# Two workload sizes, for the same reason there are two budget points: Q is the second
# axis of the protocol, the per-query budget and every workload-derived delta are
# functions of it, and a single value would leave that derivation untested. Kept at the
# short end so each question still gets enough budget to release something.
WORKLOAD_QUERIES = [1, 5]

# One entry per budget knob, at each method's operating point.
EPSILON_POINTS: dict[str, dict[str, list[float]]] = {
    "dp_rag": {
        "epsilon_targets": SHARED_AXIS_TARGETS,
        "workload_queries": WORKLOAD_QUERIES,
    },
    "dpvoterag": {
        "epsilon_targets": SHARED_AXIS_TARGETS,
        "workload_queries": WORKLOAD_QUERIES,
    },
    "dpsparsevoterag": {
        "epsilon_targets": SHARED_AXIS_TARGETS,
        "workload_queries": WORKLOAD_QUERIES,
    },
    "dpksa": {
        "epsilon_targets": SHARED_AXIS_TARGETS,
        "workload_queries": WORKLOAD_QUERIES,
    },
    # The per-token budget follows from the target under the shared protocol, so the
    # axis is the only knob to move.
    "private_rag": {
        "epsilon_targets": SHARED_AXIS_TARGETS,
        "workload_queries": WORKLOAD_QUERIES,
    },
    "private_rag_ada": {
        "epsilon_targets": SHARED_AXIS_TARGETS,
        "workload_queries": WORKLOAD_QUERIES,
    },
    # DP-SynRAG names points on the shared axis and derives all four stage parameters
    # from each one. Far looser than the protocol's targets, because the noise scales
    # are absolute while the histogram they perturb is not: over a few hundred
    # documents the protocol's sigma_h swamps every count, no cluster survives the
    # threshold, and the run exercises only the empty-corpus fallback. These points
    # are chosen so a corpus actually forms, which is what there is to verify here.
    "dp_synrag": {
        "synthetic_generation.epsilon_targets": [300],
        "workload_queries": WORKLOAD_QUERIES,
    },
    # The two reference conditions price no mechanism, so they have no budget knob and
    # no workload axis to move along.
    "vanilla": {},
    "no_rag": {},
}

HEADER = """\
# GENERATED by scripts/make_smoke_configs.py -- do not edit by hand.
#
# The common protocol (experiments/configs/full/{name}.yaml) shrunk to a few minutes:
# {dataset}, {model}, {queries} questions, two points per budget knob. Verifies that
# the method runs end to end and honours the seeds; it is too small to time the real
# grid or to calibrate a retrieval threshold. Edit the full config and regenerate.
"""


def _set_nested(cfg: dict, dotted: str, value) -> None:
    *parents, leaf = dotted.split(".")
    node = cfg
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def build(name: str, full: dict) -> dict:
    cfg = dict(full)

    cell = {
        "name": DATASET,
        "max_index_docs": MAX_INDEX_DOCS,
        "max_test_queries": MAX_TEST_QUERIES,
    }
    cfg["datasets"] = [dict(cell)]
    # The singular block is the fallback the runners read when `datasets:` is
    # removed. It carries the same caps, so dropping the list cannot silently widen
    # the smoke cell to the full config's corpus.
    fallback = cfg.get("dataset")
    cfg["dataset"] = {**fallback, **cell} if isinstance(fallback, dict) else dict(cell)

    cfg["llms"] = [{"model_name": MODEL}]
    if isinstance(cfg.get("llm"), dict):
        cfg["llm"] = {**cfg["llm"], "model_name": MODEL}

    cfg["max_queries"] = MAX_TEST_QUERIES

    # DP-SynRAG extracts keywords with the generator itself; a small model rarely
    # emits an in-vocabulary word, and the paper's no-fallback behaviour would then
    # yield empty extractions and an empty synthetic corpus, which would look like a
    # broken pipeline rather than a small model.
    if name == "dp_synrag":
        cfg.setdefault("keyword_extractor", {})["substring_fallback"] = True
        # The histogram this method clusters is noisy in absolute terms, so a corpus
        # small enough for the other seven leaves it with no cluster above the
        # threshold and an empty synthetic corpus -- which exercises the fallback path
        # and none of the method. It gets its own, larger corpus for that reason.
        cfg["datasets"][0]["max_index_docs"] = 4000
        cfg["dataset"]["max_index_docs"] = 4000
        cfg.setdefault("clustering", {})["R"] = 8
        cfg.setdefault("cluster_retrieval", {})["k"] = 20
        cfg.setdefault("synthetic_generation", {})["T"] = 12
        cfg.setdefault("inference", {})["k_retrieved"] = 4
        # The filter asks whether the text names a diagnosis. A small generator never
        # writes one, so every synthetic document would be rejected and the corpus
        # would come out empty for a reason that has nothing to do with the mechanism.
        cfg.setdefault("self_filter", {})["enabled"] = False

    for knob, values in EPSILON_POINTS[name].items():
        _set_nested(cfg, knob, values)

    # Two points, not one: a single point cannot show that the budget knob moves the
    # method along the shared axis, which is the property the main grid depends on. Any
    # knob named above with one value gets a second one here rather than in the table,
    # so a method whose axis is its own (DP-SynRAG's, far off the shared one) still ends
    # up swept.
    for knob, values in EPSILON_POINTS[name].items():
        node = cfg
        *parents, leaf = knob.split(".")
        for part in parents:
            node = node[part]
        if isinstance(node.get(leaf), list) and len(node[leaf]) == 1:
            lo = node[leaf][0]
            node[leaf] = sorted({lo, lo * 4})

    # Results land beside the full run's, under their own root.
    if "output_dir" in cfg:
        cfg["output_dir"] = f"results/smoke/{name}/"
    if "output_path" in cfg:
        cfg["output_path"] = f"results/smoke/{name}/{name}.json"

    return cfg


def main() -> int:
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    for name in sorted(EPSILON_POINTS):
        src = FULL_DIR / f"{name}.yaml"
        with open(src, encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}

        text = HEADER.format(
            name=name, dataset=DATASET, model=MODEL, queries=MAX_TEST_QUERIES
        ) + yaml.safe_dump(build(name, full), sort_keys=False, allow_unicode=True)

        dest = SMOKE_DIR / f"{name}.yaml"
        dest.write_text(text, encoding="utf-8")
        # Relative where that reads better, absolute when the destination is elsewhere
        # -- the regeneration guard generates into a temp dir and compares.
        shown = dest.relative_to(PROJECT_ROOT) if dest.is_relative_to(PROJECT_ROOT) else dest
        print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
