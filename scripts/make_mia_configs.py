#!/usr/bin/env python
"""Generate ``experiments/configs/mia/`` from ``experiments/configs/full/``.

    python scripts/make_mia_configs.py

The membership-inference run answers a different question from the utility sweep and
cannot reuse its configs unchanged. Three things differ, and each is derived here so
the attack set cannot drift away from the sweep it is supposed to attack:

* **Its own corpus.** ``mia.index_size`` indexes a few thousand member documents and
  holds the rest out as non-members, so the utility sweep's corpus size is irrelevant
  — except that the loader materialises the whole dataset before the split is taken.
  Left at ``max_index_docs: null`` the two DPR datasets would load all 21,015,325
  passages to sample two thousand of them. Capped here at ``MIA_MAX_INDEX_DOCS``.
* **A hundred probes per cell where the sweep asks forty questions.** Each target
  costs two generations (the S2MIA query and the Anderson Yes/No probe), so a cell is
  ``4 * n_per_class`` generations against the sweep's forty — the attack is an order
  of magnitude more expensive per cell than the run it attacks.
* **No workload axis for the methods that are flat in it.** MuRAG, MuRAG-Ada and
  DP-SynRAG spend a budget that does not move with Q, so their mechanism at (eps, Q=25)
  is bit-for-bit the mechanism at (eps, Q=1) and attacking both measures one thing
  twice. They keep the full epsilon axis and a single Q; the analysis replicates the
  record across the Q axis exactly as `shared_payload` already does for utility.

Everything else — the epsilon targets, the LLMs, the datasets, the embedder, the
answer length — is inherited unchanged, so a cell of the attack lands on the same
point of the shared axis as the utility cell it corresponds to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_DIR = _PROJECT_ROOT / "experiments" / "configs" / "full"
MIA_DIR = _PROJECT_ROOT / "experiments" / "configs" / "mia"

#: Members and non-members probed per cell. Every method uses the same number: an AUC
#: measured over 50+50 targets and one measured over 100+100 differ in their sampling
#: error before they differ in anything about the mechanism.
N_PER_CLASS = 50

#: Member documents actually indexed. Small on purpose — the attack asks whether a
#: document is in the index, and a larger index only dilutes the signal it looks for.
INDEX_SIZE = 2000

#: Cap on the documents the loader materialises before the member/non-member split is
#: drawn. It must exceed ``INDEX_SIZE`` comfortably, or there are no documents left to
#: hold out as non-members and ``n_per_class`` is silently capped below the configured
#: value (measured: 2000 here yielded 40 per class, not 50).
MIA_MAX_INDEX_DOCS = 20000

#: Floor on the same cap. The split needs ``INDEX_SIZE`` members *plus* a pool to draw
#: non-members from, and only documents above ``MIA_MIN_DOC_TOKENS`` count toward
#: either. A config sized for its own utility sweep can sit below that — DP-SynRAG
#: indexes 816 to 1600 documents per dataset — in which case ``get_mia_split`` returns
#: None and the runner skips the dataset without producing a cell. The attack surface
#: itself stays identical across methods regardless: that is ``INDEX_SIZE``, and only
#: the pool the held-out targets are drawn from grows here.
MIA_MIN_CORPUS_DOCS = 2 * INDEX_SIZE + 2 * N_PER_CLASS

#: Methods whose guarantee does not move with the workload size. See the module
#: docstring; they keep every epsilon target and a single Q.
FLAT_IN_Q = {"dp_synrag", "private_rag", "private_rag_ada"}

#: Methods that put no document in the prompt. Their utility config caps
#: ``max_index_docs`` only to stop a corpus they never read from being materialised, so
#: the number there names no corpus size. The attack does read the corpus — it draws its
#: members and non-members from it — and a cap carried over from the utility config
#: would sample a different subset of the documents than the runs this condition is the
#: reference for, which is a different set of targets rather than a different mechanism.
NON_INDEXING = {"no_rag"}


def _mia_block() -> dict:
    return {
        "enabled": True,
        "n_per_class": N_PER_CLASS,
        "index_size": INDEX_SIZE,
        "seed": 42,
        "anderson_doc_maxlen": 512,
    }


def _cap_corpus(entry: dict) -> dict:
    current = entry.get("max_index_docs", MIA_MAX_INDEX_DOCS)
    capped = MIA_MAX_INDEX_DOCS if current is None else min(int(current), MIA_MAX_INDEX_DOCS)
    entry["max_index_docs"] = max(capped, MIA_MIN_CORPUS_DOCS)
    return entry


def _entry(entry: dict, method: str) -> dict:
    entry = dict(entry or {})
    if method in NON_INDEXING:
        entry.pop("max_index_docs", None)
    return _cap_corpus(entry)


def build(config: dict, method: str) -> dict:
    config["mia"] = _mia_block()

    if isinstance(config.get("datasets"), list):
        config["datasets"] = [_entry(e, method) for e in config["datasets"]]
    if isinstance(config.get("dataset"), dict):
        config["dataset"] = _entry(config["dataset"], method)

    if method in FLAT_IN_Q and "workload_queries" in config:
        config["workload_queries"] = [min(int(q) for q in config["workload_queries"])]

    out = str(config.get("output_dir", f"results/{method}/"))
    config["output_dir"] = out.replace("results/", "results/mia_", 1)
    return config


def main() -> int:
    sources = sorted(FULL_DIR.glob("*.yaml"))
    if not sources:
        print(f"[error] no configs under {FULL_DIR}", file=sys.stderr)
        return 1

    MIA_DIR.mkdir(parents=True, exist_ok=True)
    for src in sources:
        method = src.stem
        config = yaml.safe_load(src.read_text(encoding="utf-8"))
        dest = MIA_DIR / src.name
        dest.write_text(
            "# GENERATED by scripts/make_mia_configs.py — edit full/ and regenerate.\n"
            + yaml.safe_dump(build(config, method), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        cells = config.get("workload_queries", [None])
        print(f"  {dest.relative_to(_PROJECT_ROOT)}  (Q={cells})")
    print(f"\n{len(sources)} config(s) written to {MIA_DIR.relative_to(_PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
