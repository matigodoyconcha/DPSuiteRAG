#!/usr/bin/env python
"""Check the invariants the cross-method comparison rests on, over a smoke run.

A run that finishes is not a run that is comparable. Every claim the benchmark makes
assumes properties that nothing in a normal run verifies: that all methods answered
the same questions, that a repetition varies the mechanism and not the sample, that a
budget knob moves the method along the shared axis, that an empty answer is reported
as empty rather than averaged in as a wrong one. Each of those can break silently and
leave a results directory that looks entirely normal.

    python experiments/runner.py --smoke --seeds 1 2
    python scripts/verify_smoke.py results/run_<stamp>

Exits non-zero if any check fails, so it can gate the main grid.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def check(self, ok: bool, name: str, detail: str = "") -> None:
        """Record one invariant. `detail` explains the failure, so it prints only then."""
        if ok:
            print(f"  [PASS] {name}")
            return
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
        self.failures.append(f"{name}: {detail}")

    def note(self, text: str) -> None:
        print(f"  [note] {text}")
        self.notes.append(text)


def _payloads(root: Path) -> list[tuple[Path, dict]]:
    out = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {"metrics_summary.csv", "last_stats.json"}:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and (data.get("results") or data.get("predictions")):
            out.append((path, data))
    return out


def _method_of(path: Path, data: dict) -> str:
    method = str(data.get("method", ""))
    if method == "private_rag" and data.get("variant") == "murag_ada":
        return "private_rag_ada"
    return method or path.parts[0]


def _seed_of(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("seed_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                return None
    return None


def _query_ids(data: dict) -> list[str]:
    return [str(it.get("query_id", "")) for it in (data.get("results") or [])]


#: Per-query fields that record what a mechanism decided, as opposed to what the
#: generator then wrote. Two seeds must differ in at least one of them.
_TRACE_KEYS = ("k_hat", "ptr_passed", "keywords", "n_private_tokens", "exit_status")


def _mechanism_traces(data: dict) -> dict[str, list]:
    items = data.get("results") or []
    return {
        key: [it.get(key) for it in items]
        for key in _TRACE_KEYS
        if any(key in it for it in items)
    }


def _first_diff(a: dict, b: dict) -> str:
    for key in a:
        if a[key] != b.get(key):
            return key
    return "?"


def _predictions(data: dict) -> list[str]:
    preds = data.get("predictions")
    if isinstance(preds, list):
        return [str(p) for p in preds]
    return [str(it.get("prediction", "")) for it in (data.get("results") or [])]


def verify(root: Path) -> Report:
    rep = Report()
    payloads = _payloads(root)
    if not payloads:
        rep.check(False, "results present", f"no result payloads under {root}")
        return rep

    by_method: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for path, data in payloads:
        by_method[_method_of(path.relative_to(root), data)].append((path, data))

    print(f"\n{len(payloads)} payload(s), {len(by_method)} method(s): {sorted(by_method)}\n")

    print("Every method answered the same questions")
    id_sets = {}
    for method, items in sorted(by_method.items()):
        ids = {tuple(_query_ids(d)) for _, d in items if _query_ids(d)}
        if not ids:
            rep.check(False, f"{method}: query_id recorded", "no query_id on any result item")
            continue
        rep.check(len(ids) == 1, f"{method}: one question set", f"{len(ids)} distinct sets")
        id_sets[method] = next(iter(ids))
    if len(set(id_sets.values())) > 1:
        ref = sorted(id_sets)[0]
        for method, ids in sorted(id_sets.items()):
            if ids != id_sets[ref]:
                rep.check(False, f"{method} vs {ref}: same questions",
                          f"{len(set(ids) ^ set(id_sets[ref]))} differing ids")
    else:
        rep.check(True, "all methods share one question set",
                  f"{len(next(iter(id_sets.values()), ()))} questions")

    print("\nSeeds separate the sample from the mechanism")
    seeded = [(p, d) for p, d in payloads if _seed_of(p.relative_to(root)) is not None]
    if not seeded:
        rep.note("run without --seeds; re-run as `--smoke --seeds 1 2` to check this")
    else:
        cells: dict[tuple, dict[int, dict]] = defaultdict(dict)
        for path, data in seeded:
            rel = path.relative_to(root)
            seed = _seed_of(rel)
            key = (_method_of(rel, data), data.get("dataset"), path.name)
            cells[key][seed] = data
        compared = 0
        for key, per_seed in sorted(cells.items()):
            if len(per_seed) < 2:
                continue
            compared += 1
            seeds = sorted(per_seed)
            a, b = per_seed[seeds[0]], per_seed[seeds[1]]
            label = f"{key[0]}/{key[1]}/{key[2]}"
            rep.check(_query_ids(a) == _query_ids(b), f"{label}: same questions across seeds",
                      "the corpus and the questions follow data_seed, not run_seed")
            if key[0] == "vanilla":
                # The reference condition has no mechanism to vary: greedy decoding
                # over a fixed corpus is meant to repeat exactly.
                rep.check(_predictions(a) == _predictions(b),
                          f"{label}: the baseline repeats exactly",
                          "the non-private baseline should not move between seeds")
                continue
            pa, pb = _predictions(a), _predictions(b)
            if all(not x.strip() for x in pa + pb):
                rep.note(f"{label}: every answer empty at both seeds — nothing to compare")
            elif pa != pb:
                rep.check(True, f"{label}: the mechanism varies across seeds")
            else:
                # Identical text does not prove the noise never ran. A small model
                # collapses to the same output whatever it is conditioned on, and a
                # concentrated histogram gives the same argmax at any noise draw. What
                # settles it is the mechanism's own output, when the payload records it.
                traces = _mechanism_traces(a), _mechanism_traces(b)
                if traces[0] and traces[0] != traces[1]:
                    rep.note(f"{label}: same text at both seeds, but the mechanism "
                             f"differs ({_first_diff(*traces)}) — the model washes it out")
                else:
                    rep.check(False, f"{label}: the mechanism varies across seeds",
                              "identical predictions and identical mechanism trace: "
                              "run_seed is not reaching the noise")
        if not compared:
            rep.note("no cell was produced at two seeds; nothing to compare")

    print("\nThe budget knob moves the method along the shared axis")
    for method, items in sorted(by_method.items()):
        if method == "vanilla":
            continue
        axis = sorted(
            {round(float(d["epsilon_workload"]), 6)
             for _, d in items if d.get("epsilon_workload") is not None}
        )
        errors = {d.get("workload_error") for _, d in items if d.get("workload_error")}
        if not axis and errors:
            rep.note(f"{method}: no point on the axis — {sorted(errors)[0]}")
            continue
        rep.check(len(axis) >= 2, f"{method}: distinct epsilon_workload per point",
                  f"values={axis}")

    print("\nThe fields the analysis reads are present")
    for method, items in sorted(by_method.items()):
        _, data = items[0]
        missing = [k for k in ("run_seed",) if data.get(k) is None]
        if method != "vanilla":
            # The baseline prices no mechanism, so it carries no workload block.
            for k in ("workload_queries", "epsilon_workload_by_q", "epsilon_workload"):
                if data.get(k) is None:
                    missing.append(k)
        rep.check(not missing, f"{method}: payload fields", f"missing {missing}")

    print("\nAn empty answer is visible as empty")
    for method, items in sorted(by_method.items()):
        for path, data in items:
            preds = _predictions(data)
            if not preds:
                continue
            empty = sum(1 for p in preds if not p.strip())
            rate = (data.get("metrics") or {}).get("answer_rate")
            if empty and rate is None:
                rep.check(False, f"{method}: answer_rate recorded",
                          f"{empty}/{len(preds)} empty answers and no answer_rate")
            elif empty:
                rep.note(f"{method} [{path.name}]: {empty}/{len(preds)} empty, "
                         f"answer_rate={rate}")

    print("\nThe workload curve is flat only where the guarantee is")
    for method, items in sorted(by_method.items()):
        for _, data in items:
            curve = data.get("epsilon_workload_by_q")
            regime = data.get("workload_regime")
            if not curve or not regime:
                continue
            values = [v for v in curve.values() if v is not None]
            if len(values) < 2:
                continue
            flat = max(values) - min(values) < 1e-9
            if regime == "per_query":
                rep.check(not flat, f"{method}: per-query cost grows with Q",
                          f"curve is flat at {values[0]:.4f}")
            else:
                rep.check(flat, f"{method}: {regime} cost is flat in Q",
                          f"curve moves {min(values):.4f}..{max(values):.4f}")
            break

    return rep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", type=str)
    args = parser.parse_args()

    root = Path(args.results_dir)
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 2

    rep = verify(root)
    print("\n" + "=" * 64)
    if rep.failures:
        print(f"{len(rep.failures)} FAILED CHECK(S):")
        for f in rep.failures:
            print(f"  - {f}")
        return 1
    print("All checks passed." + (f" {len(rep.notes)} note(s) above." if rep.notes else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
