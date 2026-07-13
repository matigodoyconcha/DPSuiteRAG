"""Offline metric scorer for BenchDP-RAG.

Computes the FULL metric set from saved result JSONs in a clean process, with
no method generator in GPU memory. This avoids the OOM that occurs when the
model-based metrics (BERTScore -> roberta-large, perplexity -> gpt2) load on top
of the method's generator, and decouples evaluation from generation so results
can be (re)scored without re-running.

The orchestrating runner (experiments/runner.py) sets BENCHDP_DEFER_METRICS=1
for the method subprocesses (so they only save raw predictions/references) and
calls this script at the end of a run. It can also be run standalone over any
results directory or a single result JSON.

Usage:
    python experiments/score_results.py results/run_YYYYmmdd_HHMMSS
    python experiments/score_results.py results/run_.../dpvoterag/triviaqa/epsilon_8.0.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.privacy import compute_privacy_metrics  # noqa: E402
from metrics.utility import compute_utility_metrics  # noqa: E402

# Canonical full metric set (mirrors the configs). The scorer is the single
# source of truth for metrics, so it always computes the whole set.
UTILITY_METRICS = [
    "match_accuracy",
    "contains_accuracy",
    "rouge_l",
    "rouge_1",
    "levenshtein",
    "f1_score",
    "bert_score_f1",
    "perplexity",
]
PRIVACY_METRICS = ["mia_auc_bleu"]
ALL_METRICS = UTILITY_METRICS + PRIVACY_METRICS
# hit_rate is computed only for set-valued golds (Movielens favorites), separately
# from UTILITY_METRICS (those are scored without reference_sets). It is emitted as a
# CSV column regardless, left blank for datasets without favorites.
CSV_METRICS = [*UTILITY_METRICS, "hit_rate", *PRIVACY_METRICS]

# Result-like JSONs only; skip aggregate/report artifacts.
_SKIP_NAMES = {"summary_table.csv", "summary_table.tex", "metrics_summary.csv"}


def _extract_preds_refs(data: dict) -> tuple[list[str], list[str]]:
    """Predictions/references from top-level lists, else from the results items."""
    preds = data.get("predictions")
    refs = data.get("references")
    if isinstance(preds, list) and isinstance(refs, list):
        return [str(p) for p in preds], [str(r) for r in refs]
    items = data.get("results") or []
    preds = [str(it.get("prediction", "")) for it in items]
    refs = [str(it.get("reference", "")) for it in items]
    return preds, refs


def score_file(path: Path) -> dict | None:
    """Compute all metrics for one result JSON and write them back. Returns the data."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [skip] {path.name}: cannot read ({exc})")
        return None
    if not isinstance(data, dict):
        return None

    preds, refs = _extract_preds_refs(data)
    if not preds:
        print(f"  [skip] {path.name}: no predictions found")
        return None

    # perplexity is computed over the synthetic corpus when available (DP-SynRAG),
    # else over the predictions (consistent with the in-process runners).
    texts = data.get("synthetic_texts") or data.get("texts") or preds

    util = compute_utility_metrics(preds, refs, UTILITY_METRICS, texts=texts)

    # hit_rate for set-valued golds (e.g. Movielens top-k favorites) when the
    # result items carry a `favorites` list. Aligned by length so other files
    # (without favorites) are untouched.
    items = data.get("results") or []
    favorites = [it.get("favorites") for it in items]
    if any(favorites) and len(items) == len(preds):
        reference_sets = [
            [str(x) for x in (fav or [])] if fav else [r]
            for fav, r in zip(favorites, refs)
        ]
        util.update(
            compute_utility_metrics(preds, refs, ["hit_rate"], reference_sets=reference_sets)
        )

    priv = compute_privacy_metrics(preds, refs, PRIVACY_METRICS)
    metrics = {**util, **priv}

    data["metrics"] = {**data.get("metrics", {}), **metrics}
    # Mirror the flat keys some summaries read.
    if "match_accuracy" in metrics:
        data["match_accuracy"] = metrics["match_accuracy"]
    if "contains_accuracy" in metrics:
        data["contains_accuracy"] = metrics["contains_accuracy"]
    if "mia_auc_bleu" in metrics:
        data["mia_auc"] = metrics["mia_auc_bleu"]

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def _csv_row(path: Path, data: dict, root: Path) -> dict:
    rel = path.relative_to(root) if root in path.parents or root == path.parent else path
    llm = data.get("llm")
    model = llm.get("model_name", "") if isinstance(llm, dict) else (llm or "")
    metrics = data.get("metrics", {})
    # dp_synrag (older runs) only stored epsilon nested under privacy_accounting.zcdp;
    # fall back to it so its epsilon column isn't blank.
    nested_eps = (data.get("privacy_accounting", {}).get("zcdp", {}) or {}).get("epsilon", "")
    row = {
        "method": data.get("method", ""),
        "dataset": data.get("dataset", ""),
        "model": model,
        "epsilon": data.get("epsilon_total", data.get("epsilon", nested_eps)),
        "file": str(rel),
    }
    row.update({m: metrics.get(m, "") for m in CSV_METRICS})
    return row


def score_path(target: Path) -> None:
    if target.is_file():
        files = [target]
        root = target.parent
    else:
        files = sorted(p for p in target.rglob("*.json") if p.name not in _SKIP_NAMES)
        root = target

    print(f"Scoring {len(files)} result file(s) under {target} ...")
    rows: list[dict] = []
    for path in files:
        print(f"- {path.relative_to(root) if root in path.parents else path.name}")
        data = score_file(path)
        if data is not None:
            rows.append(_csv_row(path, data, root))

    if rows and target.is_dir():
        csv_path = target / "metrics_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["method", "dataset", "model", "epsilon", "file", *CSV_METRICS])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote consolidated metrics to {csv_path}")
    print(f"Scored {len(rows)} file(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline metric scorer for BenchDP-RAG results.")
    parser.add_argument(
        "target",
        type=str,
        help="A run directory (results/run_...) or a single result JSON to (re)score.",
    )
    args = parser.parse_args()
    target = Path(args.target)
    if not target.exists():
        raise SystemExit(f"Target not found: {target}")
    score_path(target)


if __name__ == "__main__":
    main()
