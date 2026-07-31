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

from metrics.mia import score_mia_records  # noqa: E402
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
    # Separates "did not answer" from "answered wrongly": several methods return an
    # empty string when their budget runs out, and averaged into the overlap metrics
    # those look like ordinary mistakes.
    "answer_rate",
]
# Membership-inference AUCs, scored offline from the per-probe records saved by the
# standalone MIA run (present only in files that carry a "mia_probes" list):
#   s2mia_auc            — AUC-MIA via S2MIA (BLEU answer vs held-out doc part), the
#                          DPVoteRAG/Koga protocol; the primary privacy number.
#   s2mia_tpr_at_fpr01   — S2MIA true-positive rate at 1% false-positive rate.
#   rag_mia_auc          — AUC of the Anderson et al. black-box Yes/No attack.
#   synth_retention_auc  — DP-SynRAG only: are member input docs reproduced in the
#                          generated corpus more than held-out ones?
MIA_METRICS = ["s2mia_auc", "s2mia_tpr_at_fpr01", "rag_mia_auc", "synth_retention_auc"]
ALL_METRICS = UTILITY_METRICS + MIA_METRICS
# hit_rate is computed only for set-valued golds (Movielens favorites), separately
# from UTILITY_METRICS (those are scored without reference_sets). It is emitted as a
# CSV column regardless, left blank for datasets without favorites.
CSV_METRICS = [*UTILITY_METRICS, "hit_rate", "retrieval_precision_at_k", *MIA_METRICS]

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
    # Standalone MIA-run files carry per-probe records instead of QA predictions;
    # they are scored for the MIA AUCs alone (no utility metrics to compute).
    mia_records = data.get("mia_probes")
    if not preds and not mia_records:
        print(f"  [skip] {path.name}: no predictions or MIA probes found")
        return None

    metrics: dict = {}

    if preds:
        # perplexity is computed over the synthetic corpus when available (DP-SynRAG),
        # else over the predictions (consistent with the in-process runners).
        texts = data.get("synthetic_texts") or data.get("texts") or preds

        # alias_sets: per-item acceptable answers (TriviaQA aliases, NQ-open multi-gold)
        # when the runner saved them on each result item; the text-overlap metrics then
        # score max-over-aliases. Falls back to the single reference when absent.
        items = data.get("results") or []
        aliases = [it.get("answers") for it in items]
        alias_sets = None
        if any(isinstance(a, list) and a for a in aliases) and len(items) == len(preds):
            alias_sets = [
                [str(x) for x in a] if isinstance(a, list) and a else [r]
                for a, r in zip(aliases, refs)
            ]

        util = compute_utility_metrics(preds, refs, UTILITY_METRICS, texts=texts, alias_sets=alias_sets)

        # hit_rate for set-valued golds (e.g. Movielens top-k favorites) when the
        # result items carry a `favorites` list. Aligned by length so other files
        # (without favorites) are untouched.
        favorites = [it.get("favorites") for it in items]
        if any(favorites) and len(items) == len(preds):
            reference_sets = [
                [str(x) for x in (fav or [])] if fav else [r]
                for fav, r in zip(favorites, refs)
            ]
            util.update(
                compute_utility_metrics(preds, refs, ["hit_rate"], reference_sets=reference_sets)
            )

        # retrieval_precision_at_k needs the same index's ranking without any privacy
        # mechanism as its gold, so it is scored only for runs that saved both.
        retrieved = [it.get("retrieved_ids") for it in items]
        nonprivate = [it.get("nonprivate_retrieved_ids") for it in items]
        if any(r is not None for r in retrieved) and any(g for g in nonprivate):
            # `None` marks "never recorded" and stays NaN; `[]` marks "the mechanism
            # returned nothing", which is a real zero.
            util.update(
                compute_utility_metrics(
                    preds,
                    refs,
                    ["retrieval_precision_at_k"],
                    retrieved_ids=[r if r is not None else None for r in retrieved],
                    relevant_ids=[g or [] for g in nonprivate],
                    precision_k=int(
                        data.get("k_retrieved")
                        or (data.get("config") or {}).get("k_retrieved")
                        or (data.get("pipeline") or {}).get("k_retrieved")
                        or 50
                    ),
                )
            )
        metrics.update(util)

    if mia_records:
        # synthetic_texts (DP-SynRAG) enables the retention AUC (member input docs
        # reproduced in the generated corpus); harmless/absent for other methods.
        mia = score_mia_records(mia_records, synthetic_texts=data.get("synthetic_texts"))
        metrics.update(mia)
        if "s2mia_auc" in mia:
            data["mia_auc"] = mia["s2mia_auc"]

    data["metrics"] = {**data.get("metrics", {}), **metrics}
    # Mirror the flat keys some summaries read.
    if "match_accuracy" in metrics:
        data["match_accuracy"] = metrics["match_accuracy"]
    if "contains_accuracy" in metrics:
        data["contains_accuracy"] = metrics["contains_accuracy"]

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def _csv_row(path: Path, data: dict, root: Path) -> dict:
    rel = path.relative_to(root) if root in path.parents or root == path.parent else path
    # dp_synrag stores the generator id under hyperparams.llm, not a top-level llm.
    llm = data.get("llm") or (data.get("hyperparams") or {}).get("llm")
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
    failures: list[str] = []
    for path in files:
        print(f"- {path.relative_to(root) if root in path.parents else path.name}")
        try:
            data = score_file(path)
        except Exception as exc:  # noqa: BLE001
            # One malformed file must not cost the rest of the pass: without this the
            # loop dies, every later file goes unscored, and metrics_summary.csv is
            # never written -- discarding the scoring of a run that took days.
            print(f"  [error] {path.name}: {type(exc).__name__}: {exc}")
            failures.append(str(path))
            continue
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
    if failures:
        # Loud, and at the end where it will be seen: a partially scored run must not
        # look like a complete one.
        print(f"\n{len(failures)} file(s) could not be scored:")
        for path in failures:
            print(f"  - {path}")


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
