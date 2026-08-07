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
# n_scored / n_unscoreable are the denominator the text metrics used and the number of
# items they dropped. An item whose prediction is empty AND whose golds are all empty
# after QA normalization compares nothing to nothing, and exact match used to call that
# a hit — so the cells that halt, exhaust their budget or decline to release, which are
# the ones at the tight end of the privacy axis, collected perfect scores for answering
# nothing. Those items are no longer scored, and the count rides next to every mean so a
# reader can see a mean over 37 of 40 questions for what it is.
COVERAGE_METRICS = ["n_scored", "n_unscoreable"]
CSV_METRICS = [
    *UTILITY_METRICS, "hit_rate", "retrieval_precision_at_k", *COVERAGE_METRICS, *MIA_METRICS
]

#: Whether each method's RETRIEVAL runs a privacy mechanism, as opposed to protecting
#: only the generation. dp_synrag is absent on purpose: it does not retrieve from the
#: corpus at all, it replaces it.
_PRIVATE_RETRIEVAL = {
    "vanilla": False,          # it is the non-private ranking
    "dpvoterag": False,        # plain cosine search; the privacy is in the vote
    "dpsparsevoterag": False,
    "dpksa": False,
    "dp_rag": True,            # exponential mechanism over the scores
    "private_rag": True,       # per-document budget filter
    "private_rag_ada": True,   # + noisy prefix-sum threshold
}

#: Columns of the consolidated CSV, in order. The writer and `_csv_row` read the same
#: list: a key built without a column here raises at write time, after the run that
#: produced it has already been paid for.
CSV_FIELDS = [
    "method", "dataset", "model",
    "epsilon", "epsilon_target", "epsilon_method_own", "epsilon_classic_zcdp",
    "epsilon_workload", "epsilon_workload_basic",
    "epsilon_workload_q1", "epsilon_workload_q100",
    "epsilon_workload_spent", "budget_utilisation",
    "workload_queries", "workload_error",
    "private_retrieval",
    # The answer length every text metric is monotone in, carried by every method's
    # payload so two rows can be seen to have been measured over answers of the same
    # length rather than taken on trust from a pair of YAML files.
    "answer_tokens",
    # What the privacy mechanism actually served. DPSparseVoteRAG's Sparse Vector
    # Technique routes tokens down a budget-free PUBLIC path, so a cell whose private
    # mechanism never fires still reports a non-zero `answer_rate` and non-zero accuracy
    # over text no mechanism touched. `private_answer_rate` is the column to read
    # instead of `answer_rate` when deciding whether the mechanism served a cell;
    # `public_only` is true exactly when it never fired, and `tokens_private` counts the
    # releases it made. All three are blank — not 0 — for the methods that do not route
    # a public path, so "the mechanism served nothing" stays distinguishable from
    # "the question does not arise for this method".
    "private_answer_rate", "public_only", "tokens_private",
    # DP-RAG only: the share of the cell's total budget that went to retrieval,
    # eps_r / (eps_r + eps_g), as the cell actually ran it. The configured ratio
    # eps_r / token_eps is a constant, but eps_g composes T copies of token_eps
    # sublinearly, so the realised split moves across the sweep with the ratio fixed.
    "epsilon_retrieval_share",
    "file", *CSV_METRICS,
]

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


def _shared_payload_path(path: Path, data: dict) -> Path | None:
    """The sibling holding this record's predictions, when it points at one.

    A method whose cost does not change with the workload files one measurement at every
    workload size — the rows have to exist for a per-Q comparison to find it at every Q —
    but the predictions and the generated corpus behind them are one inference run. They
    are stored once and the other records name that file; resolved here, every record is
    scored from the same text without the text being stored four times. Restricted to a
    bare filename in the same directory: a result payload is data, and a path out of its
    own directory is not something a scorer should follow.
    """
    name = data.get("shared_payload")
    if not name or Path(str(name)).name != str(name):
        return None
    sibling = path.parent / str(name)
    return sibling if sibling.is_file() else None


def score_file(path: Path, scored: dict[Path, dict] | None = None) -> dict | None:
    """Compute all metrics for one result JSON and write them back. Returns the data.

    ``scored`` caches, by source file, the metrics already computed from a payload, so a
    record that shares another's predictions adopts them instead of recomputing.
    """
    if scored is None:
        scored = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [skip] {path.name}: cannot read ({exc})")
        return None
    if not isinstance(data, dict):
        return None

    # A record that stores its predictions elsewhere is scored from that file, and the
    # result is remembered so its siblings cost nothing.
    source = _shared_payload_path(path, data) or path
    if source in scored:
        data["metrics"] = {**data.get("metrics", {}), **scored[source]}
        if "s2mia_auc" in scored[source]:
            data["mia_auc"] = scored[source]["s2mia_auc"]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [shared] metrics adopted from {source.name}")
        return data
    borrowed: dict = {}
    if source != path:
        try:
            shared = json.loads(source.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [skip] {path.name}: shared payload {source.name} unreadable ({exc})")
            return None
        # Only the bulk is borrowed. Everything that identifies this record — its
        # workload size, its epsilon, its own protocol fields — stays its own.
        for key in ("results", "predictions", "references", "synthetic_texts", "texts"):
            if key in shared and key not in data:
                borrowed[key] = shared[key]
        data.update(borrowed)

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

    scored[source] = metrics
    # Borrowed text is not written back. Persisting it would store the shared payload in
    # every record that points at the one holding it, which is the duplication the
    # pointer exists to avoid.
    for key in borrowed:
        data.pop(key, None)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    for key, value in borrowed.items():
        data[key] = value
    return data


def _optional(value):
    """A payload field that a method may legitimately not have, as a CSV cell.

    Absent (``None``) becomes the empty cell, which reads as "not applicable to this
    method"; every value the payload does carry is written through untouched, so a
    measured ``0`` / ``False`` stays a measurement and never collapses into the blank.
    """
    return "" if value is None else value


def _shared_epsilon(data: dict, own: object) -> object:
    """The value of the common privacy axis for one result payload.

    ``epsilon_workload`` is the one figure computed by a single accountant at a single
    delta for all eight methods, so it is what "epsilon" means anywhere the methods are
    put side by side. ``epsilon_target`` stands in when the accountant declined the cell
    (it records a ``workload_error`` and no epsilon) — the point still names its place on
    the axis. ``own`` is the last resort, for runs written before the axis existed.
    """
    workload = data.get("epsilon_workload")
    if workload is not None and workload != "":
        return workload
    target = data.get("epsilon_target")
    if target is not None and target != "":
        return target
    return own


def _csv_row(path: Path, data: dict, root: Path) -> dict:
    rel = path.relative_to(root) if root in path.parents or root == path.parent else path
    # dp_synrag stores the generator id under hyperparams.llm, not a top-level llm.
    llm = data.get("llm") or (data.get("hyperparams") or {}).get("llm")
    model = llm.get("model_name", "") if isinstance(llm, dict) else (llm or "")
    metrics = data.get("metrics", {})
    # dp_synrag (older runs) only stored epsilon nested under privacy_accounting.zcdp;
    # fall back to it so its own-epsilon column isn't blank.
    nested_eps = (data.get("privacy_accounting", {}).get("zcdp", {}) or {}).get("epsilon", "")
    # MURAG and MURAG-ADA both write method="private_rag" and differ only in `variant`,
    # so without this the two collapse into one method on every grouped table and plot.
    method = data.get("method", "")
    if method == "private_rag" and data.get("variant") == "murag_ada":
        method = "private_rag_ada"
    # Runs written before the curve existed carry only their own Q; those columns stay
    # blank rather than being back-filled with the single point, which would read as a
    # method whose epsilon does not move with the workload.
    by_q = data.get("epsilon_workload_by_q") or {}
    # Whether the retrieval this row scores was itself private. Three of the methods
    # protect only the generation and retrieve with a plain cosine search, so their
    # overlap with the non-private ranking is 1.0 by identity; reported without this
    # column that reads as perfect retrieval under privacy, which is the opposite of
    # what it means.
    private_retrieval = _PRIVATE_RETRIEVAL.get(method, "")
    # Each method's own headline number, in its own units at its own delta: a per-query
    # PLD composition for DP-RAG, the closed-form zCDP conversion for DP-SynRAG, an
    # allocated budget for the vote methods. Comparing two of them is comparing a "10"
    # spent per query with a "10" spent over a whole corpus, so it is kept out of the
    # column named `epsilon` and reported under a name that says whose number it is.
    own_epsilon = data.get("epsilon_total", data.get("epsilon", nested_eps))
    row = {
        "method": method,
        "dataset": data.get("dataset", ""),
        "model": model,
        # `epsilon` is the shared axis: the privacy loss over a Q-question workload at
        # the common delta, which is the quantity every figure and table compares the
        # methods on. It is the cell's target by construction — the knob was derived by
        # inverting this accountant — so a run's grid coordinate and its plotted x are
        # the same number. Pre-axis runs fall back to their own epsilon so their rows
        # are not blank; `epsilon_method_own` says which one this is.
        "epsilon": _shared_epsilon(data, own_epsilon),
        "epsilon_target": _optional(data.get("epsilon_target")),
        "epsilon_method_own": own_epsilon,
        # DP-SynRAG's classic zCDP -> (eps, delta) conversion, rho + sqrt(4 rho ln(1/delta)).
        # The axis converts the same rho through the Renyi curve and lands tighter, by up
        # to 48% at the private end, so the two must not share a column name.
        "epsilon_classic_zcdp": _optional(
            (data.get("privacy_accounting", {}).get("zcdp", {}) or {}).get("epsilon")
        ),
        "epsilon_workload": data.get("epsilon_workload", ""),
        "epsilon_workload_basic": data.get("epsilon_workload_basic", ""),
        # The two ends of the workload curve. Q=1 is the single-query regime the source
        # papers report, where every method looks comparable; Q=100 is the same run
        # priced over a question stream, where they stop being. Utility is unchanged
        # between the two columns — it is always the mean over every question asked.
        "private_retrieval": private_retrieval,
        # What the mechanism actually spent, composed by the same accountant. NOT a
        # guarantee: the number of releases depends on the corpus, so a run that halted
        # early reports less precisely because of its own data and bounds no
        # neighbouring corpus. Informative next to the budget — the ratio shows how much
        # of what a method is charged it never uses.
        "epsilon_workload_spent": data.get("epsilon_workload_spent", ""),
        "budget_utilisation": data.get("budget_utilisation", ""),
        "epsilon_workload_q1": by_q.get("1", ""),
        "epsilon_workload_q100": by_q.get("100", ""),
        "workload_queries": data.get("workload_queries", ""),
        # Why a method has no place on the shared axis, when it has none.
        "workload_error": data.get("workload_error") or "",
        # Protocol fields, blank when the payload does not carry them: an absent key
        # means the question does not arise for this method, which is not the same fact
        # as a measured zero.
        "answer_tokens": _optional(data.get("answer_tokens")),
        "private_answer_rate": _optional(data.get("private_answer_rate")),
        "public_only": _optional(data.get("public_only")),
        "tokens_private": _optional(data.get("tokens_private")),
        "epsilon_retrieval_share": _optional(data.get("epsilon_retrieval_share")),
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
    # The text every record is scored from, keyed by the file that stores it, so a
    # payload shared by several records is read and scored once rather than once per
    # record. BERTScore and perplexity are the expensive part of a pass, and the four
    # workload-size records of one flat-regime cell are the same predictions.
    scored: dict[Path, dict] = {}
    for path in files:
        print(f"- {path.relative_to(root) if root in path.parents else path.name}")
        try:
            data = score_file(path, scored)
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
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
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
