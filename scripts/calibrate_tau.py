"""Re-measure Private-RAG's relevance threshold against whole documents.

MuRAG charges every document scoring above a fixed public threshold, so tau decides how
much of the corpus each query costs. The value in the config was calibrated when
documents were split into 200-word chunks; the benchmark now indexes whole documents,
which shifts the cosine distribution, and the failure modes on either side are both
silent:

* too high — the retriever returns fewer than m*k documents, sometimes none, the voter
  set is empty and utility degrades with no error;
* too low  — most of the corpus is charged per query, every per-document budget spends
  out at M=1, and the method looks like it collapsed under its own guarantee.

Reports the score distribution per dataset and the tau that charges a target number of
documents per query, so the threshold is set from the corpus rather than inherited.

Usage:
    python scripts/calibrate_tau.py --datasets chatdoctor movielens --target-charged 160
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import build_chunker, load_dataset_from_config  # noqa: E402
from vanilla.embedder import HFEmbedder  # noqa: E402

DEFAULT_EMBEDDER = "Snowflake/snowflake-arctic-embed-m-v1.5"


def measure(
    documents: list[str],
    queries: list[dict],
    *,
    embedder_model: str,
    target_charged: int,
    m_times_k: int,
) -> dict:
    embedder = HFEmbedder(model_name=embedder_model)
    doc_vectors = np.asarray(embedder.embed(documents), dtype=np.float32)
    doc_vectors /= np.maximum(np.linalg.norm(doc_vectors, axis=1, keepdims=True), 1e-12)

    per_query_scores = []
    for item in queries:
        q = embedder.embed_query(str(item.get("query", "")))
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        per_query_scores.append(doc_vectors @ q)
    scores = np.vstack(per_query_scores)

    # tau that charges `target_charged` documents on the median query.
    per_query_sorted = -np.sort(-scores, axis=1)
    idx = min(target_charged, per_query_sorted.shape[1]) - 1
    tau_for_target = float(np.median(per_query_sorted[:, idx]))

    def charged_at(tau: float) -> np.ndarray:
        return (scores >= tau).sum(axis=1)

    counts = charged_at(tau_for_target)
    return {
        "n_documents": len(documents),
        "n_queries": len(queries),
        "score_p50": float(np.percentile(scores, 50)),
        "score_p90": float(np.percentile(scores, 90)),
        "score_p99": float(np.percentile(scores, 99)),
        "score_max": float(scores.max()),
        "recommended_tau": round(tau_for_target, 4),
        "charged_mean": float(counts.mean()),
        "charged_min": int(counts.min()),
        # The number that matters for starvation: a query below m*k has an incomplete
        # voter set, and one at zero has none at all.
        "queries_below_m_times_k": int((counts < m_times_k).sum()),
        "queries_with_nothing": int((counts == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--max-index-docs", type=int, default=20000)
    parser.add_argument("--max-test-queries", type=int, default=100)
    parser.add_argument("--embedder", type=str, default=DEFAULT_EMBEDDER)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument(
        "--target-charged", type=int, default=160,
        help="Documents to charge on the median query (the config's calibration point).",
    )
    parser.add_argument("--m-times-k", type=int, default=50)
    parser.add_argument("--max-document-words", type=int, default=400)
    parser.add_argument("--output", type=str, default="results/tau_calibration.json")
    args = parser.parse_args()

    chunker = build_chunker(
        {"chunking": {"enabled": False, "max_document_words": args.max_document_words}}
    )
    summaries = {}
    for name in args.datasets:
        print(f"\n=== {name}")
        dataset = load_dataset_from_config(
            {
                "name": name,
                "max_index_docs": args.max_index_docs,
                "max_test_queries": args.max_test_queries,
            },
            args.data_seed,
        )
        # Measure the text that actually gets indexed, cap included.
        documents = [chunker.chunk(d)[0] for d in dataset.get_documents()]
        summary = measure(
            documents,
            dataset.get_queries(),
            embedder_model=args.embedder,
            target_charged=args.target_charged,
            m_times_k=args.m_times_k,
        )
        summaries[name] = summary
        print(
            f"  scores  p50={summary['score_p50']:.3f}  p90={summary['score_p90']:.3f}  "
            f"p99={summary['score_p99']:.3f}  max={summary['score_max']:.3f}\n"
            f"  recommended tau       : {summary['recommended_tau']}\n"
            f"  charged per query     : mean {summary['charged_mean']:.1f}, "
            f"min {summary['charged_min']}\n"
            f"  starved queries       : {summary['queries_below_m_times_k']} below m*k, "
            f"{summary['queries_with_nothing']} with nothing"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}")
    print("Set private_rag.threshold per dataset from `recommended_tau`.")


if __name__ == "__main__":
    main()
