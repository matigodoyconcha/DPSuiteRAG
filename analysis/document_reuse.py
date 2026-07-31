"""How often the same document serves different questions, per dataset.

Runs offline, against the index alone, and predicts which methods will struggle before
any GPU time is spent. A corpus where every question pulls a different document is one
where a per-document privacy filter never has to switch anything off; a corpus where a
handful of documents answer most questions is one where those documents spend out early
and the method degrades no matter how generous the budget is. The same split explains
where per-query composition hurts most: with little reuse, the workload cost is close
to Q independent releases.

The two families the papers assume are:

* *independent* — Natural Questions, TriviaQA: a large passage corpus, each question
  answered by its own passages.
* *dependent* — ChatDoctor, MovieLens: a small corpus whose documents are shared across
  many questions.

Usage:
    python analysis/document_reuse.py --datasets chatdoctor movielens --top-k 50
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

from experiments.common import load_dataset_from_config  # noqa: E402
from vanilla.embedder import HFEmbedder  # noqa: E402

DEFAULT_EMBEDDER = "Snowflake/snowflake-arctic-embed-m-v1.5"


def reuse_histogram(
    documents: list[str],
    queries: list[dict],
    *,
    embedder_model: str = DEFAULT_EMBEDDER,
    top_k: int = 50,
    batch_size: int = 64,
) -> dict:
    """Count, for each document, how many questions place it in their top-k.

    Returns the histogram plus the summary numbers the report needs: what share of
    documents are never retrieved, how concentrated the retrieved ones are, and how
    many distinct documents the whole question set touches.
    """
    if not documents or not queries:
        raise ValueError("need both documents and queries to measure reuse")

    embedder = HFEmbedder(model_name=embedder_model)
    doc_vectors = np.asarray(embedder.embed(documents), dtype=np.float32)
    doc_vectors /= np.maximum(np.linalg.norm(doc_vectors, axis=1, keepdims=True), 1e-12)

    counts = np.zeros(len(documents), dtype=np.int64)
    texts = [str(q.get("query", "")) for q in queries]
    k = min(int(top_k), len(documents))
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        # embed_query, not embed: an asymmetric retriever like arctic-embed prefixes
        # queries with a prompt the documents never get, and embedding them the same
        # way would rank against vectors the real retriever never sees.
        q_vectors = np.asarray(
            [embedder.embed_query(text) for text in chunk], dtype=np.float32
        )
        q_vectors /= np.maximum(np.linalg.norm(q_vectors, axis=1, keepdims=True), 1e-12)
        scores = q_vectors @ doc_vectors.T
        for row in scores:
            top = np.argpartition(-row, k - 1)[:k] if k < len(row) else np.arange(len(row))
            counts[top] += 1

    retrieved = counts[counts > 0]
    total_slots = int(counts.sum())
    # Share of all retrieval slots taken by the busiest 1% of documents: the single
    # number that separates the two families.
    n_top = max(1, len(counts) // 100)
    busiest = np.sort(counts)[::-1][:n_top]

    return {
        "n_documents": len(documents),
        "n_queries": len(queries),
        "top_k": k,
        "never_retrieved_fraction": float((counts == 0).mean()),
        "distinct_documents_touched": int((counts > 0).sum()),
        "max_reuse": int(counts.max()),
        "mean_reuse_when_retrieved": float(retrieved.mean()) if len(retrieved) else 0.0,
        "median_reuse_when_retrieved": float(np.median(retrieved)) if len(retrieved) else 0.0,
        "share_of_slots_in_busiest_1pct": (
            float(busiest.sum() / total_slots) if total_slots else 0.0
        ),
        "histogram": np.bincount(counts).tolist(),
    }


def classify(summary: dict) -> str:
    """Label the corpus by how concentrated retrieval is.

    Reads the busiest document's reuse as a fraction of the question count, which is
    scale-free: 1.0 means one document answers every question, and a corpus where each
    question has its own documents sits near ``top_k / n_documents``.

    The obvious statistic — the share of all retrieval slots held by the busiest 1% of
    documents — cannot be used, because its ceiling is ``max(1, N//100) / top_k``. At
    the runbook's ``top_k=50`` that caps a 200-document corpus at 0.04 and a
    943-document one at 0.18, both under any sensible threshold. A corpus where a
    single document answered all 100 questions scored 0.04 and came out "independent",
    and small corpora (MovieLens, ChatDoctor) are precisely the ones this is meant to
    catch.
    """
    if not summary["n_queries"]:
        return "unknown"
    hub_share = summary["max_reuse"] / summary["n_queries"]
    # Half the questions pulling the same document is far past the point where a
    # per-document budget starts switching documents off mid-stream.
    return "dependent" if hub_share >= 0.5 else "independent"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--max-index-docs", type=int, default=20000)
    parser.add_argument("--max-test-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--embedder", type=str, default=DEFAULT_EMBEDDER)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default="results/document_reuse.json",
        help="Where to write the per-dataset summaries.",
    )
    args = parser.parse_args()

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
        summary = reuse_histogram(
            dataset.get_documents(),
            dataset.get_queries(),
            embedder_model=args.embedder,
            top_k=args.top_k,
        )
        summary["classification"] = classify(summary)
        summaries[name] = summary
        print(
            f"  {summary['n_documents']} docs, {summary['n_queries']} queries\n"
            f"  never retrieved      : {summary['never_retrieved_fraction']:.1%}\n"
            f"  distinct docs touched: {summary['distinct_documents_touched']}\n"
            f"  max reuse            : {summary['max_reuse']} queries\n"
            f"  busiest 1% hold      : {summary['share_of_slots_in_busiest_1pct']:.1%} of retrievals\n"
            f"  -> {summary['classification']}"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
