"""DP-SynRAG orchestrator: builds the synthetic FAISS index from raw documents.

Reference: arXiv:2510.06719v2, Algorithm 1.

Pipeline (build_from_documents):
    Stage 1a  -> KeywordExtractor.extract(documents)
    Stage 1a+ -> DPHistogramClusterer.cluster(extractions)
    Stage 1c  -> DPClusterRetriever.retrieve_clusters(documents, clusters)
    Stage 2d  -> DPSyntheticGenerator.generate_for_clusters(documents, subsets)
    Stage 2e  -> SelfFilter.filter(synthetic_texts)        # optional
    Index     -> embedder.embed(synthetic_texts) + FAISSIndexer.build(...)

The result is a standard FAISS index (delegated to vanilla.indexer.FAISSIndexer)
that exposes ``index``, ``metadata`` and ``embeddings_matrix`` exactly like the
vanilla indexer, so it can be plugged into ``core.pipeline.RAGPipeline`` with
vanilla retriever/generator components at inference time.

The base interface method ``build(vectors, metadata)`` is intentionally not the
right entry point for this class — callers must use ``build_from_documents``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

from core.indexer import BaseIndexer
from vanilla.indexer import FAISSIndexer


class DPSynRAGIndexer(BaseIndexer):
    """Build a DP synthetic-corpus FAISS index in one shot."""

    def __init__(
        self,
        embedder: Any,
        keyword_extractor: Any,
        clusterer: Any,
        cluster_retriever: Any,
        synthetic_generator: Any,
        self_filter: Any | None = None,
        seed: int = 42,
    ):
        self.embedder = embedder
        self.keyword_extractor = keyword_extractor
        self.clusterer = clusterer
        self.cluster_retriever = cluster_retriever
        self.synthetic_generator = synthetic_generator
        self.self_filter = self_filter
        self.seed = int(seed)

        self._inner = FAISSIndexer()
        self.synthetic_texts: list[str] = []
        self.last_stats: dict = {}

    # ------------------------------------------------------------------
    # Properties forwarded to the inner FAISS index (so vanilla retrievers
    # and the DP-RAG retriever both work).
    # ------------------------------------------------------------------

    @property
    def index(self):
        return self._inner.index

    @property
    def metadata(self) -> list[Dict[str, Any]]:
        return self._inner.metadata

    @property
    def embeddings_matrix(self) -> np.ndarray | None:
        return self._inner.embeddings_matrix

    def get_index(self):
        return self._inner.get_index()

    # ------------------------------------------------------------------
    # BaseIndexer interface (no-op here; use build_from_documents instead)
    # ------------------------------------------------------------------

    def build(self, vectors: List[Any], metadata: List[Dict]):
        """Not the intended entry point — use build_from_documents()."""
        raise NotImplementedError(
            "DPSynRAGIndexer.build(vectors, metadata) is intentionally disabled. "
            "Call build_from_documents(documents, task_prefix=...) instead so "
            "the synthetic-corpus pipeline can run end-to-end."
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build_from_documents(
        self,
        documents: List[str],
        task_prefix: str = "",
    ) -> None:
        if not documents:
            raise ValueError("documents must be a non-empty list.")

        # Stage 1a: per-document keyword extraction (binary matrix).
        extractions = self.keyword_extractor.extract(documents)

        # Stage 1a (histogram) + 1b (soft clustering).
        top_keywords_idx, clusters = self.clusterer.cluster(extractions)

        # Stage 1c: per-cluster noisy mean + DP threshold.
        subsets = self.cluster_retriever.retrieve_clusters(documents, clusters)

        # Stage 2d: clipped-logit aggregation per cluster.
        synthetic_texts = self.synthetic_generator.generate_for_clusters(
            documents, subsets, task_prefix=task_prefix
        )

        # Stage 2e: optional self-filter (post-processing, no privacy cost).
        if self.self_filter is not None:
            keep_mask = self.self_filter.filter(synthetic_texts)
        else:
            keep_mask = [bool(t and t.strip()) for t in synthetic_texts]

        kept_records: list[dict] = []
        for r, (text, keep) in enumerate(zip(synthetic_texts, keep_mask)):
            if keep and text and text.strip():
                kept_records.append(
                    {
                        "doc_id": len(kept_records),
                        "chunk_id": 0,
                        "text": text,
                        "source_cluster": r,
                        "source_cluster_size": len(subsets[r]),
                        "source_anchor_keyword_index": int(top_keywords_idx[r])
                        if r < len(top_keywords_idx)
                        else -1,
                    }
                )

        if not kept_records:
            raise RuntimeError(
                "DP-SynRAG produced zero synthetic documents — every cluster was "
                "empty or filtered out. Check sigma_h, sigma_mu, epsilon_theta_s, "
                "T, c, tau, and the self_filter prompt."
            )

        kept_texts = [rec["text"] for rec in kept_records]
        vectors = self.embedder.embed(kept_texts)
        self._inner.build(vectors, kept_records)

        self.synthetic_texts = kept_texts
        self.last_stats = {
            "n_original_documents": len(documents),
            "n_clusters_total": len(clusters),
            "n_non_empty_subsets": sum(1 for s in subsets if s),
            "n_generated_before_filter": sum(1 for t in synthetic_texts if t and t.strip()),
            "n_synthetic_kept": len(kept_records),
            "self_filter_enabled": self.self_filter is not None,
            "clusterer": getattr(self.clusterer, "last_stats", {}),
            "cluster_retriever": getattr(self.cluster_retriever, "last_stats", {}),
            "synthetic_generator": getattr(self.synthetic_generator, "last_stats", {}),
            "self_filter": getattr(self.self_filter, "last_stats", {})
            if self.self_filter is not None
            else None,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self._inner.save(path)
        with open(os.path.join(path, "synthetic_texts.json"), "w", encoding="utf-8") as f:
            json.dump(self.synthetic_texts, f, ensure_ascii=False, indent=2)
        with open(os.path.join(path, "last_stats.json"), "w", encoding="utf-8") as f:
            json.dump(self.last_stats, f, ensure_ascii=False, indent=2, default=str)

    def load(self, path: str):
        self._inner.load(path)
        syn_file = os.path.join(path, "synthetic_texts.json")
        stats_file = os.path.join(path, "last_stats.json")
        if os.path.exists(syn_file):
            with open(syn_file, "r", encoding="utf-8") as f:
                self.synthetic_texts = json.load(f)
        if os.path.exists(stats_file):
            with open(stats_file, "r", encoding="utf-8") as f:
                self.last_stats = json.load(f)
