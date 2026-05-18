"""DP-SynRAG: Differentially Private Synthetic-Corpus RAG.

Reference paper: arXiv:2510.06719v2.

This module implements the indexing-time DP method described in the paper:
a private synthetic corpus is built once from the original documents, then
queries are served by a standard (non-DP) RAG pipeline over the synthetic
corpus.

Public components:
    - KeywordExtractor      (Stage 1a: LLM keyword extraction)
    - DPHistogramClusterer  (Stage 1a histogram + 1b soft clustering)
    - DPClusterRetriever    (Stage 1c: noisy centroid + DP threshold)
    - DPSyntheticGenerator  (Stage 2d: clipped-logit token aggregation)
    - SelfFilter            (Stage 2e: YES/NO post-hoc filter)
    - DPSynRAGIndexer       (orchestrator over Stages 1-2)
"""

from .keyword_extractor import KeywordExtractor
from .clusterer import DPHistogramClusterer
from .cluster_retriever import DPClusterRetriever
from .synthetic_generator import DPSyntheticGenerator
from .self_filter import SelfFilter
from .indexer import DPSynRAGIndexer

__all__ = [
    "KeywordExtractor",
    "DPHistogramClusterer",
    "DPClusterRetriever",
    "DPSyntheticGenerator",
    "SelfFilter",
    "DPSynRAGIndexer",
]
