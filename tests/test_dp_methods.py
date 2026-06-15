"""Smoke tests for each DP method.

Each test verifies that the DP component can be instantiated and run
connected to the vanilla pipeline. No quality or privacy comparisons are made.

Methods marked with xfail are not yet implemented — the test documents
that state rather than silencing it.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vanilla.chunker import FixedSizeChunker
from vanilla.embedder import HFEmbedder
from vanilla.indexer import FAISSIndexer
from vanilla.reranker import IdentityReranker
from vanilla.postprocessor import IdentityPostprocessor
from core.pipeline import RAGPipeline

EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TINY_MODEL = "sshleifer/tiny-gpt2"

DOCS = [
    "Differential privacy protects sensitive data by adding calibrated noise.",
    "FAISS is an efficient library for vector similarity search.",
    "A RAG pipeline combines document retrieval and text generation.",
]
QUERY = "What does differential privacy protect?"


# ---------------------------------------------------------------------------
# Shared fixtures — built once per test module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder():
    return HFEmbedder(model_name=EMBEDDER_MODEL)


@pytest.fixture(scope="module")
def built_indexer(embedder):
    indexer = FAISSIndexer()
    vectors = embedder.embed(DOCS)
    metadata = [{"doc_id": i, "chunk_id": 0, "text": d} for i, d in enumerate(DOCS)]
    indexer.build(vectors, metadata)
    return indexer


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    return model, tokenizer


# ---------------------------------------------------------------------------
# DP-RAG  (implemented ✅)
# ---------------------------------------------------------------------------

def test_dp_rag_retriever_top_k(built_indexer, embedder):
    from dp_methods.per_query.dp_rag import DPRetriever, PerDocPromptBuilder

    retriever = DPRetriever(index=built_indexer, epsilon=10.0, k=2, max_docs=10)
    qvec = embedder.embed_query(QUERY)
    results = retriever.retrieve(qvec, k=2)
    assert isinstance(results, list)
    assert all("text" in r for r in results)


def test_dp_rag_retriever_top_p(built_indexer, embedder):
    from dp_methods.per_query.dp_rag import DPTopPRetriever

    retriever = DPTopPRetriever(
        index=built_indexer, epsilon=10.0, p=0.5, alpha=1.0, max_docs=10
    )
    qvec = embedder.embed_query(QUERY)
    results = retriever.retrieve(qvec, k=2)
    assert isinstance(results, list)


def test_dp_rag_per_doc_prompt_builder():
    from dp_methods.per_query.dp_rag import PerDocPromptBuilder

    pb = PerDocPromptBuilder()
    chunks = [{"text": "doc uno"}, {"text": "doc dos"}]
    prompt = pb.build(QUERY, chunks)
    parsed = json.loads(prompt)
    assert isinstance(parsed, list)
    # PerDocPromptBuilder produces k private prompts + 1 public = k+1 total
    assert len(parsed) >= len(chunks)


def test_dp_rag_generator_smoke(built_indexer, embedder):
    from dp_methods.per_query.dp_rag import DPRetriever, PerDocPromptBuilder, DPICLGenerator

    retriever = DPRetriever(index=built_indexer, epsilon=100.0, k=2, max_docs=10)
    prompt_builder = PerDocPromptBuilder()
    generator = DPICLGenerator(
        model_name=TINY_MODEL,
        epsilon=100.0,
        delta=1e-3,
        max_new_tokens=4,
    )
    reranker = IdentityReranker()
    postproc = IdentityPostprocessor()

    pipeline = RAGPipeline(
        chunker=FixedSizeChunker(chunk_size=50, overlap=5),
        embedder=embedder,
        indexer=built_indexer,
        retriever=retriever,
        reranker=reranker,
        prompt_builder=prompt_builder,
        generator=generator,
        postprocessor=postproc,
    )
    result = pipeline.query(QUERY, k=2)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# DPVoteRAG  (implemented ✅)
# ---------------------------------------------------------------------------

def test_dpvoterag_generator_instantiates(tiny_model_and_tokenizer):
    from dp_methods.per_query.dpvoterag import VoteRAGGenerator

    model, tokenizer = tiny_model_and_tokenizer
    gen = VoteRAGGenerator(
        model=model,
        tokenizer=tokenizer,
        m=2,
        k=1,
        epsilon_token=10.0,
        epsilon_total=20.0,
    )
    assert gen is not None


def test_dpvoterag_generator_generate_returns_string(tiny_model_and_tokenizer):
    from dp_methods.per_query.dpvoterag import VoteRAGGenerator

    model, tokenizer = tiny_model_and_tokenizer
    gen = VoteRAGGenerator(
        model=model,
        tokenizer=tokenizer,
        m=2,
        k=1,
        epsilon_token=10.0,
        epsilon_total=20.0,
    )
    result = gen.generate_vote(QUERY, DOCS[:2])
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# DPSparseVoteRAG  (implemented ✅)
# ---------------------------------------------------------------------------

def test_dpsparsevoterag_generator_instantiates(tiny_model_and_tokenizer):
    from dp_methods.per_query.dpvoterag import SparseVoteRAGGenerator

    model, tokenizer = tiny_model_and_tokenizer
    gen = SparseVoteRAGGenerator(
        model=model,
        tokenizer=tokenizer,
        m=2,
        k=1,
        epsilon_token=10.0,
        epsilon_total=20.0,
    )
    assert gen is not None


# ---------------------------------------------------------------------------
# PRIVATE-RAG  (not yet implemented — xfail expected)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="BudgetManager and BudgetAwareRetriever not yet implemented", strict=False)
def test_private_rag_smoke(built_indexer, embedder):
    from dp_methods.per_query.private_rag.budget_manager import BudgetManager  # type: ignore
    from dp_methods.per_query.private_rag.retriever import BudgetAwareRetriever  # type: ignore

    budget_manager = BudgetManager(initial_budget=5.0)
    retriever = BudgetAwareRetriever(indexer=built_indexer, budget_manager=budget_manager)
    qvec = embedder.embed_query(QUERY)
    results = retriever.retrieve(qvec, k=2)
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# DP-SynRAG  (implemented ✅)
# ---------------------------------------------------------------------------

def test_dp_synrag_indexer_smoke(embedder):
    """End-to-end smoke test: build a DP synthetic corpus from a handful of docs."""
    from dp_methods.offline.dp_synrag import (
        KeywordExtractor,
        DPHistogramClusterer,
        DPClusterRetriever,
        DPSyntheticGenerator,
        DPSynRAGIndexer,
    )

    docs = [
        "Patient with diabetes presents with high blood sugar and fatigue.",
        "Patient with hypertension reports chest pain and dizziness.",
        "Patient with asthma has shortness of breath and wheezing.",
        "Patient with arthritis complains of joint pain and stiffness.",
        "Patient with migraine suffers from severe headache and nausea.",
    ] * 4  # 20 docs so clustering has enough material

    tiny_vocab = [
        "patient", "diabetes", "hypertension", "asthma", "arthritis", "migraine",
        "blood", "sugar", "pain", "joint", "breath", "headache", "nausea",
        "fatigue", "chest", "dizziness", "wheezing", "stiffness", "severe",
    ]

    keyword_extractor = KeywordExtractor(
        model_name=TINY_MODEL,
        K=3,
        vocab=tiny_vocab,
        max_new_tokens=20,
        seed=42,
    )
    clusterer = DPHistogramClusterer(R=3, L=2, sigma_h=0.01, seed=42)
    cluster_retriever = DPClusterRetriever(
        embedder=embedder,
        k=2,
        sigma_mu=0.01,
        epsilon_theta_s=100.0,  # near-deterministic threshold
        seed=42,
    )
    synthetic_generator = DPSyntheticGenerator(
        model_name=TINY_MODEL,
        T=5,
        c=1.0,
        tau=1.0,
        prompt_template="Generate a medical case similar to: {document}",
        seed=42,
    )
    indexer = DPSynRAGIndexer(
        embedder=embedder,
        keyword_extractor=keyword_extractor,
        clusterer=clusterer,
        cluster_retriever=cluster_retriever,
        synthetic_generator=synthetic_generator,
        self_filter=None,
        seed=42,
    )
    indexer.build_from_documents(docs)

    assert indexer.synthetic_texts, "no synthetic documents generated"
    assert indexer.embeddings_matrix is not None
    assert indexer.embeddings_matrix.shape[0] == len(indexer.synthetic_texts)
    assert indexer.index is not None
    assert indexer.last_stats["n_original_documents"] == len(docs)
    assert indexer.last_stats["n_synthetic_kept"] == len(indexer.synthetic_texts)


def test_dp_synrag_vanilla_inference(embedder):
    """After building synthetic index, CosineSimilarityRetriever can retrieve docs."""
    from dp_methods.offline.dp_synrag import (
        KeywordExtractor,
        DPHistogramClusterer,
        DPClusterRetriever,
        DPSyntheticGenerator,
        DPSynRAGIndexer,
    )
    from vanilla.retriever import CosineSimilarityRetriever

    docs = [
        "Patient with diabetes presents with high blood sugar and fatigue.",
        "Patient with hypertension reports chest pain and dizziness.",
        "Patient with asthma has shortness of breath and wheezing.",
    ] * 4

    tiny_vocab = [
        "patient", "diabetes", "hypertension", "asthma",
        "blood", "sugar", "pain", "chest", "breath",
    ]

    indexer = DPSynRAGIndexer(
        embedder=embedder,
        keyword_extractor=KeywordExtractor(
            model_name=TINY_MODEL, K=2, vocab=tiny_vocab, max_new_tokens=10, seed=0,
        ),
        clusterer=DPHistogramClusterer(R=2, L=1, sigma_h=0.01, seed=0),
        cluster_retriever=DPClusterRetriever(
            embedder=embedder, k=2, sigma_mu=0.01, epsilon_theta_s=50.0, seed=0,
        ),
        synthetic_generator=DPSyntheticGenerator(
            model_name=TINY_MODEL, T=4, c=1.0, tau=1.0,
            prompt_template="Case: {document}", seed=0,
        ),
        self_filter=None,
        seed=0,
    )
    indexer.build_from_documents(docs)

    retriever = CosineSimilarityRetriever(indexer=indexer)
    qvec = embedder.embed_query("What are the symptoms of diabetes?")
    results = retriever.retrieve(qvec, k=min(2, len(indexer.synthetic_texts)))
    assert isinstance(results, list)
    assert all("text" in r for r in results)
