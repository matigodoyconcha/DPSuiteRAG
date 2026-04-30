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
    "La privacidad diferencial protege datos sensibles añadiendo ruido calibrado.",
    "FAISS es una librería de búsqueda eficiente por similitud vectorial.",
    "Un pipeline RAG combina recuperación de documentos y generación de texto.",
]
QUERY = "¿Qué protege la privacidad diferencial?"


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
    from dp_methods.dp_rag import DPRetriever, PerDocPromptBuilder

    retriever = DPRetriever(index=built_indexer, epsilon=10.0, k=2, max_docs=10)
    qvec = embedder.embed_query(QUERY)
    results = retriever.retrieve(qvec, k=2)
    assert isinstance(results, list)
    assert all("text" in r for r in results)


def test_dp_rag_retriever_top_p(built_indexer, embedder):
    from dp_methods.dp_rag import DPTopPRetriever

    retriever = DPTopPRetriever(
        index=built_indexer, epsilon=10.0, p=0.5, alpha=1.0, max_docs=10
    )
    qvec = embedder.embed_query(QUERY)
    results = retriever.retrieve(qvec, k=2)
    assert isinstance(results, list)


def test_dp_rag_per_doc_prompt_builder():
    from dp_methods.dp_rag import PerDocPromptBuilder

    pb = PerDocPromptBuilder()
    chunks = [{"text": "doc uno"}, {"text": "doc dos"}]
    prompt = pb.build(QUERY, chunks)
    parsed = json.loads(prompt)
    assert isinstance(parsed, list)
    # PerDocPromptBuilder produces k private prompts + 1 public = k+1 total
    assert len(parsed) >= len(chunks)


def test_dp_rag_generator_smoke(built_indexer, embedder):
    from dp_methods.dp_rag import DPRetriever, PerDocPromptBuilder, DPICLGenerator

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
    from dp_methods.dpvoterag import VoteRAGGenerator

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
    from dp_methods.dpvoterag import VoteRAGGenerator

    model, tokenizer = tiny_model_and_tokenizer
    gen = VoteRAGGenerator(
        model=model,
        tokenizer=tokenizer,
        m=2,
        k=1,
        epsilon_token=10.0,
        epsilon_total=20.0,
    )
    chunks = [{"text": d} for d in DOCS[:2]]
    prompt_json = json.dumps([f"Contexto: {c['text']}\nPregunta: {QUERY}\nRespuesta:" for c in chunks])
    result = gen.generate(prompt_json)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# DPSparseVoteRAG  (implemented ✅)
# ---------------------------------------------------------------------------

def test_dpsparsevoterag_generator_instantiates(tiny_model_and_tokenizer):
    from dp_methods.dpvoterag import SparseVoteRAGGenerator

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
# LPRAG  (not yet implemented — xfail expected)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="EntityPerturbedChunker not yet implemented", strict=False)
def test_lprag_chunker_smoke():
    from dp_methods.lprag.chunker import EntityPerturbedChunker  # type: ignore

    chunker = EntityPerturbedChunker(epsilon=1.0)
    chunks = chunker.chunk(DOCS[0])
    assert isinstance(chunks, list)
    assert len(chunks) > 0


# ---------------------------------------------------------------------------
# RP-RAG  (not yet implemented — xfail expected)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="RandomProjectionIndexer and RPRetriever not yet implemented", strict=False)
def test_rp_rag_smoke(embedder):
    from dp_methods.rp_rag.indexer import RandomProjectionIndexer  # type: ignore
    from dp_methods.rp_rag.retriever import RPRetriever  # type: ignore

    indexer = RandomProjectionIndexer(d_projected=16, random_seed=42)
    vectors = embedder.embed(DOCS)
    metadata = [{"doc_id": i, "chunk_id": 0, "text": d} for i, d in enumerate(DOCS)]
    indexer.build(vectors, metadata)

    retriever = RPRetriever(indexer=indexer)
    qvec = embedder.embed_query(QUERY)
    results = retriever.retrieve(qvec, k=2)
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# PRIVATE-RAG  (not yet implemented — xfail expected)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="BudgetManager and BudgetAwareRetriever not yet implemented", strict=False)
def test_private_rag_smoke(built_indexer, embedder):
    from dp_methods.private_rag.budget_manager import BudgetManager  # type: ignore
    from dp_methods.private_rag.retriever import BudgetAwareRetriever  # type: ignore

    budget_manager = BudgetManager(initial_budget=5.0)
    retriever = BudgetAwareRetriever(indexer=built_indexer, budget_manager=budget_manager)
    qvec = embedder.embed_query(QUERY)
    results = retriever.retrieve(qvec, k=2)
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# DP-SynRAG  (not yet implemented — xfail expected)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="DPSyntheticIndexer not yet implemented", strict=False)
def test_dp_synrag_smoke(embedder):
    from dp_methods.dp_synrag.indexer import DPSyntheticIndexer  # type: ignore

    indexer = DPSyntheticIndexer(epsilon=1.0, delta=1e-5, n_clusters=2)
    vectors = embedder.embed(DOCS)
    metadata = [{"doc_id": i, "chunk_id": 0, "text": d} for i, d in enumerate(DOCS)]
    indexer.build(vectors, metadata)
    assert indexer.get_index() is not None
