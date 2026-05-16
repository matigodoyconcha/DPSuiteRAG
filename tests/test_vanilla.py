"""Smoke tests for the vanilla RAG pipeline components."""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vanilla.chunker import FixedSizeChunker, SentenceChunker
from vanilla.embedder import HFEmbedder
from vanilla.indexer import FAISSIndexer
from vanilla.retriever import CosineSimilarityRetriever
from vanilla.reranker import IdentityReranker
from vanilla.prompt_builder import SimplePromptBuilder
from vanilla.postprocessor import IdentityPostprocessor
from core.pipeline import RAGPipeline

EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DOCS = [
    "Differential Privacy adds calibrated noise to protect sensitive data.",
    "FAISS enables efficient similarity search over vectors.",
    "A RAG pipeline combines retrieval and text generation.",
]
QUERY = "What is differential privacy?"


# ---------------------------------------------------------------------------
# Chunkers
# ---------------------------------------------------------------------------

def test_fixed_size_chunker_basic():
    chunker = FixedSizeChunker(chunk_size=5, overlap=1)
    chunks = chunker.chunk("uno dos tres cuatro cinco seis siete")
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)


def test_fixed_size_chunker_empty_document():
    chunker = FixedSizeChunker(chunk_size=10, overlap=2)
    assert chunker.chunk("") == []


def test_fixed_size_chunker_get_config():
    chunker = FixedSizeChunker(chunk_size=128, overlap=16)
    cfg = chunker.get_config()
    assert cfg["chunk_size"] == 128
    assert cfg["overlap"] == 16


def test_sentence_chunker_basic():
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        pytest.skip("NLTK punkt_tab not downloaded — run: python -m nltk.downloader punkt_tab")
    chunker = SentenceChunker(max_sentences=2, overlap=0)
    chunks = chunker.chunk("Sentence one. Sentence two. Sentence three.")
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)


# ---------------------------------------------------------------------------
# Embedder (downloads all-MiniLM-L6-v2 on first run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder():
    return HFEmbedder(model_name=EMBEDDER_MODEL)


def test_embedder_embed_shape(embedder):
    vectors = embedder.embed(DOCS)
    assert vectors.ndim == 2
    assert vectors.shape[0] == len(DOCS)
    assert vectors.shape[1] > 0


def test_embedder_embed_query_shape(embedder):
    vec = embedder.embed_query(QUERY)
    assert vec.ndim == 1
    assert vec.shape[0] > 0


def test_embedder_embed_empty_list(embedder):
    result = embedder.embed([])
    assert result.shape[0] == 0


# ---------------------------------------------------------------------------
# Indexer + Retriever
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def built_indexer(embedder):
    indexer = FAISSIndexer()
    vectors = embedder.embed(DOCS)
    metadata = [{"doc_id": i, "chunk_id": 0, "text": d} for i, d in enumerate(DOCS)]
    indexer.build(vectors, metadata)
    return indexer


def test_faiss_indexer_builds(built_indexer):
    assert built_indexer.index is not None


def test_faiss_indexer_get_index(built_indexer):
    assert built_indexer.get_index() is not None


def test_cosine_retriever_returns_k_results(built_indexer, embedder):
    retriever = CosineSimilarityRetriever(indexer=built_indexer)
    qvec = embedder.embed_query(QUERY)
    results = retriever.retrieve(qvec, k=2)
    assert len(results) == 2
    assert all("text" in r and "score" in r for r in results)


def test_cosine_retriever_scores_are_floats(built_indexer, embedder):
    retriever = CosineSimilarityRetriever(indexer=built_indexer)
    qvec = embedder.embed_query(QUERY)
    results = retriever.retrieve(qvec, k=1)
    assert isinstance(results[0]["score"], float)


# ---------------------------------------------------------------------------
# Pass-through components
# ---------------------------------------------------------------------------

def test_identity_reranker_preserves_order():
    reranker = IdentityReranker()
    chunks = [{"text": "a", "score": 0.9}, {"text": "b", "score": 0.5}]
    assert reranker.rerank("q", chunks) == chunks


def test_simple_prompt_builder_contains_query():
    pb = SimplePromptBuilder()
    prompt = pb.build("my question", [{"text": "relevant context"}])
    assert "my question" in prompt
    assert "relevant context" in prompt


def test_identity_postprocessor_passthrough():
    pp = IdentityPostprocessor()
    assert pp.process("final answer") == "final answer"


# ---------------------------------------------------------------------------
# Pipeline end-to-end with tiny-gpt2
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vanilla_pipeline(embedder, built_indexer):
    from bench_llms import load_llm
    llm = load_llm({"name": "huggingface", "model_name": "sshleifer/tiny-gpt2"})
    generator = llm.build_generator(max_new_tokens=8)
    return RAGPipeline(
        chunker=FixedSizeChunker(chunk_size=50, overlap=5),
        embedder=embedder,
        indexer=built_indexer,
        retriever=CosineSimilarityRetriever(indexer=built_indexer),
        reranker=IdentityReranker(),
        prompt_builder=SimplePromptBuilder(),
        generator=generator,
        postprocessor=IdentityPostprocessor(),
    )


def test_vanilla_pipeline_query_returns_string(vanilla_pipeline):
    result = vanilla_pipeline.query(QUERY, k=2)
    assert isinstance(result, str)
