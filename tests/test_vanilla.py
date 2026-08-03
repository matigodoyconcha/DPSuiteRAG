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
from core.pipeline import RAGPipeline, build_index_streaming

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
# Incremental / streaming index build
# ---------------------------------------------------------------------------

def _metadata_for(texts):
    return [{"doc_id": i, "chunk_id": 0, "text": t} for i, t in enumerate(texts)]


def test_incremental_build_matches_one_shot(embedder):
    """start/add_batch/finish must produce the same index as a one-shot build."""
    vectors = embedder.embed(DOCS)
    meta = _metadata_for(DOCS)

    one_shot = FAISSIndexer()
    one_shot.build(vectors, meta)

    incremental = FAISSIndexer()
    incremental.start_incremental(vectors.shape[1], total=len(DOCS))
    incremental.add_batch(embedder.embed(DOCS[:2]), meta[:2])
    incremental.add_batch(embedder.embed(DOCS[2:]), meta[2:])
    incremental.finish_incremental()

    assert incremental.index.ntotal == one_shot.index.ntotal == len(DOCS)
    assert incremental.metadata == one_shot.metadata
    assert np.allclose(incremental.embeddings_matrix, one_shot.embeddings_matrix, atol=1e-5)


def test_build_faiss_index_false_skips_index(embedder):
    """Matrix methods (DP-RAG / Private-RAG) keep the matrix but no FAISS index."""
    vecs = embedder.embed(DOCS)
    indexer = FAISSIndexer(build_faiss_index=False)
    indexer.build(vecs, _metadata_for(DOCS))

    assert indexer.index is None
    assert indexer.get_index() is None
    assert indexer.embeddings_matrix is not None
    assert indexer.embeddings_matrix.shape == (len(DOCS), vecs.shape[1])


def test_matrix_only_index_save_load_roundtrip(embedder, tmp_path):
    """A matrix-only index round-trips through disk without an index.faiss file."""
    built = FAISSIndexer(build_faiss_index=False)
    built.build(embedder.embed(DOCS), _metadata_for(DOCS))
    built.save(str(tmp_path))

    assert not (tmp_path / "index.faiss").exists()
    assert (tmp_path / "embeddings.npy").exists()

    loaded = FAISSIndexer(build_faiss_index=False)
    loaded.load(str(tmp_path))
    assert loaded.index is None
    assert loaded.metadata == built.metadata
    assert np.allclose(loaded.embeddings_matrix, built.embeddings_matrix)


def test_build_index_streaming_matches_one_shot(embedder):
    """Streaming across a batch boundary equals a single build() call."""
    meta = _metadata_for(DOCS)

    one_shot = FAISSIndexer()
    one_shot.build(embedder.embed(DOCS), meta)

    streamed = FAISSIndexer()
    build_index_streaming(embedder, streamed, list(DOCS), meta, batch_size=2)

    assert streamed.index.ntotal == len(DOCS)
    assert streamed.metadata == meta
    assert np.allclose(streamed.embeddings_matrix, one_shot.embeddings_matrix, atol=1e-5)


def test_build_index_streaming_matrix_free_builds_faiss_only(embedder):
    """A matrix-free streamed build populates the FAISS index and keeps no matrix."""
    streamed = FAISSIndexer(store_embeddings_matrix=False)
    build_index_streaming(embedder, streamed, list(DOCS), _metadata_for(DOCS), batch_size=2)

    assert streamed.index.ntotal == len(DOCS)
    assert streamed.embeddings_matrix is None


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


class TestContextFitsTheModel:
    """The context the model reads must be the context that was ranked highest."""

    TEMPLATE = "Context: {context}\n\nQuestion: {query}\nAnswer:"

    def _builder(self, budget):
        from vanilla.prompt_builder import SimplePromptBuilder

        return SimplePromptBuilder(
            template=self.TEMPLATE,
            token_budget=budget,
            count_tokens=lambda text: len(text.split()),
        )

    def _chunks(self, n):
        # Best-ranked first, as the retriever returns them.
        return [{"text": f"doc{i} " + "word " * 8} for i in range(n)]

    def test_the_best_ranked_documents_are_the_ones_kept(self):
        prompt = self._builder(40).build("q?", self._chunks(20))
        assert "doc0" in prompt and "doc1" in prompt
        assert "doc19" not in prompt

    def test_it_stops_at_the_budget(self):
        builder = self._builder(40)
        prompt = builder.build("q?", self._chunks(20))
        assert len(prompt.split()) <= 40
        assert 0 < builder.last_kept < 20

    def test_everything_fits_when_the_budget_allows(self):
        builder = self._builder(10_000)
        builder.build("q?", self._chunks(20))
        assert builder.last_kept == 20

    def test_no_budget_keeps_the_previous_behaviour(self):
        from vanilla.prompt_builder import SimplePromptBuilder

        chunks = self._chunks(20)
        plain = SimplePromptBuilder(template=self.TEMPLATE).build("q?", chunks)
        assert all(f"doc{i}" in plain for i in range(20))

    def test_a_single_oversized_document_does_not_wedge_the_prompt(self):
        # Nothing fits, but the question must still reach the model.
        builder = self._builder(5)
        prompt = builder.build("q?", [{"text": "word " * 500}])
        assert builder.last_kept == 0
        assert "q?" in prompt
