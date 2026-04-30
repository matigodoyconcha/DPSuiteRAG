"""Smoke tests for core abstract classes and RAGPipeline."""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.chunker import BaseChunker
from core.embedder import BaseEmbedder
from core.generator import BaseGenerator
from core.indexer import BaseIndexer
from core.pipeline import RAGPipeline
from core.postprocessor import BasePostprocessor
from core.prompt_builder import BasePromptBuilder
from core.reranker import BaseReranker
from core.retriever import BaseRetriever


# ---------------------------------------------------------------------------
# Stubs — minimal implementations satisfying each abstract contract
# ---------------------------------------------------------------------------

class StubChunker(BaseChunker):
    def chunk(self, document: str) -> list[str]:
        return [document]
    def get_config(self) -> dict:
        return {}

class StubEmbedder(BaseEmbedder):
    DIM = 8
    def embed(self, texts):
        return np.zeros((len(texts), self.DIM), dtype=np.float32)
    def embed_query(self, query: str):
        return np.zeros(self.DIM, dtype=np.float32)

class StubIndexer(BaseIndexer):
    def build(self, vectors, metadata):
        self._meta = metadata
    def save(self, path: str): pass
    def load(self, path: str): pass
    def get_index(self): return None

class StubRetriever(BaseRetriever):
    def retrieve(self, query_vector, k: int):
        return [{"text": "chunk", "score": 1.0}] * k

class StubReranker(BaseReranker):
    def rerank(self, query: str, chunks):
        return chunks

class StubPromptBuilder(BasePromptBuilder):
    def build(self, query: str, chunks) -> str:
        return f"Q: {query}"

class StubGenerator(BaseGenerator):
    def generate(self, prompt: str) -> str:
        return "respuesta"
    def generate_with_dp(self, prompts, epsilon, delta) -> str:
        return "respuesta_dp"

class StubPostprocessor(BasePostprocessor):
    def process(self, response: str) -> str:
        return response


# ---------------------------------------------------------------------------
# Abstract classes — direct instantiation must raise TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [
    BaseChunker, BaseEmbedder, BaseIndexer, BaseRetriever,
    BaseReranker, BasePromptBuilder, BaseGenerator, BasePostprocessor,
])
def test_abstract_classes_cannot_be_instantiated(cls):
    with pytest.raises(TypeError):
        cls()


# ---------------------------------------------------------------------------
# RAGPipeline with stubs
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline():
    indexer = StubIndexer()
    return RAGPipeline(
        chunker=StubChunker(),
        embedder=StubEmbedder(),
        indexer=indexer,
        retriever=StubRetriever(),
        reranker=StubReranker(),
        prompt_builder=StubPromptBuilder(),
        generator=StubGenerator(),
        postprocessor=StubPostprocessor(),
    )


def test_pipeline_index_runs_without_error(pipeline):
    pipeline.index(["Documento uno.", "Documento dos."])


def test_pipeline_query_returns_string(pipeline):
    pipeline.index(["Documento uno."])
    result = pipeline.query("¿Qué es esto?", k=1)
    assert isinstance(result, str)


def test_pipeline_query_without_index_still_runs(pipeline):
    result = pipeline.query("¿Qué es esto?", k=2)
    assert isinstance(result, str)


def test_pipeline_index_empty_documents(pipeline):
    pipeline.index([])


def test_pipeline_query_k_zero(pipeline):
    pipeline.index(["doc"])
    result = pipeline.query("q", k=0)
    assert isinstance(result, str)
