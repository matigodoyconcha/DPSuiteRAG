from typing import List

import numpy as np

from core.chunker import BaseChunker
from core.embedder import BaseEmbedder
from core.indexer import BaseIndexer
from core.query_embedder import BaseQueryEmbedder
from core.retriever import BaseRetriever
from core.reranker import BaseReranker
from core.prompt_builder import BasePromptBuilder
from core.generator import BaseGenerator
from core.postprocessor import BasePostprocessor

# Chunks embedded per streaming batch. A batch of 100k arctic (768-d) vectors is
# ~0.3 GiB, negligible next to the growing index/matrix, so the peak during a
# 21M-passage build is essentially just the index/matrix itself.
DEFAULT_INDEX_BATCH = 100_000


def build_index_streaming(embedder, indexer, all_chunks, all_metadata, batch_size: int = DEFAULT_INDEX_BATCH):
    """Embed *all_chunks* in batches and stream them into *indexer*.

    A one-shot ``indexer.build(embedder.embed(all_chunks), ...)`` holds the full
    (N, d) embedding array, a normalised copy of it, and the FAISS copy in memory
    at once — ~3x the index size, which OOM-kills a 21M-passage build on a 125 GiB
    host. Streaming embeds one batch at a time and adds it directly to the index,
    freeing it before the next, so peak memory is one batch on top of the index.

    Indexers without the incremental API (stubs, custom indexers) and empty corpora
    fall back to a single ``indexer.build(...)`` call, preserving prior behaviour
    exactly.
    """
    n = len(all_chunks)
    if n == 0 or not hasattr(indexer, "start_incremental"):
        indexer.build(embedder.embed(all_chunks), all_metadata)
        return
    started = False
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        vecs = np.asarray(embedder.embed(all_chunks[start:end]), dtype=np.float32)
        if not started:
            indexer.start_incremental(vecs.shape[1], total=n)
            started = True
        indexer.add_batch(vecs, all_metadata[start:end])
    indexer.finish_incremental()

class RAGPipeline:
    """Orchestrates the complete Retrieval-Augmented Generation (RAG) pipeline.

    The components are the stages of the pipeline, split across two phases.

    Offline, once per corpus, in ``index``:
    1. Chunking the documents
    2. Embedding the chunks
    3. Indexing the vectors

    Online, once per question, in ``query``:
    4. Embedding the query
    5. Retrieving relevant chunks from the index
    6. Reranking them for relevance, then building a prompt with the context
    7. Generating a response using the language model
    8. Post-processing the response for final delivery

    Stages 2 and 4 both embed text and a single-tower retriever runs them through the
    same weights, but they are separate components because they are separate stages:
    one reads the private corpus offline, the other reads the user's question online,
    and a mechanism placed on either one leaves the other untouched.

    Attributes:
        chunker: Document chunking strategy.
        embedder: Corpus embedding model.
        indexer: Vector index management.
        query_embedder: Query embedding model.
        retriever: Document retrieval mechanism.
        reranker: Relevance-based chunk reranking.
        prompt_builder: Prompt construction strategy.
        generator: Language model for response generation.
        postprocessor: Response formatting and cleaning.
    """

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        indexer: BaseIndexer,
        query_embedder: BaseQueryEmbedder,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        prompt_builder: BasePromptBuilder,
        generator: BaseGenerator,
        postprocessor: BasePostprocessor
    ):
        """Initialize the RAG pipeline with all required components.

        Args:
            chunker: Document chunking implementation.
            embedder: Corpus embedding model implementation.
            indexer: Vector index implementation.
            query_embedder: Query embedding model implementation.
            retriever: Document retrieval implementation.
            reranker: Chunk reranking implementation.
            prompt_builder: Prompt construction implementation.
            generator: Language model implementation.
            postprocessor: Response post-processing implementation.
        """
        self.chunker = chunker
        self.embedder = embedder
        self.indexer = indexer
        self.query_embedder = query_embedder
        self.retriever = retriever
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.postprocessor = postprocessor

    def index(self, documents: List[str]):
        """Build an index from a collection of documents.

        This method chunks all documents, embeds the chunks, and builds
        a searchable vector index for efficient retrieval.

        Args:
            documents: List of document texts to index.
        """
        all_chunks = []
        all_metadata = []

        for doc_idx, doc in enumerate(documents):
            chunks = self.chunker.chunk(doc)
            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunks.append(chunk_text)
                all_metadata.append({"doc_id": doc_idx, "chunk_id": chunk_idx, "text": chunk_text})

        build_index_streaming(self.embedder, self.indexer, all_chunks, all_metadata)

    def query(self, query: str, k: int) -> str:
        """Process a query through the RAG pipeline.

        Retrieves relevant documents, ranks them, builds a prompt,
        and generates a response.

        Args:
            query: User's query text.
            k: Number of top documents to retrieve.

        Returns:
            Generated answer based on retrieved context.
        """
        query_vector = self.query_embedder.embed_query(query)
        retrieved_chunks = self.retriever.retrieve(query_vector, k)
        reranked_chunks = self.reranker.rerank(query, retrieved_chunks)
        prompt = self.prompt_builder.build(query, reranked_chunks)
        response = self.generator.generate(prompt)
        final_answer = self.postprocessor.process(response)

        return final_answer
