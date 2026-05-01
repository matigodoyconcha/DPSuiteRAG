from typing import List
from core.chunker import BaseChunker
from core.embedder import BaseEmbedder
from core.indexer import BaseIndexer
from core.retriever import BaseRetriever
from core.reranker import BaseReranker
from core.prompt_builder import BasePromptBuilder
from core.generator import BaseGenerator
from core.postprocessor import BasePostprocessor

class RAGPipeline:
    """Orchestrates the complete Retrieval-Augmented Generation (RAG) pipeline.

    This pipeline coordinates multiple components to answer user queries by:
    1. Chunking documents and building an index during initialization
    2. Retrieving relevant chunks for a query
    3. Reranking for better relevance
    4. Building a prompt with the retrieved context
    5. Generating a response using the language model
    6. Post-processing the response for final delivery

    Attributes:
        chunker: Document chunking strategy.
        embedder: Text embedding model.
        indexer: Vector index management.
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
        retriever: BaseRetriever,
        reranker: BaseReranker,
        prompt_builder: BasePromptBuilder,
        generator: BaseGenerator,
        postprocessor: BasePostprocessor
    ):
        """Initialize the RAG pipeline with all required components.

        Args:
            chunker: Document chunking implementation.
            embedder: Text embedding model implementation.
            indexer: Vector index implementation.
            retriever: Document retrieval implementation.
            reranker: Chunk reranking implementation.
            prompt_builder: Prompt construction implementation.
            generator: Language model implementation.
            postprocessor: Response post-processing implementation.
        """
        self.chunker = chunker
        self.embedder = embedder
        self.indexer = indexer
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

        vectors = self.embedder.embed(all_chunks)
        self.indexer.build(vectors, all_metadata)

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
        query_vector = self.embedder.embed_query(query)
        retrieved_chunks = self.retriever.retrieve(query_vector, k)
        reranked_chunks = self.reranker.rerank(query, retrieved_chunks)
        prompt = self.prompt_builder.build(query, reranked_chunks)
        response = self.generator.generate(prompt)
        final_answer = self.postprocessor.process(response)

        return final_answer
