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
        self.chunker = chunker
        self.embedder = embedder
        self.indexer = indexer
        self.retriever = retriever
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.postprocessor = postprocessor

    def index(self, documents: List[str]):
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
        query_vector = self.embedder.embed_query(query)
        retrieved_chunks = self.retriever.retrieve(query_vector, k)
        reranked_chunks = self.reranker.rerank(query, retrieved_chunks)
        prompt = self.prompt_builder.build(query, reranked_chunks)
        response = self.generator.generate(prompt)
        final_answer = self.postprocessor.process(response)
        
        return final_answer
