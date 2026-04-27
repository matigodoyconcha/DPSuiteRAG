from __future__ import annotations

from core.pipeline import RAGPipeline


class DPVoteRAGPipeline(RAGPipeline):
    def query(self, query: str, k: int | None = None) -> str:
        query_vector = self.embedder.embed_query(query)

        if k is None:
            retrieve_k = int(self.generator.m * self.generator.k)
        else:
            retrieve_k = int(k)

        retrieved_chunks = self.retriever.retrieve(query_vector, retrieve_k)
        retrieved_docs = [str(chunk.get("text", "")).strip() for chunk in retrieved_chunks if str(chunk.get("text", "")).strip()]

        response = self.generator.generate_vote(query, retrieved_docs)
        final_answer = self.postprocessor.process(response)
        return final_answer
