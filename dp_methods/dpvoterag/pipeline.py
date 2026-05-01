from __future__ import annotations

from core.pipeline import RAGPipeline


class DPVoteRAGPipeline(RAGPipeline):
    """RAG pipeline with voting-based differential privacy.

    This pipeline extends the base RAG pipeline to implement differential privacy
    through voting aggregation. Instead of using a single retrieved document,
    it retrieves multiple documents and uses voting to aggregate model responses
    while satisfying differential privacy guarantees.
    """

    def query(self, query: str, k: int | None = None) -> str:
        """Process a query with voting-based differential privacy.

        Retrieves multiple documents and uses voting on model responses
        to provide differentially private answers.

        Args:
            query: User's query text.
            k: Number of document chunks per vote round. If None, uses
               generator.m * generator.k (total retrieval multiplied by rounds).

        Returns:
            Differentially private answer aggregated through voting.
        """
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
