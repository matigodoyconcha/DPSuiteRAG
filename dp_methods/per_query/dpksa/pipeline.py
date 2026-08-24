from __future__ import annotations

from core.pipeline import RAGPipeline


class DPKSAPipeline(RAGPipeline):
    """RAG pipeline for DP-KSA (keyword selection and aggregation).

    Retrieves the top-``N`` documents, hands them to ``DPKSAGenerator.generate_ksa``
    (which runs sample-and-aggregate, FindBestK, TopKWithPTR and the final
    keyword-conditioned generation), and post-processes the answer. This does not
    fit the standard ``retrieve -> single generate`` flow, so the pipeline is
    subclassed (the permitted exception documented in the guide, as with
    ``DPVoteRAGPipeline``).
    """

    def query(self, query: str, k: int | None = None) -> str:
        query_vector = self.query_embedder.embed_query(query)
        retrieve_k = int(k) if k is not None else int(self.generator.n_docs)

        retrieved_chunks = self.retriever.retrieve(query_vector, retrieve_k)
        retrieved_docs = [
            str(chunk.get("text", "")).strip()
            for chunk in retrieved_chunks
            if str(chunk.get("text", "")).strip()
        ]

        response = self.generator.generate_ksa(query, retrieved_docs)
        return self.postprocessor.process(response)
