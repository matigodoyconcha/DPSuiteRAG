from typing import List, Dict
from core.reranker import BaseReranker

class IdentityReranker(BaseReranker):
    """No-op reranker that returns chunks in original order.

    This is a baseline reranker that does not modify the ranking.
    Useful for vanilla RAG pipelines without additional reranking.
    """

    def rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Return chunks unchanged.

        Args:
            query: The user's query (unused).
            chunks: List of retrieved chunks.

        Returns:
            The same list of chunks unchanged.
        """
        return chunks
