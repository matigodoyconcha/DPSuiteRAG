from abc import ABC, abstractmethod
from typing import List, Dict

class BaseReranker(ABC):
    """Base class for reranking retrieved document chunks.

    Rerankers are responsible for reordering retrieved chunks based on their
    relevance to the query. This improves the quality of the prompt by
    prioritizing the most relevant documents for the generator.
    """

    @abstractmethod
    def rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Rerank retrieved document chunks by relevance to the query.

        Args:
            query: The user's query string.
            chunks: List of retrieved document chunks to rerank.

        Returns:
            Reranked list of document chunks, ordered by relevance.
        """
        pass
