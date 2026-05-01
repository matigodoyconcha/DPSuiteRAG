from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    """Base class for document retrieval from a vector index.

    Retrievers are responsible for finding the most similar document chunks
    to a given query embedding. Implementations may use different similarity
    metrics and index structures (FAISS, Annoy, etc.).
    """

    @abstractmethod
    def retrieve(self, query_vector: Any, k: int) -> List[Dict]:
        """Retrieve the k most similar document chunks to the query.

        Args:
            query_vector: The embedding vector of the query.
            k: Number of top similar chunks to retrieve.

        Returns:
            List of retrieved document chunks with metadata, ordered by similarity.
        """
        pass
