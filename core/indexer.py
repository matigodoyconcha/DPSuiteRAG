from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseIndexer(ABC):
    """Base class for vector index management.

    Indexers are responsible for building and maintaining vector indices
    that enable fast similarity search over document embeddings. Implementations
    may use different index structures (FAISS, Annoy, LSH, etc.).
    """

    @abstractmethod
    def build(self, vectors: List[Any], metadata: List[Dict]):
        """Build an index from document embeddings and metadata.

        Args:
            vectors: List of document embedding vectors.
            metadata: List of metadata dictionaries corresponding to each vector.
                     Each dictionary contains information like document ID and text.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save the index to disk.

        Args:
            path: File path where the index should be saved.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """Load an index from disk.

        Args:
            path: File path to the saved index.
        """
        pass

    @abstractmethod
    def get_index(self) -> Any:
        """Get the underlying index object.

        Returns:
            The index data structure used for similarity search.
        """
        pass
