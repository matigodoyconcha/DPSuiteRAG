from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbedder(ABC):
    """Base class for text embedding models.

    Embedders convert text into dense vector representations that capture
    semantic meaning. These embeddings are used for similarity-based
    document retrieval in the RAG pipeline.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts into vector representations.

        Args:
            texts: List of text strings to embed.

        Returns:
            2D numpy array of shape (len(texts), embedding_dim) containing
            the embedding vectors for each input text.
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            query: The query text to embed.

        Returns:
            1D numpy array of shape (embedding_dim,) containing
            the embedding vector for the query.
        """
        pass
