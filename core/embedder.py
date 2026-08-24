from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbedder(ABC):
    """Base class for embedding the corpus that is indexed.

    Embedders convert document text into dense vector representations that capture
    semantic meaning. These embeddings are what the index is built from, so this
    stage runs offline, once per corpus. Embedding the query is a separate stage
    with a separate component (``BaseQueryEmbedder``): it runs online, per question,
    and asymmetric or two-tower retrievers encode questions differently from the
    passages they are matched against.
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
