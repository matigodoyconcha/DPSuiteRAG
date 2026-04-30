from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Returns a 2D array of shape (len(texts), embedding_dim)."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Returns a 1D array of shape (embedding_dim,)."""
        pass
