from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from core.device import cuda_oom_warning  # also ensures load_dotenv() runs
from core.embedder import BaseEmbedder

class HFEmbedder(BaseEmbedder):
    """Corpus embedding model using HuggingFace sentence-transformers.

    Provides efficient embedding generation using pre-trained transformer models
    from the sentence-transformers library. Queries go through ``HFQueryEmbedder``,
    which reuses the model loaded here.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the HuggingFace embedder.

        Args:
            model_name: Model identifier from HuggingFace model hub.
                       Defaults to all-MiniLM-L6-v2, a lightweight model.
        """
        self.model_name = model_name
        with cuda_oom_warning(model_name):
            self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            2D numpy array of embeddings with shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self.model.encode(texts, convert_to_numpy=True).astype(np.float32)


def build_embedder(embedder_model: str) -> BaseEmbedder:
    """Construct the corpus embedder for a run."""
    return HFEmbedder(model_name=embedder_model)


def embedder_identity(embedder_model: str) -> str:
    """Stable identity string for index cache keys."""
    return embedder_model
