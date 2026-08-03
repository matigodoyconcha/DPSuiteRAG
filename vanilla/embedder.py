from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from core.device import cuda_oom_warning  # also ensures load_dotenv() runs
from core.embedder import BaseEmbedder

class HFEmbedder(BaseEmbedder):
    """Embedding model using HuggingFace sentence-transformers.

    Provides efficient embedding generation using pre-trained transformer models
    from the sentence-transformers library.
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
        # Query embeddings are a deterministic function of the query text under this
        # instance's model, so repeats of the same query resolve from here. The cache
        # belongs to the instance because the same text embeds differently under a
        # different model, and it is bounded because callers sweep whole datasets of
        # distinct queries through it.
        self._query_cache: dict[str, np.ndarray] = {}

    _QUERY_CACHE_MAX = 4096

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

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query.

        Args:
            query: Query string to embed.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        cached = self._query_cache.get(query)
        if cached is None:
            # Asymmetric retrievers (e.g. arctic-embed) register a "query" prompt that
            # prefixes the query but not the documents; their default_prompt_name is null,
            # so encode() applies it only when asked. Symmetric models register no such
            # prompt, so guard on its presence and fall back to a bare encode.
            kwargs = {}
            prompts = getattr(self.model, "prompts", None)
            if prompts and "query" in prompts:
                kwargs["prompt_name"] = "query"
            cached = self.model.encode([query], convert_to_numpy=True, **kwargs)[0].astype(np.float32)
            if len(self._query_cache) >= self._QUERY_CACHE_MAX:
                # Insertion-ordered dicts make the oldest entry the first key.
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[query] = cached
        # Callers normalise the vector they receive in place (FAISS retrieval does),
        # so each call gets its own array and the stored one stays as encoded.
        return cached.copy()


def build_embedder(embedder_model: str) -> BaseEmbedder:
    """Construct the (single-tower) embedder for a run."""
    return HFEmbedder(model_name=embedder_model)


def embedder_identity(embedder_model: str) -> str:
    """Stable identity string for index cache keys."""
    return embedder_model
