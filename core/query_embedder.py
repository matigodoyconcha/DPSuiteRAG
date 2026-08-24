from abc import ABC, abstractmethod

import numpy as np


class BaseQueryEmbedder(ABC):
    """Base class for embedding the user query at inference time.

    Query embedding is a stage of its own, distinct from the corpus embedding that
    feeds the index. The two run in different phases and on different data: the
    corpus is embedded once, offline, over the private documents, while the query is
    embedded online, once per question, over text the user supplies. A mechanism that
    protects the query therefore attaches here and nowhere else, and a retriever is
    free to encode questions differently from passages — asymmetric models prefix the
    query with a prompt the documents never receive, and two-tower retrievers use a
    different set of weights entirely.

    Sharing one model between the two stages is a property of the retriever being
    single-tower, not of the pipeline, so it is expressed by the implementation
    rather than by folding both into one component.
    """

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
