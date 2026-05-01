import os
import pickle
from typing import List, Dict, Any
import numpy as np
import faiss
from core.indexer import BaseIndexer


class FAISSIndexer(BaseIndexer):
    """Vector index using Facebook AI Similarity Search (FAISS).

    Builds and manages a FAISS index for efficient similarity search over
    document embeddings using cosine similarity (IndexFlatIP on L2-normalised
    vectors).

    The L2-normalised embedding matrix is stored separately as a plain numpy
    array (``embeddings_matrix``).  The DP-RAG retriever uses this matrix to
    score **all N documents** with a single BLAS matrix-vector multiply, which
    is required for the exponential mechanism utility function to be correct.
    (See ``dp_methods/dp_rag/retriever.py`` for the DP justification.)

    Attributes:
        index: FAISS index object (IndexFlatIP).
        metadata: List of metadata dictionaries, one per indexed vector.
        embeddings_matrix: Float32 numpy array of shape (N, d) containing the
            L2-normalised embedding of every indexed document.  Persisted
            alongside the FAISS index so the retriever can load it without
            rebuilding.
    """

    def __init__(self):
        self.index = None
        self.metadata: list = []
        self.embeddings_matrix: np.ndarray | None = None

    def build(self, vectors: List[Any], metadata: List[Dict]):
        """Build a FAISS index from embedding vectors.

        Normalises vectors for cosine similarity and constructs an IndexFlatIP.
        The normalised matrix is also kept in ``embeddings_matrix`` so that
        all-document scoring can be done with a direct numpy multiply instead
        of a FAISS search call.

        Args:
            vectors: List of embedding vectors to index.
            metadata: Corresponding metadata for each vector (same order).

        Raises:
            ValueError: If vectors list is empty.
        """
        if vectors is None or len(vectors) == 0:
            raise ValueError("Empty vectors list passed to indexer.")

        vectors_np = np.array(vectors, dtype=np.float32)
        d = vectors_np.shape[1]

        faiss.normalize_L2(vectors_np)

        self.index = faiss.IndexFlatIP(d)
        self.index.add(vectors_np)
        self.metadata = metadata
        # Keep a reference so the retriever can score all N docs in one call.
        self.embeddings_matrix = vectors_np

    def save(self, path: str):
        """Save the FAISS index, metadata, and embeddings matrix to disk.

        Args:
            path: Directory path where files will be saved.
        """
        os.makedirs(path, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        if self.embeddings_matrix is not None:
            np.save(os.path.join(path, "embeddings.npy"), self.embeddings_matrix)

    def load(self, path: str):
        """Load a FAISS index, metadata, and embeddings matrix from disk.

        ``embeddings_matrix`` is loaded when ``embeddings.npy`` is present.
        Indices built before that file was introduced (old caches) will have
        ``embeddings_matrix = None``; the DP retriever falls back to a FAISS
        exhaustive search in that case.

        Args:
            path: Directory path containing saved files.
        """
        index_file = os.path.join(path, "index.faiss")
        meta_file = os.path.join(path, "metadata.pkl")
        emb_file = os.path.join(path, "embeddings.npy")

        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
        if os.path.exists(meta_file):
            with open(meta_file, "rb") as f:
                self.metadata = pickle.load(f)
        if os.path.exists(emb_file):
            self.embeddings_matrix = np.load(emb_file)

    def get_index(self) -> Any:
        """Return the underlying FAISS index object."""
        return self.index
