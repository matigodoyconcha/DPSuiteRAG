import os
import pickle
from typing import List, Dict, Any
import numpy as np
import faiss
from core.indexer import BaseIndexer

class FAISSIndexer(BaseIndexer):
    """Vector index using Facebook AI Similarity Search (FAISS).

    Builds and manages a FAISS index for efficient similarity search over
    document embeddings using cosine similarity.

    Attributes:
        index: FAISS index object for similarity search.
        metadata: List of metadata dictionaries for retrieved vectors.
    """

    def __init__(self):
        """Initialize an empty FAISS indexer."""
        self.index = None
        self.metadata = []

    def build(self, vectors: List[Any], metadata: List[Dict]):
        """Build a FAISS index from embedding vectors.

        Normalizes vectors for cosine similarity and constructs an IndexFlatIP
        (inner product index) for efficient retrieval.

        Args:
            vectors: List of embedding vectors to index.
            metadata: Corresponding metadata for each vector.

        Raises:
            ValueError: If vectors list is empty.
        """
        if vectors is None or len(vectors) == 0:
            raise ValueError("Empty vectors list passed to indexer.")
            
        vectors_np = np.array(vectors).astype('float32')
        d = vectors_np.shape[1]
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors_np)
        
        self.index = faiss.IndexFlatIP(d)
        self.index.add(vectors_np)
        self.metadata = metadata

    def save(self, path: str):
        """Save the FAISS index and metadata to disk.

        Args:
            path: Directory path where index and metadata will be saved.
        """
        os.makedirs(path, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path: str):
        """Load a FAISS index and metadata from disk.

        Args:
            path: Directory path containing saved index and metadata.
        """
        index_file = os.path.join(path, "index.faiss")
        meta_file = os.path.join(path, "metadata.pkl")

        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
        if os.path.exists(meta_file):
            with open(meta_file, "rb") as f:
                self.metadata = pickle.load(f)

    def get_index(self) -> Any:
        """Get the underlying FAISS index object.

        Returns:
            The FAISS index for direct access if needed.
        """
        return self.index
