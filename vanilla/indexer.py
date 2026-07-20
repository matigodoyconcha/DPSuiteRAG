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

    def __init__(self, store_embeddings_matrix: bool = True, build_faiss_index: bool = True):
        # store_embeddings_matrix=False skips keeping/saving the (N, d) fp32 matrix of
        # every document embedding. Only the all-document BLAS scoring in the DP-RAG /
        # Private-RAG retrievers needs it; methods that retrieve via FAISS top-k search
        # (CosineSimilarityRetriever) never touch it, so dropping it saves N*d*4 bytes
        # of RAM (~60 GiB at 21M docs) plus the equally large embeddings.npy on disk.
        #
        # build_faiss_index=False skips the IndexFlatIP altogether. The DP-RAG /
        # Private-RAG retrievers score every document through embeddings_matrix and
        # never call index.search, so for them the FAISS index is dead weight — a
        # second ~N*d*4 bytes (~60 GiB at 21M docs). It is required, and kept by
        # default, for every retriever that does a FAISS top-k / exhaustive search
        # (vanilla, DP-KSA, DP-VoteRAG, DP-SparseVoteRAG).
        self.store_embeddings_matrix = bool(store_embeddings_matrix)
        self.build_faiss_index = bool(build_faiss_index)
        self.index = None
        self.metadata: list = []
        self.embeddings_matrix: np.ndarray | None = None
        self._pending = 0
        self._expected = 0

    def build(self, vectors: List[Any], metadata: List[Dict]):
        """Build the index in one shot from already-embedded vectors.

        Thin wrapper over the incremental API (:meth:`start_incremental`,
        :meth:`add_batch`, :meth:`finish_incremental`) for small corpora and
        callers that already hold every embedding in memory. Large corpora should
        stream through the incremental API (see ``core.pipeline.build_index_streaming``)
        so the full (N, d) array and the FAISS copy are never both resident at once.

        Normalises vectors for cosine similarity (IndexFlatIP on L2-normalised
        vectors). The normalised matrix is kept in ``embeddings_matrix`` when
        ``store_embeddings_matrix`` is set so all-document scoring is a direct numpy
        multiply instead of a FAISS search.

        Args:
            vectors: Embedding vectors to index (list or (N, d) array).
            metadata: Corresponding metadata for each vector (same order).

        Raises:
            ValueError: If vectors is empty.
        """
        if vectors is None or len(vectors) == 0:
            raise ValueError("Empty vectors list passed to indexer.")
        # Force a copy so add_batch's in-place normalisation never mutates a caller's
        # array (large streaming callers pass freshly embedded batches instead).
        vectors_np = np.array(vectors, dtype=np.float32)
        self.start_incremental(vectors_np.shape[1], total=vectors_np.shape[0])
        self.add_batch(vectors_np, metadata)
        self.finish_incremental()

    def start_incremental(self, dim: int, total: int):
        """Begin an incremental build of *total* vectors of dimension *dim*.

        Allocates the FAISS index (unless ``build_faiss_index`` is False) and, when
        the embeddings matrix is kept, pre-allocates it at full size so batches fill
        it in place — no growing array and no second full-size copy. Follow with
        repeated :meth:`add_batch` calls and a final :meth:`finish_incremental`.
        """
        self.index = faiss.IndexFlatIP(int(dim)) if self.build_faiss_index else None
        self.metadata = []
        self._pending = 0
        self._expected = int(total)
        self.embeddings_matrix = (
            np.empty((int(total), int(dim)), dtype=np.float32)
            if self.store_embeddings_matrix
            else None
        )

    def add_batch(self, vectors: Any, metadata: List[Dict]):
        """Add one batch of embeddings (and their metadata) to the growing index.

        The batch is L2-normalised **in place** (for cosine similarity) before being
        added to the FAISS index and/or written into the pre-allocated embeddings
        matrix. Peak memory holds only this batch on top of the index/matrix, so a
        21M-passage corpus never materialises a second full-size copy.
        """
        vectors_np = np.asarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors_np)
        n = vectors_np.shape[0]
        if self.index is not None:
            self.index.add(vectors_np)
        if self.embeddings_matrix is not None:
            self.embeddings_matrix[self._pending : self._pending + n] = vectors_np
        self._pending += n
        self.metadata.extend(metadata)

    def finish_incremental(self):
        """Finalise an incremental build started with :meth:`start_incremental`.

        Trims the pre-allocated embeddings matrix if fewer vectors than announced
        arrived (defensive; normally the counts match exactly).
        """
        if self.embeddings_matrix is not None and self._pending != self._expected:
            self.embeddings_matrix = self.embeddings_matrix[: self._pending]

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
