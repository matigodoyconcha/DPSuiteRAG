from typing import List, Dict, Any
import numpy as np
import faiss
from core.retriever import BaseRetriever

class CosineSimilarityRetriever(BaseRetriever):
    """Document retriever using cosine similarity with FAISS index.

    Performs k-nearest neighbor search using the inner product over normalized
    vectors, which is equivalent to cosine similarity.

    Attributes:
        indexer: FAISS indexer containing the vector index and metadata.
    """

    def __init__(self, indexer):
        """Initialize the retriever with a FAISS indexer.

        Args:
            indexer: FAISSIndexer instance containing the built index.
        """
        self.indexer = indexer

    def retrieve(self, query_vector: Any, k: int) -> List[Dict]:
        """Retrieve top-k most similar document chunks.

        Normalizes the query vector and searches the FAISS index for
        the k nearest neighbors by cosine similarity.

        Args:
            query_vector: Query embedding vector.
            k: Number of top results to retrieve.

        Returns:
            List of retrieved document chunks with scores and metadata.

        Raises:
            RuntimeError: If the index has not been built or loaded yet.
        """
        if self.indexer.index is None:
            raise RuntimeError("Index has not been built or loaded yet.")
            
        query_vector_np = np.array([query_vector]).astype('float32')
        faiss.normalize_L2(query_vector_np)
        
        scores, indices = self.indexer.index.search(query_vector_np, k)
        
        results = []
        for j, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.indexer.metadata):
                chunk_meta = self.indexer.metadata[idx].copy()
                chunk_meta["score"] = float(scores[0][j])
                results.append(chunk_meta)
                
        return results
