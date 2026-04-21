from typing import List, Dict, Any
import numpy as np
import faiss
from core.retriever import BaseRetriever

class CosineSimilarityRetriever(BaseRetriever):
    def __init__(self, indexer):
        self.indexer = indexer

    def retrieve(self, query_vector: Any, k: int) -> List[Dict]:
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
