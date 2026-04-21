import os
import pickle
from typing import List, Dict, Any
import numpy as np
import faiss
from core.indexer import BaseIndexer

class FAISSIndexer(BaseIndexer):
    def __init__(self):
        self.index = None
        self.metadata = []

    def build(self, vectors: List[Any], metadata: List[Dict]):
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
        os.makedirs(path, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path: str):
        index_file = os.path.join(path, "index.faiss")
        meta_file = os.path.join(path, "metadata.pkl")
        
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
        if os.path.exists(meta_file):
            with open(meta_file, "rb") as f:
                self.metadata = pickle.load(f)

    def get_index(self) -> Any:
        return self.index
