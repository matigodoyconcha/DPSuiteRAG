from typing import List
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from core.embedder import BaseEmbedder

load_dotenv()

class HFEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self.model.encode(texts, convert_to_numpy=True).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
