from typing import List, Any
import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from core.embedder import BaseEmbedder

load_dotenv()

class HFEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[Any]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def embed_query(self, query: str) -> Any:
        return self.model.encode([query], convert_to_numpy=True)[0]
