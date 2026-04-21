from typing import List, Dict
from core.reranker import BaseReranker

class IdentityReranker(BaseReranker):
    def rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        return chunks
