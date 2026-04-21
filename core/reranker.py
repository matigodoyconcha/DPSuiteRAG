from abc import ABC, abstractmethod
from typing import List, Dict

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        pass
