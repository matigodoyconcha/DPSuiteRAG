from abc import ABC, abstractmethod
from typing import List, Any

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[Any]:
        pass

    @abstractmethod
    def embed_query(self, query: str) -> Any:
        pass
