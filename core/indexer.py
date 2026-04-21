from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseIndexer(ABC):
    @abstractmethod
    def build(self, vectors: List[Any], metadata: List[Dict]):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def get_index(self) -> Any:
        pass
