from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query_vector: Any, k: int) -> List[Dict]:
        pass
