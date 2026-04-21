from abc import ABC, abstractmethod
from typing import List, Dict

class BasePromptBuilder(ABC):
    @abstractmethod
    def build(self, query: str, chunks: List[Dict]) -> str:
        pass
