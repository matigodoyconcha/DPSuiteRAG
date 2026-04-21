from abc import ABC, abstractmethod
from typing import List

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def generate_with_dp(self, prompts: List[str], epsilon: float, delta: float) -> str:
        pass
