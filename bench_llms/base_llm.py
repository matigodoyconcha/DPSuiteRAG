from abc import ABC, abstractmethod
from core.generator import BaseGenerator


class BaseLLM(ABC):
    @abstractmethod
    def build_generator(self, max_new_tokens: int) -> BaseGenerator:
        pass

    @abstractmethod
    def get_config(self) -> dict[str, str]:
        pass
