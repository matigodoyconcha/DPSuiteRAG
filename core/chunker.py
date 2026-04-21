from abc import ABC, abstractmethod

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: str) -> list[str]:
        pass

    @abstractmethod
    def get_config(self) -> dict:
        pass
