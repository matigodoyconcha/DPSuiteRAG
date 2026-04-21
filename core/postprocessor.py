from abc import ABC, abstractmethod

class BasePostprocessor(ABC):
    @abstractmethod
    def process(self, response: str) -> str:
        pass
