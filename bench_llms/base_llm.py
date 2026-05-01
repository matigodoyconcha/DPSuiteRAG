from abc import ABC, abstractmethod
from core.generator import BaseGenerator


class BaseLLM(ABC):
    """Base class for language model wrappers.

    Provides a unified interface for different language model backends
    used in the RAG pipeline benchmarks.
    """

    @abstractmethod
    def build_generator(self, max_new_tokens: int) -> BaseGenerator:
        """Create a generator instance from this LLM.

        Args:
            max_new_tokens: Maximum tokens to generate.

        Returns:
            A BaseGenerator instance wrapping this language model.
        """
        pass

    @abstractmethod
    def get_config(self) -> dict[str, str]:
        """Get configuration details of this language model.

        Returns:
            Dictionary containing model configuration.
        """
        pass
