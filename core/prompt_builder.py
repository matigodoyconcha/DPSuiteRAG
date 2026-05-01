from abc import ABC, abstractmethod
from typing import List, Dict

class BasePromptBuilder(ABC):
    """Base class for prompt builders in the RAG pipeline.

    Prompt builders are responsible for constructing prompts from a query
    and retrieved document chunks. Implementations may vary in how they
    format and structure the input for the language model generator.
    """

    @abstractmethod
    def build(self, query: str, chunks: List[Dict]) -> str:
        """Build a prompt from a query and retrieved document chunks.

        Args:
            query: The user's query string.
            chunks: List of retrieved document chunks with metadata.
                   Each chunk is a dictionary containing 'text' and other metadata.

        Returns:
            A formatted prompt string ready for the language model generator.
        """
        pass
