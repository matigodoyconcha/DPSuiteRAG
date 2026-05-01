from abc import ABC, abstractmethod
from typing import List

class BaseGenerator(ABC):
    """Base class for language model generation.

    Generators are responsible for producing text responses from prompts.
    Implementations support both standard generation and differentially private
    generation that combines outputs from multiple prompts.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from a single prompt.

        Args:
            prompt: The input prompt for the language model.

        Returns:
            Generated text response.
        """
        pass

    @abstractmethod
    def generate_with_dp(self, prompts: List[str], epsilon: float, delta: float) -> str:
        """Generate a differentially private response from multiple prompts.

        This method implements differential privacy by combining outputs from
        multiple prompts using voting or other aggregation mechanisms.

        Args:
            prompts: List of prompts to generate from.
            epsilon: Privacy budget parameter (lower epsilon = more privacy).
            delta: Privacy parameter for (epsilon, delta)-differential privacy.

        Returns:
            A single aggregated response that satisfies differential privacy.
        """
        pass
