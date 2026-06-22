from bench_llms.base_llm import BaseLLM
from core.generator import BaseGenerator
from vanilla.generator import HFGenerator


class HuggingFaceLLM(BaseLLM):
    """Language model wrapper for HuggingFace transformer models.

    Wraps any HuggingFace causal language model as a generator for
    use in the RAG pipeline.

    Attributes:
        model_name: HuggingFace model identifier.
    """

    def __init__(self, model_name: str = "sshleifer/tiny-gpt2"):
        """Initialize the HuggingFace LLM wrapper.

        Args:
            model_name: HuggingFace model identifier (e.g., "gpt2", "EleutherAI/gpt-neo-125M").
        """
        self.model_name = model_name

    def build_generator(
        self,
        max_new_tokens: int,
        use_chat_template: bool | str = "auto",
        stop_strings: list[str] | None = None,
    ) -> BaseGenerator:
        """Create a generator from this HuggingFace model.

        Args:
            max_new_tokens: Maximum tokens to generate.
            use_chat_template: C2 — apply the model chat template when present.
            stop_strings: C3 — optional stop strings to truncate output (OFF by default).

        Returns:
            HFGenerator instance wrapping the specified model.
        """
        return HFGenerator(
            model_name=self.model_name,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            stop_strings=stop_strings,
        )

    def get_config(self) -> dict[str, str]:
        """Get configuration of this language model.

        Returns:
            Dictionary with model type and name.
        """
        return {
            "name": "huggingface",
            "model_name": self.model_name,
        }
