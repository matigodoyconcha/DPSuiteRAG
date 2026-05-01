"""Language model factory and wrapper classes.

Provides a unified interface for loading and managing different language models
used in the RAG benchmarks.
"""

from typing import Any

from bench_llms.base_llm import BaseLLM
from bench_llms.huggingface import HuggingFaceLLM


def load_llm(config: dict[str, Any]) -> BaseLLM:
    """Factory function to load a language model from configuration.

    Args:
        config: Configuration dictionary with 'name' and model-specific parameters.
               - name: Model type (e.g., "huggingface", "hf")
               - model_name: Model identifier for HuggingFace models

    Returns:
        BaseLLM instance of the requested model type.

    Raises:
        ValueError: If the specified LLM type is not supported.
    """
    llm_name = str(config.get("name", "huggingface")).lower()

    if llm_name in {"huggingface", "hf"}:
        model_name = str(config.get("model_name", "sshleifer/tiny-gpt2"))
        return HuggingFaceLLM(model_name=model_name)

    raise ValueError(
        f"LLM '{llm_name}' is not supported. Use 'huggingface' (or alias 'hf')."
    )


__all__ = [
    "BaseLLM",
    "HuggingFaceLLM",
    "load_llm",
]
