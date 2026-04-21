from typing import Any

from bench_llms.base_llm import BaseLLM
from bench_llms.huggingface import HuggingFaceLLM


def load_llm(config: dict[str, Any]) -> BaseLLM:
    llm_name = str(config.get("name", "huggingface")).lower()

    if llm_name in {"huggingface", "hf"}:
        model_name = str(config.get("model_name", "sshleifer/tiny-gpt2"))
        return HuggingFaceLLM(model_name=model_name)

    raise ValueError(
        f"LLM '{llm_name}' no soportado. Usa 'huggingface' (o alias 'hf')."
    )


__all__ = [
    "BaseLLM",
    "HuggingFaceLLM",
    "load_llm",
]
