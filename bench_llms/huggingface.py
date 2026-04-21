from bench_llms.base_llm import BaseLLM
from core.generator import BaseGenerator
from vanilla.generator import HFGenerator


class HuggingFaceLLM(BaseLLM):
    def __init__(self, model_name: str = "sshleifer/tiny-gpt2"):
        self.model_name = model_name

    def build_generator(self, max_new_tokens: int) -> BaseGenerator:
        return HFGenerator(model_name=self.model_name, max_new_tokens=max_new_tokens)

    def get_config(self) -> dict[str, str]:
        return {
            "name": "huggingface",
            "model_name": self.model_name,
        }
