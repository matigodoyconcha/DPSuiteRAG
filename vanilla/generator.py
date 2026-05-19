from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.device import resolve_device
from core.generator import BaseGenerator

class HFGenerator(BaseGenerator):
    """Text generator using HuggingFace transformers.

    Generates text using any causal language model from HuggingFace,
    with automatic device selection (CUDA, MPS, or CPU).

    Attributes:
        model: The HuggingFace language model.
        tokenizer: The corresponding tokenizer.
        device: Device to run inference on (cuda, mps, or cpu).
        max_new_tokens: Maximum tokens to generate.
    """

    def __init__(self, model_name: str = "gpt2", max_new_tokens: int = 50):
        """Initialize the HuggingFace generator.

        Args:
            model_name: HuggingFace model identifier.
            max_new_tokens: Maximum number of tokens to generate.
        """
        self.device = resolve_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt text.

        Returns:
            Generated text (without the prompt prefix).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        
        # Return only the generated text (stripped of prompt) if possible
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_output[len(prompt):].strip()

    def generate_with_dp(self, prompts: List[str], epsilon: float, delta: float) -> str:
        """Differential privacy generation not supported for vanilla generator.

        Args:
            prompts: List of prompts (unused).
            epsilon: Privacy budget (unused).
            delta: Privacy parameter (unused).

        Raises:
            NotImplementedError: This method is not available for vanilla generation.
        """
        raise NotImplementedError("Vanilla generator does not support Differential Privacy guarantees.")
