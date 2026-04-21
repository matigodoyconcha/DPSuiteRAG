from typing import List
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.generator import BaseGenerator

load_dotenv()

class HFGenerator(BaseGenerator):
    def __init__(self, model_name: str = "gpt2", max_new_tokens: int = 50):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Optional: load a specific model. Make sure you have enough resources.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        
        # Return only the generated text (stripped of prompt) if possible
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_output[len(prompt):].strip()

    def generate_with_dp(self, prompts: List[str], epsilon: float, delta: float) -> str:
        raise NotImplementedError("Vanilla generator does not support Differential Privacy guarantees.")
