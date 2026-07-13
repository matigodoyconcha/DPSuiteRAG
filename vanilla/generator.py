from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer

from core.device import cuda_oom_warning, model_dtype, resolve_device
from core.generator import BaseGenerator
from core.text_gen import chat_template_active, to_model_prompt, trim_with_stop_strings

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

    def __init__(
        self,
        model_name: str = "gpt2",
        max_new_tokens: int = 50,
        use_chat_template: bool | str = "auto",
        stop_strings: list[str] | None = None,
    ):
        """Initialize the HuggingFace generator.

        Args:
            model_name: HuggingFace model identifier.
            max_new_tokens: Maximum number of tokens to generate.
            use_chat_template: C2 — apply the model's chat template when present
                ("auto", the default), or force True/False. No-op for base models.
            stop_strings: C3 — optional list of stop strings to truncate the answer
                at; OFF by default (None).
        """
        self.device = resolve_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Half precision (bf16 on Ampere+, else fp16 — core.device.model_dtype) +
        # low_cpu_mem_usage so large checkpoints stream straight to the target device
        # instead of being fully materialized in fp32 host RAM first (an 8B model in
        # fp32 is ~32 GB and stalls/swaps before .to(device)).
        with cuda_oom_warning(model_name, self.device):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=model_dtype(), low_cpu_mem_usage=True
            ).to(self.device)
        self.max_new_tokens = max_new_tokens
        self.use_chat_template = use_chat_template
        self.stop_strings = stop_strings

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt text.

        Returns:
            Generated text (without the prompt prefix).
        """
        templated = chat_template_active(self.tokenizer, self.use_chat_template)
        model_prompt = to_model_prompt(self.tokenizer, prompt, use_chat_template=self.use_chat_template)
        # add_special_tokens=False when templated: the chat template already injects
        # BOS / control tokens, so the tokenizer must not add a second BOS.
        inputs = self.tokenizer(
            model_prompt, return_tensors="pt", add_special_tokens=not templated
        ).to(self.device)
        input_len = int(inputs["input_ids"].shape[1])
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)

        # Decode only the generated continuation by slicing on token ids (robust to
        # chat templating, where the decoded prompt is not a verbatim prefix).
        gen_ids = outputs[0][input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return trim_with_stop_strings(text, self.stop_strings)

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
