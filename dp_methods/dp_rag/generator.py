from __future__ import annotations

import hashlib
import json
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.generator import BaseGenerator


class DPAggregatedGenerator(BaseGenerator):
	def __init__(
		self,
		model_name: str,
		epsilon: float,
		delta: float = 1e-5,
		max_new_tokens: int = 64,
		sensitivity: float = None,
		device: str = "auto",
		seed: int = 42,
		use_batch: bool = True,
	):
		if epsilon <= 0:
			raise ValueError("epsilon must be > 0 for DP generation.")
		if max_new_tokens <= 0:
			raise ValueError("max_new_tokens must be >= 1.")

		self.model_name = model_name
		self.epsilon = float(epsilon)
		self.delta = float(delta)
		self.max_new_tokens = int(max_new_tokens)
		self.sensitivity = sensitivity
		self.device = self._resolve_device(device)
		self.seed = int(seed)
		self.use_batch = bool(use_batch)

		np.random.seed(self.seed)
		torch.manual_seed(self.seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(self.seed)
		torch.set_grad_enabled(False)

		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		if self.tokenizer.pad_token_id is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token

		self.model = self._load_model(model_name)
		self.model.eval()

	def _resolve_device(self, device: str) -> str:
		if device == "auto":
			return "cuda" if torch.cuda.is_available() else "cpu"
		return device

	def _load_model(self, model_name: str):
		try:
			kwargs: dict[str, Any] = {"torch_dtype": torch.float16}
			if self.device == "auto":
				kwargs["device_map"] = "auto"
			model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
			if not hasattr(model, "hf_device_map"):
				model = model.to(self.device)
			return model
		except Exception:
			self.device = "cpu"
			model = AutoModelForCausalLM.from_pretrained(model_name)
			model = model.to(self.device)
			return model

	def _model_device(self) -> torch.device:
		return next(self.model.parameters()).device

	def generate(self, prompt: str) -> str:
		try:
			parsed = json.loads(prompt)
			if isinstance(parsed, list):
				return self.generate_with_dp(prompt)
		except (json.JSONDecodeError, TypeError):
			pass

		inputs = self.tokenizer(prompt, return_tensors="pt").to(self._model_device())
		with torch.no_grad():
			outputs = self.model.generate(
				**inputs,
				max_new_tokens=self.max_new_tokens,
				do_sample=False,
				pad_token_id=self.tokenizer.eos_token_id,
			)

		full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
		return full_output[len(prompt) :].strip()

	def _next_logits_batch(self, prompt_ids_list: list[torch.Tensor], generated_suffix: list[int]) -> np.ndarray:
		current_ids: list[torch.Tensor] = []
		suffix = torch.tensor(generated_suffix, dtype=torch.long) if generated_suffix else None
		for base_ids in prompt_ids_list:
			if suffix is not None and suffix.numel() > 0:
				current_ids.append(torch.cat([base_ids, suffix], dim=0))
			else:
				current_ids.append(base_ids)

		padded = torch.nn.utils.rnn.pad_sequence(
			current_ids,
			batch_first=True,
			padding_value=self.tokenizer.pad_token_id,
		)
		attention_mask = (padded != self.tokenizer.pad_token_id).long()

		device = self._model_device()
		padded = padded.to(device)
		attention_mask = attention_mask.to(device)

		with torch.no_grad():
			outputs = self.model(input_ids=padded, attention_mask=attention_mask)
			logits = outputs.logits

		last_positions = attention_mask.sum(dim=1) - 1
		row_ids = torch.arange(logits.size(0), device=logits.device)
		last_logits = logits[row_ids, last_positions, :]
		probs = F.softmax(last_logits.float(), dim=-1)
		return probs.detach().cpu().numpy()

	def _next_logits_sequential(self, prompt_ids_list: list[torch.Tensor], generated_suffix: list[int]) -> np.ndarray:
		suffix = torch.tensor(generated_suffix, dtype=torch.long) if generated_suffix else None
		all_probs = []
		device = self._model_device()

		for base_ids in prompt_ids_list:
			if suffix is not None and suffix.numel() > 0:
				current_ids = torch.cat([base_ids, suffix], dim=0)
			else:
				current_ids = base_ids

			input_ids = current_ids.unsqueeze(0).to(device)
			with torch.no_grad():
				outputs = self.model(input_ids=input_ids)
				logits = outputs.logits[0, -1, :]
			probs = F.softmax(logits.float(), dim=-1)
			all_probs.append(probs.detach().cpu().numpy())

		return np.stack(all_probs, axis=0)

	def generate_with_dp(self, prompts_json: Any, epsilon: float = None, delta: float = None) -> str:
		if isinstance(prompts_json, list):
			prompts = [str(p) for p in prompts_json]
		else:
			prompts = json.loads(prompts_json)

		if not isinstance(prompts, list):
			raise ValueError("prompts_json must decode to a list of prompts.")

		prompts = [str(p) for p in prompts if str(p).strip()]
		k_docs = len(prompts)
		if k_docs == 0:
			return ""

		# Query-level deterministic offset: same query/prompts => same run, different queries => different noise streams.
		query_key = "\n||\n".join(prompts).encode("utf-8")
		query_offset = int(hashlib.sha256(query_key).hexdigest()[:8], 16)

		eps_generation = float(epsilon) if epsilon is not None else self.epsilon
		_ = float(delta) if delta is not None else self.delta

		encoded = self.tokenizer(prompts, add_special_tokens=True)
		prompt_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]]

		generated_ids: list[int] = []
		eos_token_id = self.tokenizer.eos_token_id

		for step in range(self.max_new_tokens):
			if self.use_batch and k_docs <= 8:
				probs_per_doc = self._next_logits_batch(prompt_ids_list, generated_ids)
			else:
				probs_per_doc = self._next_logits_sequential(prompt_ids_list, generated_ids)

			p_agg = np.mean(probs_per_doc, axis=0)

			sens = float(self.sensitivity) if self.sensitivity is not None else (2.0 / k_docs)
			rng = np.random.default_rng(self.seed + query_offset + step)
			noise = rng.laplace(loc=0.0, scale=sens / eps_generation, size=p_agg.shape)
			p_noisy = p_agg + noise

			p_noisy = np.clip(p_noisy, 0.0, None)
			total = float(p_noisy.sum())
			if total <= 0.0:
				p_noisy = np.ones_like(p_agg) / float(len(p_agg))
			else:
				p_noisy = p_noisy / total

			next_token_id = int(np.argmax(p_noisy))
			if eos_token_id is not None and next_token_id == eos_token_id:
				break

			generated_ids.append(next_token_id)

		return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
