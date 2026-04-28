"""
DP-ICL Generator: Privacy-preserving generation with DP In-Context Learning.

Reference: Grislain et al., "RAG with Differential Privacy", arXiv:2412.19291v2 (2024)

This module implements the DP-ICL mechanism which combines:
1. Per-document prompts for private context
2. A public document for prior
3. Token-by-token sampling using exponential mechanism
4. Logit transformations (l_norm, centering, clipping) for bounded sensitivity
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.generator import BaseGenerator


class DPICLGenerator(BaseGenerator):
	"""
	DP-RAG Generator with DP-ICL: Implements privacy-preserving token generation.
	
	Privacy mechanism:
	- Each token sampled from exponential mechanism over candidate tokens
	- Utility combines log-likelihood from public and private documents
	- Utility: U_ICL(r) = θ · ln(L_pub(r)) + Σⱼ l_clipped_{j}(r)
	- Token sampled from: r ∝ exp(ε · U_ICL(r) / (2C))
	- Sensitivity controlled by clipping parameter C
	
	The transformation l_norm, followed by centering and clipping, ensures
	bounded sensitivity (Δ = 1) for the exponential mechanism.
	"""
	
	def __init__(
		self,
		model_name: str,
		epsilon: float,
		delta: float = 1e-5,
		max_new_tokens: int = 64,
		theta: float = 1.0,
		C: float = 1.0,
		alpha: float = 1.0,
		device: str = "auto",
		seed: int = 42,
		use_batch: bool = True,
	):
		"""
		Initialize DP-ICL generator.
		
		Args:
			model_name: HuggingFace model identifier
			epsilon: Privacy budget for token generation
			delta: Failure probability (typically 1e-5 to 1e-3)
			max_new_tokens: Maximum tokens to generate
			theta: Weight for public document prior (default 1.0)
				- θ=0: Only private documents influence generation
				- θ large: Generation closer to LLM without context
			C: Clipping parameter controlling sensitivity (default 1.0)
				- Controls the norm of clipped logits
				- Appears in the denominator: ε/(2C)
			alpha: Parameter for l_norm transformation (default 1.0)
			device: "auto", "cuda", or "cpu"
			seed: Random seed for reproducibility
			use_batch: Use batch inference when k_docs <= 8
		"""
		if epsilon <= 0:
			raise ValueError("epsilon must be > 0 for DP generation.")
		if max_new_tokens <= 0:
			raise ValueError("max_new_tokens must be >= 1.")
		if C <= 0:
			raise ValueError("C (clipping parameter) must be > 0.")
		if alpha <= 0:
			raise ValueError("alpha must be > 0.")

		self.model_name = model_name
		self.epsilon = float(epsilon)
		self.delta = float(delta)
		self.max_new_tokens = int(max_new_tokens)
		self.theta = float(theta)
		self.C = float(C)
		self.alpha = float(alpha)
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

	def _get_logits_batch(self, prompt_ids_list: list[torch.Tensor], generated_suffix: list[int]) -> np.ndarray:
		"""
		Get raw logits for next token in batch mode.
		
		Returns:
			Array of shape (k_docs, vocab_size) with logits
		"""
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
		return last_logits.float().detach().cpu().numpy()

	def _get_logits_sequential(self, prompt_ids_list: list[torch.Tensor], generated_suffix: list[int]) -> np.ndarray:
		"""
		Get raw logits for next token in sequential mode.
		
		Returns:
			Array of shape (k_docs, vocab_size) with logits
		"""
		suffix = torch.tensor(generated_suffix, dtype=torch.long) if generated_suffix else None
		all_logits = []
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
			all_logits.append(logits.float().detach().cpu().numpy())

		return np.stack(all_logits, axis=0)

	def _transform_l_norm(self, logits: np.ndarray) -> np.ndarray:
		"""
		Apply l_norm transformation to logits.
		
		Formula: l_norm(r) = (exp(α·(ln L - ln max L)) - 1) / α
		
		Args:
			logits: Raw logits vector (shape: vocab_size)
			
		Returns:
			Transformed logits
		"""
		# Numerical stability: subtract max
		max_logit = np.max(logits)
		logits_shifted = logits - max_logit
		
		# l_norm(r) = (exp(α·logits_shifted) - 1) / α
		alpha = self.alpha
		if alpha < 1e-6:
			# Use limit as α → 0: l_norm ≈ logits_shifted
			l_norm_logits = logits_shifted
		else:
			exp_term = np.exp(alpha * logits_shifted)
			l_norm_logits = (exp_term - 1.0) / alpha
		
		return l_norm_logits.astype(np.float32)

	def _center_logits(self, l_norm_logits: np.ndarray) -> np.ndarray:
		"""
		Center the l_norm logits.
		
		Formula: l_centered(r) = l_norm(r) - (max + min) / 2
		"""
		max_val = np.max(l_norm_logits)
		min_val = np.min(l_norm_logits)
		center = (max_val + min_val) / 2.0
		l_centered = l_norm_logits - center
		return l_centered.astype(np.float32)

	def _clip_logits(self, l_centered: np.ndarray) -> np.ndarray:
		"""
		Clip centered logits using L∞ norm (GLOBAL clipping, not component-wise).
		
		Formula: l_clipped(r) = l_centered(r) · min(1, C / max|l_centered|)
		"""
		# L∞ norm: max absolute value
		max_abs = np.max(np.abs(l_centered))
		
		if max_abs <= 1e-8:
			return l_centered.astype(np.float32)
		
		# Global clipping factor
		clipping_factor = min(1.0, self.C / max_abs)
		l_clipped = l_centered * clipping_factor
		
		return l_clipped.astype(np.float32)

	def _compute_utility_icl(
		self,
		logits_pub: np.ndarray,
		logits_per_doc: list[np.ndarray],
	) -> np.ndarray:
		"""
		Compute DP-ICL utility function.
		
		Formula: U_ICL(r) = θ · ln(L_pub(r)) + Σⱼ l_clipped_{j}(r)
		"""
		# Public component: θ · logits_pub (logits are already log-space)
		public_utility = self.theta * logits_pub
		
		# Private component: sum of clipped logits from each document
		private_utility = np.zeros_like(logits_pub, dtype=np.float32)
		for l_clipped_j in logits_per_doc:
			private_utility += l_clipped_j
		
		utility_icl = public_utility + private_utility
		return utility_icl.astype(np.float32)

	def _sample_token_exponential_mechanism(
		self,
		utilities: np.ndarray,
		seed: int,
	) -> int:
		"""
		Sample token using exponential mechanism.
		
		Token sampled from: r ∝ exp(ε · U_ICL(r) / (2C))
		"""
		# Numerical stability: subtract max before exponentiating
		max_utility = np.max(utilities)
		utilities_shifted = utilities - max_utility
		
		# Exponential weights with ε/(2C) scaling
		exp_weights = np.exp((self.epsilon * utilities_shifted) / (2.0 * self.C))
		
		# Normalize to probabilities
		probabilities = exp_weights / np.sum(exp_weights)
		probabilities = np.clip(probabilities, 0.0, 1.0)
		probabilities = probabilities / np.sum(probabilities)
		
		# Sample token
		rng = np.random.default_rng(seed)
		token_id = int(rng.choice(len(utilities), p=probabilities))
		
		return token_id

	def generate(self, prompt: str) -> str:
		"""
		Generate text from a prompt (non-DP fallback).
		
		If prompt is JSON list, delegates to generate_with_dp.
		Otherwise, uses greedy decoding.
		"""
		try:
			parsed = json.loads(prompt)
			if isinstance(parsed, list):
				return self.generate_with_dp(prompt)
		except (json.JSONDecodeError, TypeError):
			pass

		# Non-DP generation (greedy)
		inputs = self.tokenizer(prompt, return_tensors="pt").to(self._model_device())
		with torch.no_grad():
			outputs = self.model.generate(
				**inputs,
				max_new_tokens=self.max_new_tokens,
				do_sample=False,
				pad_token_id=self.tokenizer.eos_token_id,
			)

		full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
		return full_output[len(prompt):].strip()

	def generate_with_dp(self, prompts_json: Any) -> str:
		"""
		Generate text with DP-ICL mechanism.
		
		The prompts_json should be a JSON list of k+1 prompts:
		- First k prompts: one per retrieved private document
		- Last prompt: public document (for prior L_pub)
		
		For each token position:
		1. Get raw logits from each prompt
		2. Transform private logits: l_norm → center → clip
		3. Keep public logits raw (only weight by θ)
		4. Compute utility: U_ICL = θ·logits_pub + Σ l_clipped
		5. Sample token from exp(ε·U/(2C))
		"""
		# Parse prompts
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
		
		if k_docs == 1:
			# Only public document, no private ones
			logits_pub_only = True
			k_private = 0
		else:
			# First k_docs-1 are private, last is public
			logits_pub_only = False
			k_private = k_docs - 1

		# Query-level seed: same query/prompts => same noise, different queries => different noise
		query_key = "\n||\n".join(prompts).encode("utf-8")
		query_offset = int(hashlib.sha256(query_key).hexdigest()[:8], 16)

		# Tokenize all prompts
		encoded = self.tokenizer(prompts, add_special_tokens=True)
		prompt_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]]

		generated_ids: list[int] = []
		eos_token_id = self.tokenizer.eos_token_id

		# Token-by-token generation
		for step in range(self.max_new_tokens):
			# Get logits for all prompts
			if self.use_batch and k_docs <= 8:
				all_logits = self._get_logits_batch(prompt_ids_list, generated_ids)
			else:
				all_logits = self._get_logits_sequential(prompt_ids_list, generated_ids)

			if logits_pub_only:
				# Only public document
				logits_pub = all_logits[0]
				logits_private_list = []
			else:
				# Separate public and private
				logits_pub = all_logits[-1]  # Last is public
				logits_private_list = list(all_logits[:-1])  # Others are private

			# Transform private logits
			clipped_logits_list = []
			for logits_private in logits_private_list:
				l_norm = self._transform_l_norm(logits_private)
				l_centered = self._center_logits(l_norm)
				l_clipped = self._clip_logits(l_centered)
				clipped_logits_list.append(l_clipped)

			# Compute utility
			utility_icl = self._compute_utility_icl(logits_pub, clipped_logits_list)

			# Sample token using exponential mechanism
			seed = self.seed + query_offset + step
			next_token_id = self._sample_token_exponential_mechanism(utility_icl, seed)

			# Check for EOS
			if eos_token_id is not None and next_token_id == eos_token_id:
				break

			generated_ids.append(next_token_id)

		return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
