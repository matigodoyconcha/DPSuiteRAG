from __future__ import annotations

import math
import random
import warnings
from collections import Counter
from typing import Any

import numpy as np
import torch

from core.generator import BaseGenerator


class VoteRAGGenerator(BaseGenerator):
	"""Generator implementing voting-based differentially private generation.

	This generator produces multiple independent responses from different
	document sets and aggregates them through voting to achieve differential
	privacy guarantees while maintaining utility.

	Attributes:
		model: Language model for generation.
		tokenizer: Tokenizer for the language model.
		m: Number of voting rounds (independent generations).
		k: Number of documents per voting round.
		epsilon_token: Privacy budget per token.
		delta_token: Privacy parameter per token.
		epsilon_total: Total privacy budget for the query.
		delta_total: Total privacy parameter for the query.
		k_bar: Maximum number of documents to use (defaults to m).
	"""

	def __init__(
		self,
		model: Any,
		tokenizer: Any,
		m: int = 30,
		k: int = 1,
		epsilon_token: float = 2.0,
		delta_token: float = 1e-5,
		epsilon_total: float = 10.0,
		delta_total: float = 1e-4,
		k_bar: int | None = None,
		random_seed: int = 42,
	):
		"""Initialize the voting-based DP generator.

		Args:
			model: Language model instance for text generation.
			tokenizer: Tokenizer compatible with the model.
			m: Number of voting rounds.
			k: Documents per round.
			epsilon_token: Privacy budget per individual token.
			delta_token: Privacy delta parameter per token.
			epsilon_total: Total privacy budget across all tokens.
			delta_total: Total privacy delta parameter.
			k_bar: Maximum documents to consider (defaults to m).
			random_seed: Random seed for reproducibility.
		"""
		self.model = model
		self.tokenizer = tokenizer
		self.m = int(m)
		self.k = int(k)
		self.epsilon_token = float(epsilon_token)
		self.delta_token = float(delta_token)
		self.epsilon_total = float(epsilon_total)
		self.delta_total = float(delta_total)
		self.k_bar = int(k_bar) if k_bar is not None else int(m)
		self.random_seed = int(random_seed)

		torch.manual_seed(self.random_seed)
		np.random.seed(self.random_seed)
		random.seed(self.random_seed)

		self._py_rng = random.Random(self.random_seed)
		self._np_rng = np.random.default_rng(self.random_seed)

		eos_id = getattr(self.tokenizer, "eos_token_id", None)
		if eos_id is None:
			eos_id = 0
		self.eos_token_id = int(eos_id)

	def _compute_tmax(self) -> int:
		if self.epsilon_token <= 0.0:
			return 1

		tmax_seq = int(max(0, math.floor(self.epsilon_total / self.epsilon_token)))

		tmax_adv = 0
		if 0.0 < self.delta_total < 1.0:
			log_term = math.log(1.0 / self.delta_total)

			def lhs_minus_eps_total(t: int) -> float:
				t_float = float(max(t, 0))
				lhs = math.sqrt(2.0 * t_float * log_term) * self.epsilon_token
				lhs += t_float * (self.epsilon_token ** 2)
				return lhs - self.epsilon_total

			if lhs_minus_eps_total(1) <= 0.0:
				lo = 1
				hi = 2
				while lhs_minus_eps_total(hi) <= 0.0 and hi < 10_000_000:
					lo = hi
					hi *= 2

				while lo + 1 < hi:
					mid = (lo + hi) // 2
					if lhs_minus_eps_total(mid) <= 0.0:
						lo = mid
					else:
						hi = mid
				tmax_adv = lo

		return max(1, tmax_seq, tmax_adv)

	def _limited_domain(self, histogram: dict[int, int], eps: float, delta: float) -> int:
		_ = delta
		if not histogram:
			return self.eos_token_id

		max_count = max(int(v) for v in histogram.values())
		if max_count <= 0:
			return self.eos_token_id

		top_items = sorted(histogram.items(), key=lambda item: item[1], reverse=True)[: self.k_bar]
		if not top_items:
			return self.eos_token_id

		if eps <= 0.0:
			return int(top_items[0][0])

		noisy_scores: list[tuple[int, float]] = []
		scale = 1.0 / eps
		for token_id, count in top_items:
			noise = float(self._np_rng.gumbel(loc=0.0, scale=scale))
			noisy_scores.append((int(token_id), float(count) + noise))

		best_token, best_score = max(noisy_scores, key=lambda item: item[1])
		if best_score <= 0.0:
			return self.eos_token_id
		return int(best_token)

	def _generate_next_token(self, input_ids: torch.Tensor) -> int:
		model_device = next(self.model.parameters()).device
		input_ids = input_ids.to(model_device)
		with torch.no_grad():
			outputs = self.model(input_ids=input_ids)
			logits = outputs.logits[:, -1, :]
			token_id = int(torch.argmax(logits, dim=-1).item())
		return token_id

	def _partition_docs(self, docs: list[str]) -> list[list[str]]:
		docs_copy = list(docs)
		self._py_rng.shuffle(docs_copy)
		return [docs_copy[i * self.k : (i + 1) * self.k] for i in range(self.m)]

	def _build_prompt(self, query: str, docs: list[str], generated_so_far: str) -> str:
		prompt_parts: list[str] = []
		if docs:
			prompt_parts.append(f"Context: {'\\n'.join(docs)}")
		prompt_parts.append(f"Question: {query}")
		prompt_parts.append(f"Answer: {generated_so_far}")
		return "\n\n".join(prompt_parts)

	def generate(self, prompt: str) -> str:
		_ = prompt
		raise NotImplementedError(
			"VoteRAGGenerator requiere llamar a generate_vote(query, docs) directamente."
		)

	def generate_vote(self, query: str, docs: list[str]) -> str:
		if not docs:
			return ""

		required_docs = self.m * self.k
		effective_m = self.m
		if len(docs) < required_docs:
			effective_m = max(1, len(docs) // max(self.k, 1))
			if effective_m <= 0:
				effective_m = 1
			warnings.warn(
				(
					"DPVoteRAG recibio menos documentos que m*k; "
					f"ajustando m de {self.m} a {effective_m}."
				),
				RuntimeWarning,
			)

		original_m = self.m
		self.m = effective_m
		needed_docs = self.m * self.k
		if needed_docs <= 0:
			self.m = original_m
			return ""

		if len(docs) >= needed_docs:
			working_docs = list(docs[:needed_docs])
		else:
			working_docs = list(docs)

		partitions = self._partition_docs(working_docs)
		tmax = self._compute_tmax()

		generated_tokens: list[int] = []

		for _ in range(tmax):
			generated_so_far = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
			voter_tokens: list[int] = []

			for voter_docs in partitions:
				prompt = self._build_prompt(query=query, docs=voter_docs, generated_so_far=generated_so_far)
				encoded = self.tokenizer(prompt, return_tensors="pt")
				next_token = self._generate_next_token(encoded["input_ids"])
				voter_tokens.append(next_token)

			histogram = dict(Counter(voter_tokens))
			winning_token = self._limited_domain(histogram, self.epsilon_token, self.delta_token)

			if int(winning_token) == self.eos_token_id:
				break
			generated_tokens.append(int(winning_token))

		self.m = original_m
		return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

	def generate_with_dp(self, prompts: list[str], epsilon: float, delta: float) -> str:
		_ = (prompts, epsilon, delta)
		raise NotImplementedError(
			"VoteRAGGenerator no usa generate_with_dp(); usa generate_vote(query, docs)."
		)
