from __future__ import annotations

import warnings
from collections import Counter
from typing import Any

import numpy as np

from dp_methods.dpvoterag.generator import VoteRAGGenerator


class SparseVoteRAGGenerator(VoteRAGGenerator):
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
		tau: float | None = None,
		tmax_absolute: int = 200,
	):
		super().__init__(
			model=model,
			tokenizer=tokenizer,
			m=m,
			k=k,
			epsilon_token=epsilon_token,
			delta_token=delta_token,
			epsilon_total=epsilon_total,
			delta_total=delta_total,
			k_bar=k_bar,
			random_seed=random_seed,
		)
		self.tau = float(tau) if tau is not None else float(self.m) / 2.0
		self.tmax_absolute = int(tmax_absolute)
		self.last_run_stats: dict[str, int] = {
			"n_private_tokens": 0,
			"n_nonprivate_tokens": 0,
			"total_tokens": 0,
		}

	def _sample_laplace(self, scale: float) -> float:
		return float(np.random.laplace(0.0, float(scale)))

	def generate_vote(self, query: str, docs: list[str]) -> str:
		self.last_run_stats = {
			"n_private_tokens": 0,
			"n_nonprivate_tokens": 0,
			"total_tokens": 0,
		}

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
					"DPSparseVoteRAG recibio menos documentos que m*k; "
					f"ajustando m de {self.m} a {effective_m}."
				),
				RuntimeWarning,
			)

		original_m = self.m
		original_tau = self.tau
		self.m = effective_m
		self.tau = float(original_tau if original_tau is not None else (self.m / 2.0))

		try:
			needed_docs = self.m * self.k
			if needed_docs <= 0:
				return ""

			if len(docs) >= needed_docs:
				working_docs = list(docs[:needed_docs])
			else:
				working_docs = list(docs)

			partitions = self._partition_docs(working_docs)

			cmax = self._compute_tmax()
			c = cmax

			epsilon_token_lap = self.epsilon_token / 2.0
			epsilon_token_rag = self.epsilon_token / 2.0
			delta_token_rag = self.delta_token

			if epsilon_token_lap <= 0.0:
				tau_hat = self.tau
			else:
				tau_hat = self.tau + self._sample_laplace(2.0 / epsilon_token_lap)

			generated_tokens: list[int] = []

			for _ in range(max(1, self.tmax_absolute)):
				decoded_so_far = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

				# Step i: token without private docs.
				nonrag_prompt = self._build_prompt(query=query, docs=[], generated_so_far=decoded_so_far)
				nonrag_encoded = self.tokenizer(nonrag_prompt, return_tensors="pt")
				y_nonrag = self._generate_next_token(nonrag_encoded["input_ids"])

				# Step ii: private histogram from m voters.
				voter_tokens: list[int] = []
				for voter_docs in partitions:
					prompt = self._build_prompt(query=query, docs=voter_docs, generated_so_far=decoded_so_far)
					encoded = self.tokenizer(prompt, return_tensors="pt")
					token_id = self._generate_next_token(encoded["input_ids"])
					voter_tokens.append(token_id)
				histogram = dict(Counter(voter_tokens))

				a_t = int(histogram.get(y_nonrag, 0))
				if epsilon_token_lap <= 0.0:
					a_t_noisy = float(a_t)
				else:
					a_t_noisy = float(a_t) + self._sample_laplace(4.0 / epsilon_token_lap)

				chose_private = False
				if a_t_noisy <= tau_hat:
					chose_private = True
					y_t = self._limited_domain(histogram, epsilon_token_rag, delta_token_rag)
					if int(y_t) != self.eos_token_id:
						c -= 1
						if epsilon_token_lap > 0.0:
							tau_hat = self.tau + self._sample_laplace(2.0 / epsilon_token_lap)
						else:
							tau_hat = self.tau
				else:
					y_t = y_nonrag

				if int(y_t) == self.eos_token_id:
					break
				if c == 0:
					break

				generated_tokens.append(int(y_t))
				if chose_private:
					self.last_run_stats["n_private_tokens"] += 1
				else:
					self.last_run_stats["n_nonprivate_tokens"] += 1

			self.last_run_stats["total_tokens"] = len(generated_tokens)
			return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
		finally:
			self.m = original_m
			self.tau = original_tau
