from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from dp_methods.dpvoterag.generator import VoteRAGGenerator


class SparseVoteRAGGenerator(VoteRAGGenerator):
	"""Generator implementing sparse voting-based differential privacy.

	Extends VoteRAGGenerator with sparse voting where only a threshold number
	of agreement votes triggers private output generation. This approach reduces
	privacy budget consumption for high-confidence predictions.

	Attributes:
		tau: Agreement threshold for non-private generation.
		tmax_absolute: Maximum absolute number of tokens to generate.
		last_run_stats: Statistics from the last generation, including per-token DP flags.
	"""

	_DEFAULT_STOPWORDS: tuple[str, ...] = (
		"a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
		"how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
		"to", "was", "what", "when", "where", "which", "who", "why", "with",
	)

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
		sparse_eps: float = 1.0,
		tau: float | None = None,
		tmax_absolute: int = 200,
		exclude_stopwords: bool = False,
		exclude_question_words: bool = False,
		stopwords_path: str | None = None,
	):
		"""Initialize the sparse voting-based DP generator.

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
			sparse_eps: Privacy budget for Sparse Vector Technique threshold/query noise.
			tau: Agreement threshold (defaults to m/2).
			tmax_absolute: Absolute maximum tokens to generate.
			exclude_stopwords: Whether to bypass SVT/DP vote for stopword tokens from public model.
			exclude_question_words: Whether to bypass SVT/DP vote for tokens appearing in the query.
			stopwords_path: Optional path to newline-separated stopwords list.
		"""
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
		if float(sparse_eps) <= 0.0:
			raise ValueError(f"sparse_eps must be > 0, got {sparse_eps}")
		self.sparse_eps = float(sparse_eps)
		self.tau = float(tau) if tau is not None else float(self.m) / 2.0
		self.tmax_absolute = int(tmax_absolute)
		self.exclude_stopwords = bool(exclude_stopwords)
		self.exclude_question_words = bool(exclude_question_words)
		self.stopwords_path = stopwords_path
		self._stopword_token_ids: set[int] = self._build_stopword_token_ids(stopwords_path)

		self.last_run_stats: dict[str, Any] = {
			"n_private_tokens": 0,
			"n_nonprivate_tokens": 0,
			"n_excluded_tokens": 0,
			"total_tokens": 0,
			"dp_flags_per_token": [],
		}

	def _encode_text_to_ids(self, text: str) -> list[int]:
		"""Encode text and return flattened token ids.

		Uses add_special_tokens=False when supported by the tokenizer.
		"""
		if not text:
			return []
		try:
			encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
		except TypeError:
			encoded = self.tokenizer(text, return_tensors="pt")
		input_ids = encoded.get("input_ids", None)
		if input_ids is None:
			return []
		try:
			return [int(x) for x in input_ids[0].tolist()]
		except Exception:
			return []

	def _load_stopwords(self, stopwords_path: str | None) -> list[str]:
		"""Load stopwords from file if available, else use internal defaults."""
		candidates: list[Path] = []
		if stopwords_path:
			candidates.append(Path(stopwords_path))
		repo_default = Path(__file__).resolve().parents[2] / "data" / "stopword_words.txt"
		candidates.append(repo_default)

		for path in candidates:
			if path.exists() and path.is_file():
				lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
				words = [w for w in lines if w and not w.startswith("#")]
				if words:
					return words

		return list(self._DEFAULT_STOPWORDS)

	def _build_stopword_token_ids(self, stopwords_path: str | None) -> set[int]:
		"""Pre-tokenize stopwords once so per-token checks are O(1)."""
		stopword_ids: set[int] = set()
		for word in self._load_stopwords(stopwords_path):
			for tok_id in self._encode_text_to_ids(word):
				stopword_ids.add(int(tok_id))
		return stopword_ids

	def _sample_laplace(self, scale: float) -> float:
		return float(np.random.laplace(0.0, float(scale)))

	def generate_vote(self, query: str, docs: list[str]) -> str:
		self.last_run_stats = {
			"n_private_tokens": 0,
			"n_nonprivate_tokens": 0,
			"n_excluded_tokens": 0,
			"total_tokens": 0,
			"dp_flags_per_token": [],
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

			epsilon_token_lap = self.sparse_eps
			epsilon_token_rag = self.epsilon_token
			delta_token_rag = self.delta_token

			if epsilon_token_lap <= 0.0:
				tau_hat = self.tau
			else:
				tau_hat = self.tau + self._sample_laplace(2.0 / epsilon_token_lap)

			generated_tokens: list[int] = []
			dp_flags_per_token: list[bool | None] = []

			exclude_ids: set[int] = set()
			if self.exclude_stopwords:
				exclude_ids.update(self._stopword_token_ids)
			if self.exclude_question_words:
				exclude_ids.update(self._encode_text_to_ids(query))

			for _ in range(max(1, self.tmax_absolute)):
				decoded_so_far = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

				# Evaluate all voter prompts + one public prompt in a single batched call.
				batched_prompts = [
					self._build_prompt(query=query, docs=voter_docs, generated_so_far=decoded_so_far)
					for voter_docs in partitions
				]
				batched_prompts.append(
					self._build_prompt(query=query, docs=[], generated_so_far=decoded_so_far)
				)
				next_tokens = self._generate_next_tokens(batched_prompts)
				voter_tokens = list(next_tokens[:-1])
				y_nonrag = int(next_tokens[-1])
				histogram = dict(Counter(voter_tokens))

				a_t = int(histogram.get(y_nonrag, 0))
				if epsilon_token_lap <= 0.0:
					a_t_noisy = float(a_t)
				else:
					a_t_noisy = float(a_t) + self._sample_laplace(4.0 / epsilon_token_lap)

				flag: bool | None = False
				chose_private = False
				if int(y_nonrag) in exclude_ids:
					flag = None
					y_t = y_nonrag
				elif a_t_noisy <= tau_hat:
					flag = True
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
				dp_flags_per_token.append(flag)
				if chose_private:
					self.last_run_stats["n_private_tokens"] += 1
				elif flag is None:
					self.last_run_stats["n_excluded_tokens"] += 1
					self.last_run_stats["n_nonprivate_tokens"] += 1
				else:
					self.last_run_stats["n_nonprivate_tokens"] += 1

			self.last_run_stats["total_tokens"] = len(generated_tokens)
			self.last_run_stats["dp_flags_per_token"] = dp_flags_per_token
			return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
		finally:
			self.m = original_m
			self.tau = original_tau
