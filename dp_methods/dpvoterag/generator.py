from __future__ import annotations

import math
import random
import warnings
from collections import Counter
from typing import Any

import numpy as np
import torch

from core.generator import BaseGenerator
from metrics.privacy_accounting_ld import LDGumbelAccountant


class NotFoundLDTop1(Exception):
	"""Exception raised when the Limited Domain Gumbel mechanism fails to find a valid top-1 candidate.
	
	This occurs when the max of the Gumbel-perturbed top-k candidates is less than the
	Gumbel-perturbed threshold h_perp, indicating a privacy failure that must be handled
	according to the fail_mode policy.
	"""
	pass


class DPExpenseOverflow(Exception):
	"""Exception raised when the cumulative privacy budget (epsilon or delta) is exhausted.
	
	This signals that generating further tokens would violate the differential privacy
	budget constraints (target_eps or target_delta), so generation must stop.
	"""
	pass


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
		fail_mode: str = "ld_pate",
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
			fail_mode: Strategy when LD mechanism fails: 'ld_pate' (default, retry with k_bar-1),
				'rand' (random token), 'stop' (terminate with exit_status=1), or 'raise'.
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
		
		# Validate and set fail_mode
		valid_fail_modes = {"ld_pate", "rand", "stop", "raise"}
		if fail_mode not in valid_fail_modes:
			raise ValueError(
				f"Invalid fail_mode '{fail_mode}'. Must be one of: {valid_fail_modes}"
			)
		self.fail_mode = fail_mode
		
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
		
		# Get vocab size from tokenizer. This should work for most HF tokenizers.
		# Try multiple attributes since different tokenizers expose it differently.
		self.vocab_size = None
		for attr_name in ["vocab_size", "get_vocab_size", "n_vocab"]:
			if hasattr(self.tokenizer, attr_name):
				attr = getattr(self.tokenizer, attr_name)
				if callable(attr):
					try:
						self.vocab_size = int(attr())
						break
					except (TypeError, ValueError):
						pass
				else:
					try:
						self.vocab_size = int(attr)
						break
					except (TypeError, ValueError):
						pass
		
		# If still None, raise an error rather than assuming a default
		if self.vocab_size is None:
			raise ValueError(
				f"Could not determine vocabulary size from tokenizer {type(self.tokenizer).__name__}. "
				f"Please ensure the tokenizer has a 'vocab_size' attribute or implement custom logic."
			)
		
		self.vocab_size = int(self.vocab_size)
		
		# Initialize privacy accountant for online budget tracking
		self.accountant = LDGumbelAccountant(
			target_epsilon=self.epsilon_total,
			target_delta=self.delta_total,
			epsilon_token=self.epsilon_token,
			delta_token=self.delta_token,
			m=self.m,
			k=self.k,
			k_bar=self.k_bar,
			sensitivity=2.0,  # Default for l0-neighboring
		)
		self.last_run_stats: dict[str, Any] = {
			"exit_status": 0,
			"tokens_generated": 0,
			"n_private_tokens": 0,
			"n_nonprivate_tokens": 0,
			"dp_spent_epsilon": 0.0,
			"dp_spent_delta": 0.0,
			"dp_summary": {},
		}

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

	def _limited_domain_or_fail(
		self,
		histogram: dict[int, int],
		eps: float,
		delta: float,
		k_bar: int | None = None,
		sens: float = 2.0,
	) -> int:
		"""Implement the Durfee-Rogers Limited Domain Gumbel mechanism (NeurIPS 2019).
		
		This mechanism achieves (eps, delta)-DP by adding Gumbel noise to the top-k_bar
		items and to a privacy-paying threshold computed from delta.
		
		Raises NotFoundLDTop1 if the max of the perturbed top-k_bar is below the
		perturbed threshold, signaling a formal privacy failure.
		
		Args:
			histogram: Frequency counts of candidate tokens from m voters.
			eps: Privacy budget epsilon for this token query (typically epsilon_token).
			delta: Privacy parameter delta for this token query (typically delta_token).
			k_bar: Max domain size; defaults to self.k_bar. Must satisfy 0 < k_bar <= len(histogram).
			sens: Sensitivity of the mechanism (default 2 for l0-neighboring).
		
		Returns:
			The (index of the) selected token ID (an element of histogram.keys()).
		
		Raises:
			NotFoundLDTop1: If the mechanism fails (v_max < v_perp).
			ValueError: If parameters are invalid.
		"""
		if k_bar is None:
			k_bar = self.k_bar
		if not histogram:
			raise NotFoundLDTop1("Empty histogram: no candidates to vote on.")
		
		if eps <= 0.0 or delta <= 0.0 or delta >= 1.0:
			raise ValueError(f"Invalid privacy parameters: eps={eps}, delta={delta}")
		
		# Sort histogram by count (descending)
		sorted_items = sorted(histogram.items(), key=lambda item: item[1], reverse=True)
		dim = len(sorted_items)  # Total number of candidates in histogram
		k_bar = min(k_bar, dim)  # Can't use more than dim candidates
		
		if k_bar < 1:
			raise ValueError(f"k_bar must be >= 1, got {k_bar}")
		
		# Extract top-k_bar items and their counts
		top_items = sorted_items[:k_bar]
		top_counts = np.array([float(count) for _, count in top_items], dtype=np.float64)
		top_token_ids = [int(token_id) for token_id, _ in top_items]
		
		# Compute the perp threshold h_perp following Durfee-Rogers
		if k_bar < dim:
			# There are items outside the top-k_bar; use the (k_bar)-th item as reference
			h_perp = float(sorted_items[k_bar][1]) if k_bar < len(sorted_items) else 0.0
			# Add privacy-paying term: 1 + log(min(sens, k_bar, dim - k_bar) / delta) / eps
			log_arg = min(sens, float(k_bar), float(dim - k_bar)) / delta
			log_term = np.log(log_arg) if log_arg > 0 else 0.0
			h_perp = h_perp + 1.0 + log_term / eps
		else:
			# k_bar == dim; threshold is just the privacy-paying term
			h_perp = 1.0 + np.log(sens / delta) / eps
		
		# Sample Gumbel noise: scale is sens / eps
		gumbel_scale = sens / eps
		
		# Add Gumbel noise to top-k_bar counts
		v_cnts = top_counts + self._np_rng.gumbel(loc=0.0, scale=gumbel_scale, size=len(top_counts))
		v_max_idx = np.argmax(v_cnts)
		v_max = float(v_cnts[v_max_idx])
		
		# Add Gumbel noise to threshold
		v_perp = h_perp + float(self._np_rng.gumbel(loc=0.0, scale=gumbel_scale))
		
		# Compare: mechanism succeeds iff v_max >= v_perp
		if v_max >= v_perp:
			return int(top_token_ids[v_max_idx])
		else:
			raise NotFoundLDTop1(
				f"v_max={v_max:.6f} < v_perp={v_perp:.6f}; mechanism fails (formal privacy event)."
			)

	def _limited_domain(self, histogram: dict[int, int], eps: float, delta: float) -> int:
		"""Backward-compatible wrapper for legacy callers.

		This keeps the previous private API shape used by older tests/callers while
		routing through the formal Limited Domain implementation and configured
		fail-mode handling.
		"""
		try:
			return self._limited_domain_or_fail(
				histogram=histogram,
				eps=eps,
				delta=delta,
				k_bar=self.k_bar,
				sens=2.0,
			)
		except NotFoundLDTop1:
			token_id, exit_status = self._handle_ld_failure(
				histogram=histogram,
				eps=eps,
				delta=delta,
				k_bar=self.k_bar,
				sens=2.0,
			)
			if exit_status == 1 or token_id is None:
				return int(self.eos_token_id)
			return int(token_id)

	def _handle_ld_failure(
		self,
		histogram: dict[int, int],
		eps: float,
		delta: float,
		k_bar: int,
		sens: float = 2.0,
	) -> tuple[int | None, int]:
		"""Handle Limited Domain Gumbel mechanism failure according to fail_mode strategy.
		
		When the LD-Gumbel mechanism fails (v_max < v_perp), this method applies one of
		four strategies:
		
		- 'ld_pate': Retry with k_bar reduced by 1 (PATE-style excluding 1 candidate).
				If all retries exhausted, falls back to 'rand'.
		- 'rand': Return a random token from the vocabulary.
		- 'stop': Return None, signaling termination with exit_status=1.
		- 'raise': Re-raise NotFoundLDTop1.
		
		Args:
			histogram: Frequency counts from m voters.
			eps: Privacy epsilon for this token.
			delta: Privacy delta for this token.
			k_bar: Current domain size limit.
			sens: Sensitivity parameter.
		
		Returns:
			Tuple (token_id or None, exit_status): token_id is int or None (for 'stop');
			exit_status is 0 for success or 1 for failure/termination.
		"""
		if self.fail_mode == "ld_pate":
			# Retry with reduced k_bar (PATE-style: exclude 1 more candidate)
			for attempt_k_bar in range(k_bar - 1, 0, -1):
				try:
					token_id = self._limited_domain_or_fail(
						histogram, eps, delta, k_bar=attempt_k_bar, sens=sens
					)
					return int(token_id), 0  # Success with reduced k_bar
				except NotFoundLDTop1:
					continue  # Try with even smaller k_bar
			
			# All retries failed; fall back to random token
			token_id = int(self._np_rng.integers(0, self.vocab_size))
			return token_id, 0
		
		elif self.fail_mode == "rand":
			# Return a random token from the vocabulary
			token_id = int(self._np_rng.integers(0, self.vocab_size))
			return token_id, 0
		
		elif self.fail_mode == "stop":
			# Signal termination: return None with exit_status=1
			return None, 1
		
		elif self.fail_mode == "raise":
			# Re-raise the exception
			raise NotFoundLDTop1(
				f"LD mechanism failed with fail_mode='raise'; histogram has "
				f"{len(histogram)} unique tokens, k_bar={k_bar}."
			)
		else:
			# Should never reach here due to __init__ validation
			raise ValueError(f"Unknown fail_mode: {self.fail_mode}")

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
		self.accountant.reset()
		self.last_run_stats = {
			"exit_status": 0,
			"tokens_generated": 0,
			"n_private_tokens": 0,
			"n_nonprivate_tokens": 0,
			"dp_spent_epsilon": 0.0,
			"dp_spent_delta": 0.0,
			"dp_summary": {},
		}

		if not docs:
			self.last_run_stats["exit_status"] = 1
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
			self.last_run_stats["exit_status"] = 1
			return ""

		if len(docs) >= needed_docs:
			working_docs = list(docs[:needed_docs])
		else:
			working_docs = list(docs)

		partitions = self._partition_docs(working_docs)

		generated_tokens: list[int] = []
		exit_status: int = 0  # 0 = normal completion, 1 = failure/stop, 2 = budget overflow

		while True:
			# Check budget before generating next token
			budget_status = self.accountant.check_budget()
			if not budget_status["ok"]:
				exit_status = 2
				break
			
			generated_so_far = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
			voter_tokens: list[int] = []

			for voter_docs in partitions:
				prompt = self._build_prompt(query=query, docs=voter_docs, generated_so_far=generated_so_far)
				encoded = self.tokenizer(prompt, return_tensors="pt")
				next_token = self._generate_next_token(encoded["input_ids"])
				voter_tokens.append(next_token)

			histogram = dict(Counter(voter_tokens))
			try:
				winning_token = self._limited_domain_or_fail(
					histogram, self.epsilon_token, self.delta_token, k_bar=self.k_bar, sens=2.0
				)
			except NotFoundLDTop1:
				# Apply fail-mode strategy
				winning_token, fail_exit_status = self._handle_ld_failure(
					histogram, self.epsilon_token, self.delta_token, k_bar=self.k_bar, sens=2.0
				)
				if fail_exit_status == 1:
					# 'stop' mode: terminate generation
					exit_status = 1
					break
				if winning_token is None:
					# Safety check (should not happen)
					break

			if int(winning_token) == self.eos_token_id:
				break
			
			# Account for this DP token
			budget_ok = self.accountant.account_token()
			if not budget_ok:
				exit_status = 2  # Budget overflow
				break
			
			generated_tokens.append(int(winning_token))

		self.last_run_stats = {
			"exit_status": int(exit_status),
			"tokens_generated": len(generated_tokens),
			"n_private_tokens": len(generated_tokens),
			"n_nonprivate_tokens": 0,
			"dp_spent_epsilon": float(self.accountant.spent_epsilon),
			"dp_spent_delta": float(self.accountant.spent_delta),
			"dp_summary": self.accountant.get_summary(),
		}

		self.m = original_m
		return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

	def generate_with_dp(self, prompts: list[str], epsilon: float, delta: float) -> str:
		_ = (prompts, epsilon, delta)
		raise NotImplementedError(
			"VoteRAGGenerator no usa generate_with_dp(); usa generate_vote(query, docs)."
		)
