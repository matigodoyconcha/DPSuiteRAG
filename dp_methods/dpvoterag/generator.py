from __future__ import annotations

import logging
import math
import random
import warnings
from collections import Counter
from typing import Any

import numpy as np
import torch

from core.generator import BaseGenerator
from metrics.privacy_accounting_ld import make_accountant

logger = logging.getLogger(__name__)


class NotFoundLDTop1(Exception):
	"""Exception raised when the Limited Domain Gumbel mechanism fails to find a valid top-1 candidate.

	This occurs when the max of the Gumbel-perturbed top-k candidates is less than the
	Gumbel-perturbed threshold h_perp, indicating a privacy failure that must be handled
	according to the fail_mode policy.
	"""
	pass


class DPExpenseOverflow(Exception):
	"""Exception raised when the cumulative privacy budget (epsilon or delta) is exhausted."""
	pass


class VoteRAGGenerator(BaseGenerator):
	"""Generator implementing voting-based differentially private generation.

	Supports configurable flags for paper-faithful vs local operation modes.
	Use dp_methods/dpvoterag/presets.py:resolve_dp_config to load preset configs.
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
		# Issue 2: Gumbel noise scale control
		sensitivity: float = 2.0,
		gumbel_scale_uses_sens: bool = True,
		# Issue 5 & 11: Prompt construction mode and special tokens
		prompt_construction: str = "string_retokenize",
		add_special_tokens_initial: bool = True,
		# Issue 7: Prompt template
		prompt_template: str = "local",
		# Issue 4: Tmax strategy
		tmax_strategy: str = "legacy_max",
		tmax_absolute: int = 200,
		# Issue 8: LD extend to vocab
		ld_extend_to_vocab: bool = False,
		# Issue 1: Accounting method
		accounting_method: str = "simple_bound",
		# Issue 12: Subsampling rate
		subsampling_rate: float = 1.0,
		# B13: number of EM peels per token (em_peeling_autodp only).
		# 1 matches tacchan7412/DPRAG (utils/dp.py:125 default).
		n_peels: int = 1,
	):
		self.model = model
		self.tokenizer = tokenizer
		self.m = int(m)
		self.k = int(k)
		self.epsilon_token = float(epsilon_token)
		self.delta_token = float(delta_token)
		self.epsilon_total = float(epsilon_total)
		self.delta_total = float(delta_total)
		self.k_bar = int(k_bar) if k_bar is not None else int(m)

		valid_fail_modes = {"ld_pate", "rand", "stop", "raise"}
		if fail_mode not in valid_fail_modes:
			raise ValueError(
				f"Invalid fail_mode '{fail_mode}'. Must be one of: {valid_fail_modes}"
			)
		self.fail_mode = fail_mode

		self.random_seed = int(random_seed)

		# Issue 2
		self.sensitivity = float(sensitivity)
		self.gumbel_scale_uses_sens = bool(gumbel_scale_uses_sens)
		# Issue 5 & 11
		self.prompt_construction = str(prompt_construction)
		self.add_special_tokens_initial = bool(add_special_tokens_initial)
		# Issue 7
		self.prompt_template = str(prompt_template)
		# Issue 4
		self.tmax_strategy = str(tmax_strategy)
		self.tmax_absolute = int(tmax_absolute)
		# Issue 8
		self.ld_extend_to_vocab = bool(ld_extend_to_vocab)
		# Issue 1
		self.accounting_method = str(accounting_method)
		# Issue 12
		self.subsampling_rate = float(subsampling_rate)
		# B13
		self.n_peels = int(n_peels)

		torch.manual_seed(self.random_seed)
		np.random.seed(self.random_seed)
		random.seed(self.random_seed)

		self._py_rng = random.Random(self.random_seed)
		self._np_rng = np.random.default_rng(self.random_seed)

		eos_id = getattr(self.tokenizer, "eos_token_id", None)
		if eos_id is None:
			eos_id = 0
		self.eos_token_id = int(eos_id)

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

		if self.vocab_size is None:
			raise ValueError(
				f"Could not determine vocabulary size from tokenizer {type(self.tokenizer).__name__}. "
				f"Please ensure the tokenizer has a 'vocab_size' attribute or implement custom logic."
			)

		self.vocab_size = int(self.vocab_size)

		self.accountant = make_accountant(
			self.accounting_method,
			target_epsilon=self.epsilon_total,
			target_delta=self.delta_total,
			epsilon_token=self.epsilon_token,
			delta_token=self.delta_token,
			m=self.m,
			k=self.k,
			k_bar=self.k_bar,
			sensitivity=self.sensitivity,
			subsampling_rate=self.subsampling_rate,
			n_peels=self.n_peels,
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

	def _estimate_tmax_legacy(self, strategy: str = "max") -> int:
		"""Estimate max tokens from DP budget bounds (legacy; accountant_only strategy ignores this)."""
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

		if strategy == "min":
			# Issue 4: corrección formal (min evita exceder cualquier cota individual)
			return max(1, min(tmax_seq, tmax_adv) if tmax_adv > 0 else tmax_seq)
		else:
			# legacy_max: comportamiento original
			return max(1, tmax_seq, tmax_adv)

	def _limited_domain_or_fail(
		self,
		histogram: dict[int, int],
		eps: float,
		delta: float,
		k_bar: int | None = None,
		sens: float | None = None,
	) -> int:
		"""Implement the Durfee-Rogers Limited Domain Gumbel mechanism (NeurIPS 2019).

		Issue 2: gumbel_scale controlled by self.gumbel_scale_uses_sens.
		Issue 8: virtual candidates (ld_extend_to_vocab) extend selection to full vocab.
		"""
		if k_bar is None:
			k_bar = self.k_bar
		if sens is None:
			sens = self.sensitivity
		if not histogram:
			raise NotFoundLDTop1("Empty histogram: no candidates to vote on.")

		if eps <= 0.0 or delta <= 0.0 or delta >= 1.0:
			raise ValueError(f"Invalid privacy parameters: eps={eps}, delta={delta}")

		sorted_items = sorted(histogram.items(), key=lambda item: item[1], reverse=True)
		dim = len(sorted_items)

		# Issue 2: paper uses scale=1/eps; local uses scale=sens/eps
		gumbel_scale = (sens / eps) if self.gumbel_scale_uses_sens else (1.0 / eps)

		# Issue 8: virtual candidates from full vocab when k_bar > observed dim
		n_extra = max(0, k_bar - dim)
		k_bar_real = min(k_bar, dim)

		if k_bar_real < 1:
			raise ValueError(f"k_bar_real must be >= 1, got {k_bar_real}")

		top_items = sorted_items[:k_bar_real]
		top_counts = np.array([float(count) for _, count in top_items], dtype=np.float64)
		top_token_ids = [int(token_id) for token_id, _ in top_items]

		# Compute threshold h_perp following Durfee-Rogers
		if k_bar_real < dim:
			h_perp = float(sorted_items[k_bar_real][1]) if k_bar_real < len(sorted_items) else 0.0
			log_arg = min(sens, float(k_bar_real), float(dim - k_bar_real)) / delta
			log_term = np.log(log_arg) if log_arg > 0 else 0.0
			h_perp = h_perp + 1.0 + log_term / eps
		else:
			h_perp = 1.0 + np.log(sens / delta) / eps

		# Gumbel noise for real top-k_bar candidates
		v_cnts = top_counts + self._np_rng.gumbel(loc=0.0, scale=gumbel_scale, size=k_bar_real)
		v_max_real_idx = int(np.argmax(v_cnts))
		v_max_real = float(v_cnts[v_max_real_idx])

		# Gumbel noise for threshold
		v_perp = h_perp + float(self._np_rng.gumbel(loc=0.0, scale=gumbel_scale))

		# Issue 8: virtual candidates (count=0) compete as Gumbel samples
		winner_is_virtual = False
		if n_extra > 0 and self.ld_extend_to_vocab:
			v_virtual = self._np_rng.gumbel(0.0, gumbel_scale, size=n_extra)
			v_max_virtual = float(np.max(v_virtual))
			if v_max_virtual > v_max_real:
				v_max = v_max_virtual
				winner_is_virtual = True
			else:
				v_max = v_max_real
		else:
			v_max = v_max_real

		if v_max >= v_perp:
			if winner_is_virtual:
				observed = set(top_token_ids)
				candidates = [t for t in range(self.vocab_size) if t not in observed]
				if candidates:
					return int(self._np_rng.choice(candidates))
				return int(self._np_rng.integers(0, self.vocab_size))
			return int(top_token_ids[v_max_real_idx])
		else:
			raise NotFoundLDTop1(
				f"v_max={v_max:.6f} < v_perp={v_perp:.6f}; mechanism fails (formal privacy event)."
			)

	def _limited_domain(self, histogram: dict[int, int], eps: float, delta: float) -> int:
		"""Backward-compatible wrapper that applies fail_mode on NotFoundLDTop1."""
		try:
			return self._limited_domain_or_fail(
				histogram=histogram,
				eps=eps,
				delta=delta,
				k_bar=self.k_bar,
				sens=self.sensitivity,
			)
		except NotFoundLDTop1:
			token_id, exit_status = self._handle_ld_failure(
				histogram=histogram,
				eps=eps,
				delta=delta,
				k_bar=self.k_bar,
				sens=self.sensitivity,
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
		sens: float | None = None,
	) -> tuple[int | None, int]:
		"""Handle Limited Domain Gumbel mechanism failure according to fail_mode strategy."""
		if sens is None:
			sens = self.sensitivity

		if self.fail_mode == "ld_pate":
			# Replicate tacchan7412/DPRAG (utils/ensemble.py:124-132): a single
			# retry at k_bar-1; if that also fails, stop (exit_status=1).
			retry_k_bar = int(k_bar) - 1
			if retry_k_bar < 1:
				return None, 1
			try:
				token_id = self._limited_domain_or_fail(
					histogram, eps, delta, k_bar=retry_k_bar, sens=sens
				)
				return int(token_id), 0
			except NotFoundLDTop1:
				return None, 1

		elif self.fail_mode == "rand":
			return int(self._np_rng.integers(0, self.vocab_size)), 0

		elif self.fail_mode == "stop":
			return None, 1

		elif self.fail_mode == "raise":
			raise NotFoundLDTop1(
				f"LD mechanism failed with fail_mode='raise'; histogram has "
				f"{len(histogram)} unique tokens, k_bar={k_bar}."
			)
		else:
			raise ValueError(f"Unknown fail_mode: {self.fail_mode}")

	def _generate_next_token(self, input_ids: torch.Tensor) -> int:
		model_device = next(self.model.parameters()).device
		input_ids = input_ids.to(model_device)
		with torch.no_grad():
			try:
				outputs = self.model(input_ids=input_ids, use_cache=True)
			except TypeError:
				outputs = self.model(input_ids=input_ids)
			logits = outputs.logits[:, -1, :]
			token_id = int(torch.argmax(logits, dim=-1).item())
		return token_id

	def _generate_next_tokens_from_ids(self, ids_list: list[list[int]]) -> list[int]:
		"""Generate one next-token per prompt from pre-tokenized ID lists (Issue 5).

		Avoids re-tokenization entirely: each call receives the full accumulated token
		IDs per voter and left-pads to the same length.
		"""
		if not ids_list:
			return []
		model_device = next(self.model.parameters()).device

		max_len = max(len(ids) for ids in ids_list)
		pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0

		padded = []
		attention_masks = []
		for ids in ids_list:
			pad_len = max_len - len(ids)
			padded.append([pad_id] * pad_len + list(ids))
			attention_masks.append([0] * pad_len + [1] * len(ids))

		input_ids = torch.tensor(padded, dtype=torch.long).to(model_device)
		attn_mask = torch.tensor(attention_masks, dtype=torch.long).to(model_device)

		with torch.no_grad():
			try:
				outputs = self.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
			except TypeError:
				outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
			logits = outputs.logits[:, -1, :]
			next_ids = torch.argmax(logits, dim=-1)
		return [int(x) for x in next_ids.tolist()]

	def _generate_next_tokens(self, prompts: list[str]) -> list[int]:
		"""Generate one next-token prediction per prompt using HF batched inference."""
		if not prompts:
			return []
		return self._generate_next_tokens_hf_batched(prompts)

	def _generate_next_tokens_hf_batched(self, prompts: list[str]) -> list[int]:
		"""Generate tokens using HuggingFace with batching.

		Issue 11: uses add_special_tokens=self.add_special_tokens_initial.
		"""
		model_device = next(self.model.parameters()).device

		try:
			encoded = self.tokenizer(
				prompts,
				return_tensors="pt",
				padding=True,
				add_special_tokens=self.add_special_tokens_initial,
			)
			input_ids = encoded.get("input_ids", None)
			attention_mask = encoded.get("attention_mask", None)

			if input_ids is None:
				raise ValueError("Tokenizer output does not contain input_ids")
			if int(getattr(input_ids, "shape", [0])[0]) != len(prompts):
				raise ValueError("Tokenizer did not return a batch with expected size")

			input_ids = input_ids.to(model_device)
			model_kwargs: dict[str, Any] = {"input_ids": input_ids}
			if attention_mask is not None:
				model_kwargs["attention_mask"] = attention_mask.to(model_device)

			with torch.no_grad():
				try:
					model_kwargs["use_cache"] = True
					outputs = self.model(**model_kwargs)
				except TypeError:
					model_kwargs.pop("use_cache", None)
					outputs = self.model(**model_kwargs)
				logits = outputs.logits[:, -1, :]
				next_ids = torch.argmax(logits, dim=-1)
			return [int(x) for x in next_ids.tolist()]
		except Exception as e:
			logger.debug(f"HF batching failed ({e}); falling back to sequential generation")
			next_tokens: list[int] = []
			for prompt in prompts:
				encoded_single = self.tokenizer(
					prompt,
					return_tensors="pt",
					add_special_tokens=self.add_special_tokens_initial,
				)
				next_tokens.append(self._generate_next_token(encoded_single["input_ids"]))
			return next_tokens

	def _partition_docs(self, docs: list[str]) -> list[list[str]]:
		docs_copy = list(docs)
		self._py_rng.shuffle(docs_copy)
		return [docs_copy[i * self.k : (i + 1) * self.k] for i in range(self.m)]

	def _build_prompt(self, query: str, docs: list[str], generated_so_far: str) -> str:
		"""Build a generation prompt.

		Issue 7: 'paper' template replicates tacchan7412/DPRAG eval_rag_generator.py exactly.
		"""
		if self.prompt_template == "paper":
			if docs:
				ctx_list = [f"{i+1}. {d.split('//')[0]}" for i, d in enumerate(docs)]
				instruction = "Instruction: Give a simple short answer for the question based on the context\n"
				context = "Context: " + "; ".join(ctx_list) + ".\n"
			else:
				instruction = "Instruction: Give a simple short answer for the question\n"
				context = ""
			qa = f"Question: {query}\nAnswer:{generated_so_far}"
			return instruction + context + qa
		else:
			# local (original) template
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
		exit_status: int = 0

		if self.prompt_construction == "token_ids_append":
			# Issue 5: tokenize once per voter; extend with winning token each step
			prompt_token_ids_list: list[list[int]] = [
				list(
					self.tokenizer(
						self._build_prompt(query, voter_docs, ""),
						add_special_tokens=self.add_special_tokens_initial,
					)["input_ids"]
				)
				for voter_docs in partitions
			]

			while True:
				budget_status = self.accountant.check_budget()
				if not budget_status["ok"]:
					exit_status = 2
					break

				voter_tokens = self._generate_next_tokens_from_ids(prompt_token_ids_list)
				histogram = dict(Counter(voter_tokens))

				try:
					winning_token = self._limited_domain_or_fail(
						histogram, self.epsilon_token, self.delta_token, k_bar=self.k_bar,
					)
				except NotFoundLDTop1:
					winning_token, fail_exit_status = self._handle_ld_failure(
						histogram, self.epsilon_token, self.delta_token, k_bar=self.k_bar,
					)
					if fail_exit_status == 1:
						exit_status = 1
						break
					if winning_token is None:
						break

				budget_ok = self.accountant.account_token()
				if not budget_ok:
					exit_status = 2
					break

				if int(winning_token) == self.eos_token_id:
					break

				generated_tokens.append(int(winning_token))
				prompt_token_ids_list = [ids + [int(winning_token)] for ids in prompt_token_ids_list]

		else:
			# string_retokenize: original behavior
			while True:
				budget_status = self.accountant.check_budget()
				if not budget_status["ok"]:
					exit_status = 2
					break

				generated_so_far = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
				prompts = [
					self._build_prompt(query=query, docs=voter_docs, generated_so_far=generated_so_far)
					for voter_docs in partitions
				]
				voter_tokens = self._generate_next_tokens(prompts)

				histogram = dict(Counter(voter_tokens))
				try:
					winning_token = self._limited_domain_or_fail(
						histogram, self.epsilon_token, self.delta_token, k_bar=self.k_bar,
					)
				except NotFoundLDTop1:
					winning_token, fail_exit_status = self._handle_ld_failure(
						histogram, self.epsilon_token, self.delta_token, k_bar=self.k_bar,
					)
					if fail_exit_status == 1:
						exit_status = 1
						break
					if winning_token is None:
						break

				budget_ok = self.accountant.account_token()
				if not budget_ok:
					exit_status = 2
					break

				if int(winning_token) == self.eos_token_id:
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
