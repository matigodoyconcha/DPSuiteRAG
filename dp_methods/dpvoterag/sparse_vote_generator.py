from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from dp_methods.dpvoterag.generator import NotFoundLDTop1, VoteRAGGenerator


class SparseVoteRAGGenerator(VoteRAGGenerator):
	"""Generator implementing sparse voting-based differential privacy.

	Extends VoteRAGGenerator with sparse voting where only a threshold number
	of agreement votes triggers private output generation.

	Issue 3: sparse_use_online_budget wires the inherited LDGumbelAccountant into
	the sparse generation loop, tracking dp_spent per query.
	Issue 6: fail_mode is now forwarded to VoteRAGGenerator.__init__.
	Issue 10: stopwords_format controls whether stopwords are loaded from the
	paper's .pt file (paper_pt) or a local .txt file (local_txt).
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
		# Issue 6: expose fail_mode (paper uses 'stop')
		fail_mode: str = "ld_pate",
		# Issue 3: online budget accounting for sparse loop
		sparse_use_online_budget: bool = False,
		sparse_account_svt: bool = False,
		# Issue 10: stopwords source and tokenization strategy
		stopwords_format: str = "local_txt",
		stopwords_token_index: int | None = None,
		# Pass-through flags for VoteRAGGenerator (Issues 2, 4, 5, 7, 8, 11, 12)
		sensitivity: float = 2.0,
		gumbel_scale_uses_sens: bool = True,
		prompt_construction: str = "string_retokenize",
		add_special_tokens_initial: bool = True,
		prompt_template: str = "local",
		tmax_strategy: str = "legacy_max",
		ld_extend_to_vocab: bool = False,
		accounting_method: str = "simple_bound",
		subsampling_rate: float = 1.0,
		n_peels: int = 1,
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
			fail_mode=fail_mode,
			sensitivity=sensitivity,
			gumbel_scale_uses_sens=gumbel_scale_uses_sens,
			prompt_construction=prompt_construction,
			add_special_tokens_initial=add_special_tokens_initial,
			prompt_template=prompt_template,
			tmax_strategy=tmax_strategy,
			tmax_absolute=tmax_absolute,
			ld_extend_to_vocab=ld_extend_to_vocab,
			accounting_method=accounting_method,
			subsampling_rate=subsampling_rate,
			n_peels=n_peels,
		)
		if float(sparse_eps) <= 0.0:
			raise ValueError(f"sparse_eps must be > 0, got {sparse_eps}")
		self.sparse_eps = float(sparse_eps)
		self.tau = float(tau) if tau is not None else float(self.m) / 2.0
		self.exclude_stopwords = bool(exclude_stopwords)
		self.exclude_question_words = bool(exclude_question_words)
		self.stopwords_path = stopwords_path
		# Issue 3
		self.sparse_use_online_budget = bool(sparse_use_online_budget)
		self.sparse_account_svt = bool(sparse_account_svt)
		# Issue 10
		self.stopwords_format = str(stopwords_format)
		self.stopwords_token_index = stopwords_token_index  # int | None

		self._stopword_token_ids: set[int] = self._build_stopword_token_ids()

		self.last_run_stats: dict[str, Any] = {
			"exit_status": 0,
			"n_private_tokens": 0,
			"n_nonprivate_tokens": 0,
			"n_excluded_tokens": 0,
			"total_tokens": 0,
			"dp_flags_per_token": [],
		}

	def _encode_text_to_ids(self, text: str) -> list[int]:
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

	def _load_stopwords(self) -> list[str]:
		"""Load stopwords according to stopwords_format.

		Issue 10: 'paper_pt' loads a torch-saved list (same source as the paper);
		'local_txt' loads a newline-separated text file (prior behavior).
		"""
		path = Path(self.stopwords_path) if self.stopwords_path else None

		if self.stopwords_format == "paper_pt":
			if path is None:
				path = Path(__file__).resolve().parents[2] / "data" / "stopword_words.pt"
			if path.exists():
				import torch
				return list(torch.load(path))
			warnings.warn(
				f"stopwords_format='paper_pt' but {path} not found; falling back to local_txt.",
				RuntimeWarning,
			)

		# local_txt (or fallback from failed paper_pt)
		candidates: list[Path] = []
		if self.stopwords_path:
			# If explicit path given, try it regardless of extension
			p = Path(self.stopwords_path)
			if not p.suffix == ".pt":
				candidates.append(p)
		repo_default = Path(__file__).resolve().parents[2] / "data" / "stopword_words.txt"
		candidates.append(repo_default)

		for p in candidates:
			if p.exists() and p.is_file():
				lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]
				words = [w for w in lines if w and not w.startswith("#")]
				if words:
					return words

		return list(self._DEFAULT_STOPWORDS)

	def _build_stopword_token_ids(self) -> set[int]:
		"""Tokenize stopwords and build the exclusion ID set.

		Issue 10: when stopwords_token_index is set (e.g., 1 for paper mode),
		only that position is taken from each word's token list (skips <bos>).
		When None, all token ids from each word are included (prior behavior).
		"""
		ids: set[int] = set()
		for word in self._load_stopwords():
			# Use the full tokenizer call (with special tokens) to match paper behavior
			encoded = self.tokenizer(word)["input_ids"]
			if self.stopwords_token_index is not None:
				if len(encoded) > self.stopwords_token_index:
					ids.add(int(encoded[self.stopwords_token_index]))
			else:
				# Prior behavior: add all token ids
				for tok_id in self._encode_text_to_ids(word):
					ids.add(int(tok_id))
		return ids

	def _sample_laplace(self, scale: float) -> float:
		# Issue 9: use instance RNG instead of global np.random
		return float(self._np_rng.laplace(0.0, float(scale)))

	def generate_vote(self, query: str, docs: list[str]) -> str:
		self.last_run_stats = {
			"exit_status": 0,
			"n_private_tokens": 0,
			"n_nonprivate_tokens": 0,
			"n_excluded_tokens": 0,
			"total_tokens": 0,
			"dp_flags_per_token": [],
			"dp_spent_epsilon": 0.0,
			"dp_spent_delta": 0.0,
		}

		# Issue 3: reset online accountant if budget tracking is active
		if self.sparse_use_online_budget:
			self.accountant.reset()

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

			# Issue 4: tmax strategy controls the loop limit
			if self.tmax_strategy == "accountant_only":
				cmax = self.tmax_absolute  # hard safety net only
			elif self.tmax_strategy == "min_of_bounds":
				cmax = self._estimate_tmax_legacy(strategy="min")
			elif self.tmax_strategy == "absolute":
				cmax = self.tmax_absolute
			else:
				# legacy_max (default local behavior)
				cmax = self._estimate_tmax_legacy(strategy="max")

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
				# Match tacchan7412/DPRAG (eval_rag_generator.py:90):
				# tokenizer(question, return_tensors="pt")["input_ids"][0].numpy().tolist()
				# Default add_special_tokens=True keeps <bos> in the exclusion set.
				question_ids = self.tokenizer(query, return_tensors="pt")["input_ids"][0].tolist()
				exclude_ids.update(int(t) for t in question_ids)

			exit_status: int = 0

			# Issue 5: bifurcation between paper-faithful token-id append and
			# the legacy string re-tokenization path.
			use_token_ids_append = self.prompt_construction == "token_ids_append"
			voter_prompt_ids: list[list[int]] = []
			public_prompt_ids: list[int] = []
			if use_token_ids_append:
				voter_prompt_ids = [
					list(self.tokenizer(
						self._build_prompt(query=query, docs=voter_docs, generated_so_far=""),
						add_special_tokens=self.add_special_tokens_initial,
					)["input_ids"])
					for voter_docs in partitions
				]
				public_prompt_ids = list(self.tokenizer(
					self._build_prompt(query=query, docs=[], generated_so_far=""),
					add_special_tokens=self.add_special_tokens_initial,
				)["input_ids"])

			# B12: SVT cost is only meaningful when the online accountant is in
			# use; otherwise the accountant is not reset between queries and the
			# charge would leak across runs.
			account_svt = (
				self.sparse_use_online_budget
				and self.sparse_account_svt
				and epsilon_token_lap > 0.0
			)

			for _ in range(max(1, self.tmax_absolute)):
				if account_svt:
					ok = self.accountant.account_svt_step(epsilon_token_lap)
					if not ok:
						exit_status = 2
						break

				if use_token_ids_append:
					batch_ids = list(voter_prompt_ids) + [list(public_prompt_ids)]
					next_tokens = self._generate_next_tokens_from_ids(batch_ids)
				else:
					decoded_so_far = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
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

					# Issue 3: check online budget before private token
					if self.sparse_use_online_budget:
						budget = self.accountant.check_budget()
						if not budget["ok"]:
							exit_status = 2
							break

					# B14: capture NotFoundLDTop1 directly so a fail_mode='stop'
					# (or fail_mode='ld_pate' with a failed retry) propagates as
					# exit_status=1 instead of being swallowed into a fake EOS.
					try:
						y_t = self._limited_domain_or_fail(
							histogram, epsilon_token_rag, delta_token_rag,
							k_bar=self.k_bar, sens=self.sensitivity,
						)
					except NotFoundLDTop1:
						fail_token, fail_exit = self._handle_ld_failure(
							histogram, epsilon_token_rag, delta_token_rag,
							k_bar=self.k_bar, sens=self.sensitivity,
						)
						if fail_exit == 1 or fail_token is None:
							exit_status = 1
							break
						y_t = int(fail_token)

					# The LD draw consumed epsilon_token_rag regardless of which
					# token was selected; account before any EOS-driven break to
					# avoid silently undercounting the last token.
					if self.sparse_use_online_budget:
						ok = self.accountant.account_token()
						if not ok:
							exit_status = 2
							break

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
					exit_status = 2
					break

				generated_tokens.append(int(y_t))
				dp_flags_per_token.append(flag)
				if use_token_ids_append:
					voter_prompt_ids = [ids + [int(y_t)] for ids in voter_prompt_ids]
					public_prompt_ids = public_prompt_ids + [int(y_t)]
				if chose_private:
					self.last_run_stats["n_private_tokens"] += 1
				elif flag is None:
					self.last_run_stats["n_excluded_tokens"] += 1
					self.last_run_stats["n_nonprivate_tokens"] += 1
				else:
					self.last_run_stats["n_nonprivate_tokens"] += 1

			self.last_run_stats["exit_status"] = exit_status
			self.last_run_stats["total_tokens"] = len(generated_tokens)
			self.last_run_stats["dp_flags_per_token"] = dp_flags_per_token

			# Issue 3: persist accountant state
			if self.sparse_use_online_budget:
				self.last_run_stats["dp_spent_epsilon"] = float(self.accountant.spent_epsilon)
				self.last_run_stats["dp_spent_delta"] = float(self.accountant.spent_delta)

			return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
		finally:
			self.m = original_m
			self.tau = original_tau
