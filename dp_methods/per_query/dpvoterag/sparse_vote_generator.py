from __future__ import annotations

import hashlib
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from dp_methods.per_query.dpvoterag.generator import NotFoundLDTop1, VoteRAGGenerator


class SparseVoteRAGGenerator(VoteRAGGenerator):
    """Voting-based DP generator with Sparse Vector Technique (Koga et al., arXiv:2412.04697).

    Extends VoteRAGGenerator: at each step, a public (no-context) next-token is proposed.
    If its noisy agreement count exceeds the noisy threshold, it is accepted for free;
    otherwise the LD-Gumbel mechanism is invoked.

    Per-token budget split (paper §3.3):
        ε_token-Lap  = epsilon_token × sparse_fraction   → SVT Laplace noise
        ε_token-RAG  = epsilon_token × (1 − sparse_fraction) → LD-Gumbel
    Paper default is equal split (sparse_fraction=0.5):
        "(ε_token-RAG, δ_token-RAG) ← (ε_token/2, δ_token)"
        "ε_token-Lap ← ε_token/2"

    Stopwords are loaded from data/stopword_words.pt (paper convention, token index 1).
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
        sparse_fraction: float = 0.5,
        tau: float | None = None,
        tmax_absolute: int = 200,
        exclude_stopwords: bool = False,
        exclude_question_words: bool = False,
        stopwords_path: str | None = None,
        sensitivity: float = 2.0,
        n_peels: int = 1,
        fail_mode: str = "stop",
    ):
        if not (0.0 < float(sparse_fraction) < 1.0):
            raise ValueError(f"sparse_fraction must be in (0, 1), got {sparse_fraction}")
        self.sparse_fraction = float(sparse_fraction)
        # Split per-token budget: SVT gets sparse_fraction, LD-Gumbel gets the rest
        self.sparse_eps = float(epsilon_token) * self.sparse_fraction
        ld_eps = float(epsilon_token) * (1.0 - self.sparse_fraction)

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            m=m,
            k=k,
            epsilon_token=ld_eps,
            delta_token=delta_token,
            epsilon_total=epsilon_total,
            delta_total=delta_total,
            k_bar=k_bar,
            random_seed=random_seed,
            sensitivity=sensitivity,
            tmax_absolute=tmax_absolute,
            n_peels=n_peels,
            fail_mode=fail_mode,
        )
        self.tau = float(tau) if tau is not None else float(self.m) / 2.0
        self.exclude_stopwords = bool(exclude_stopwords)
        self.exclude_question_words = bool(exclude_question_words)
        self.stopwords_path = stopwords_path

        self._stopword_token_ids: set[int] = self._build_stopword_token_ids()

    def _load_stopwords(self) -> list[str]:
        """Load stopwords from data/stopword_words.pt (paper source)."""
        import torch
        path = Path(self.stopwords_path) if self.stopwords_path else None
        if path is None:
            path = Path(__file__).resolve().parents[2] / "data" / "stopword_words.pt"
        if path.exists():
            return list(torch.load(path))
        warnings.warn(
            f"stopword_words.pt not found at {path}; stopword exclusion will be empty.",
            RuntimeWarning,
        )
        return []

    def _build_stopword_token_ids(self) -> set[int]:
        """Tokenize stopwords using paper convention: token at index 1 (skipping <bos>)."""
        ids: set[int] = set()
        for word in self._load_stopwords():
            encoded = self.tokenizer(word)["input_ids"]
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            if encoded and isinstance(encoded[0], list):
                encoded = encoded[0]
            if len(encoded) > 1:
                ids.add(int(encoded[1]))
        return ids

    def _sample_laplace(self, scale: float) -> float:
        return float(self._np_rng.laplace(0.0, float(scale)))

    def generate_vote(self, query: str, docs: list[str]) -> str:
        query_offset = int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:8], 16)
        query_seed = self.random_seed + query_offset
        self._py_rng = random.Random(query_seed)
        self._np_rng = np.random.default_rng(query_seed)

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
        self.accountant.reset()

        if not docs:
            self.last_run_stats["exit_status"] = 1
            return ""

        required_docs = self.m * self.k
        if len(docs) < required_docs:
            raise ValueError(
                f"DPSparseVoteRAG requires len(docs) >= m*k = {required_docs}, got {len(docs)}. "
                "Increase k_retrieved in the runner config or reduce m/k."
            )

        original_tau = self.tau

        try:
            partitions = self._partition_docs(list(docs[:required_docs]))

            tau_hat = self.tau + self._sample_laplace(2.0 / self.sparse_eps)

            generated_tokens: list[int] = []
            dp_flags_per_token: list[bool | None] = []

            exclude_ids: set[int] = set()
            if self.exclude_stopwords:
                exclude_ids.update(self._stopword_token_ids)
            if self.exclude_question_words:
                # Paper: tokenizer(question)["input_ids"][0] includes <bos>
                q_ids = self.tokenizer(query, return_tensors="pt")["input_ids"][0].tolist()
                exclude_ids.update(int(t) for t in q_ids)

            exit_status: int = 0

            # Paper mode: tokenize once per voter + one public prompt, extend with winning token
            voter_prompt_ids: list[list[int]] = [
                self._tokenize_to_ids(self._build_prompt(query=query, docs=voter_docs, generated_so_far=""))
                for voter_docs in partitions
            ]
            public_prompt_ids: list[int] = self._tokenize_to_ids(
                self._build_prompt(query=query, docs=[], generated_so_far="")
            )

            for _ in range(max(1, self.tmax_absolute)):
                # Charge SVT step budget (Dwork-Roth: 2*sparse_eps per comparison)
                if not self.accountant.account_svt_step(self.sparse_eps):
                    exit_status = 2
                    break

                batch_ids = list(voter_prompt_ids) + [list(public_prompt_ids)]
                next_tokens = self._generate_next_tokens_from_ids(batch_ids)
                voter_tokens = list(next_tokens[:-1])
                y_nonrag = int(next_tokens[-1])
                histogram = dict(Counter(voter_tokens))

                a_t = int(histogram.get(y_nonrag, 0))
                a_t_noisy = float(a_t) + self._sample_laplace(4.0 / self.sparse_eps)

                flag: bool | None = False
                chose_private = False

                if int(y_nonrag) in exclude_ids:
                    # Token is in exclusion list: use public token, no budget cost
                    flag = None
                    y_t = y_nonrag
                elif a_t_noisy <= tau_hat:
                    # Low agreement with public token: invoke LD-Gumbel (private)
                    flag = True
                    chose_private = True

                    if not self.accountant.check_budget()["ok"]:
                        exit_status = 2
                        break

                    try:
                        y_t = self._limited_domain_or_fail(
                            histogram, self.epsilon_token, self.delta_token,
                            k_bar=self.k_bar, sens=self.sensitivity,
                        )
                    except NotFoundLDTop1:
                        exit_status = 1
                        break

                    if not self.accountant.account_token():
                        exit_status = 2
                        break

                    # Resample threshold only after a private token (SVT convention)
                    if int(y_t) != self.eos_token_id:
                        tau_hat = self.tau + self._sample_laplace(2.0 / self.sparse_eps)
                else:
                    # High agreement: accept public token for free
                    y_t = y_nonrag

                if int(y_t) == self.eos_token_id:
                    break

                generated_tokens.append(int(y_t))
                dp_flags_per_token.append(flag)
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
            self.last_run_stats["dp_spent_epsilon"] = float(self.accountant.spent_epsilon)
            self.last_run_stats["dp_spent_delta"] = float(self.accountant.spent_delta)

            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        finally:
            self.tau = original_tau
