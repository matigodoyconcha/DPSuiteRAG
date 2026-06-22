from __future__ import annotations

import hashlib
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from core.text_gen import to_model_prompt, trim_with_stop_strings
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
        verbose: bool = False,
        task_instruction: str | None = None,
        use_chat_template: bool | str = "auto",
        stop_strings: list[str] | None = None,
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
            verbose=verbose,
            task_instruction=task_instruction,
            use_chat_template=use_chat_template,
            stop_strings=stop_strings,
        )
        self.tau = float(tau) if tau is not None else float(self.m) / 2.0
        self.exclude_stopwords = bool(exclude_stopwords)
        self.exclude_question_words = bool(exclude_question_words)
        self.stopwords_path = stopwords_path

        self._stopword_token_ids: set[int] = self._build_stopword_token_ids()

    def _load_stopwords(self) -> list[str]:
        """Load stopwords from data/stopword_words.pt (paper source)."""
        import torch
        # Resolve relative paths against the project root (parents[3]); the file
        # lives at <root>/data/stopword_words.pt. Using parents[2] (= dp_methods/)
        # silently missed the file and emptied the exclusion set.
        project_root = Path(__file__).resolve().parents[3]
        if self.stopwords_path:
            path = Path(self.stopwords_path)
            if not path.is_absolute():
                path = project_root / path
        else:
            path = project_root / "data" / "stopword_words.pt"
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

    def _log_sparse_pre(
        self,
        step: int,
        voter_tokens: list[int],
        histogram: dict[int, int],
        y_nonrag: int,
        a_t: int,
        a_t_noisy: float,
        tau_hat: float,
    ) -> None:
        """Print voter proposals, the public token, and the noisy SVT comparison.

        Logging only — the noisy ``a_t_noisy`` and ``tau_hat`` are the values
        already drawn by the mechanism; no extra RNG is consumed here.
        """
        print(f"  --- token step {step} ---")
        props = ", ".join(repr(self._decode_token(t)) for t in voter_tokens)
        print(f"    voter proposals (m={len(voter_tokens)} argmax votes): {props}")
        top = sorted(histogram.items(), key=lambda kv: kv[1], reverse=True)[:8]
        hist_str = ", ".join(f"{repr(self._decode_token(t))}:{c}" for t, c in top)
        print(f"    vote histogram (top): {hist_str}")
        print(
            f"    public (no-context) token y_nonrag: "
            f"{repr(self._decode_token(y_nonrag))} [id={int(y_nonrag)}]"
        )
        print(
            f"    SVT: agreement a_t={a_t} -> noisy a_t_noisy={a_t_noisy:.3f}  "
            f"vs threshold tau_hat={tau_hat:.3f}"
        )

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

            # Paper mode: tokenize once per voter + one public prompt, extend with winning
            # token. C2 — wrap in the chat template for instruct models (no-op for base
            # models); only generated tokens are decoded, so the SVT/LD mechanism is unchanged.
            voter_prompt_ids: list[list[int]] = [
                self._tokenize_to_ids(
                    to_model_prompt(
                        self.tokenizer,
                        self._build_prompt(query=query, docs=voter_docs, generated_so_far=""),
                        use_chat_template=self.use_chat_template,
                    )
                )
                for voter_docs in partitions
            ]
            public_prompt_ids: list[int] = self._tokenize_to_ids(
                to_model_prompt(
                    self.tokenizer,
                    self._build_prompt(query=query, docs=[], generated_so_far=""),
                    use_chat_template=self.use_chat_template,
                )
            )

            # SVT (Lyu-Su-Li 2017): one Above-Threshold epoch costs the threshold
            # budget `sparse_eps`; below-threshold comparisons and the agreement-count
            # Laplace (4/sparse_eps) are covered by that same budget and are NOT charged
            # separately. Charge `sparse_eps` once per threshold instantiation — here for
            # the initial threshold, and again at each resample after a private token.
            if not self.accountant.account_svt_step(self.sparse_eps):
                exit_status = 2

            if self.verbose:
                print("=" * 30)
                print(f"[DPSparseVoteRAG mechanism] Question: {self.verbose_label}")
                print(
                    f"  m={self.m} voters, sparse_fraction={self.sparse_fraction}, "
                    f"sparse_eps={self.sparse_eps:.4f}, ld_eps={self.epsilon_token:.4f}, "
                    f"initial tau_hat={tau_hat:.3f}"
                )

            # DPSPARSE-04 (paper Alg.3 line 25): bound the number of private LD-Gumbel
            # releases by c and return once it hits 0 — otherwise generation can run
            # all-public up to tmax_absolute. c starts at the max private-token count the
            # composition budget supports (evaluated after the initial SVT charge).
            c = max(1, int(self.accountant.check_budget()["t_max"]))

            for step in range(max(1, self.tmax_absolute)):
                if exit_status == 2:
                    break

                batch_ids = list(voter_prompt_ids) + [list(public_prompt_ids)]
                next_tokens = self._generate_next_tokens_from_ids(batch_ids)
                voter_tokens = list(next_tokens[:-1])
                y_nonrag = int(next_tokens[-1])
                histogram = dict(Counter(voter_tokens))

                a_t = int(histogram.get(y_nonrag, 0))
                a_t_noisy = float(a_t) + self._sample_laplace(4.0 / self.sparse_eps)

                if self.verbose:
                    self._log_sparse_pre(step, voter_tokens, histogram, y_nonrag, a_t, a_t_noisy, tau_hat)

                flag: bool | None = False
                chose_private = False

                if int(y_nonrag) in exclude_ids:
                    # Token is in exclusion list: use public token, no budget cost
                    flag = None
                    y_t = y_nonrag
                    if self.verbose:
                        print("    branch: EXCLUDED token (stopword/question word) -> public token, FREE")
                elif a_t_noisy <= tau_hat:
                    # Low agreement with public token: invoke LD-Gumbel (private)
                    flag = True
                    chose_private = True
                    if self.verbose:
                        print("    branch: LOW agreement (a_t_noisy <= tau_hat) -> LD-Gumbel (PRIVATE, costs budget)")

                    if not self.accountant.check_budget()["ok"]:
                        exit_status = 2
                        if self.verbose:
                            print("    budget exhausted before LD-Gumbel -> stop (exit_status=2)")
                        break

                    try:
                        y_t = self._limited_domain_or_fail(
                            histogram, self.epsilon_token, self.delta_token,
                            k_bar=self.k_bar, sens=self.sensitivity,
                        )
                    except NotFoundLDTop1:
                        # DPSPARSE-08: a failed get_top1 still inspected the private
                        # histogram, so we charge it before halting (mirrors DPVOTE-07).
                        # Slightly more conservative than the reference, which on the fail
                        # branch increments only the delta counter (total_queries), not the
                        # epsilon counter (total_k) — sound (epsilon stays a valid upper
                        # bound), bounded to one terminal token.
                        self.accountant.account_token()
                        exit_status = 1
                        if self.verbose:
                            print("    LD-Gumbel FAILED (null/limited-domain miss) -> halt (exit_status=1)")
                        break

                    if not self.accountant.account_token():
                        exit_status = 2
                        if self.verbose:
                            print("    budget exhausted on token accounting -> stop (exit_status=2)")
                        break

                    # Resample threshold only after a private token: this opens a new
                    # SVT epoch, so charge `sparse_eps` again.
                    if int(y_t) != self.eos_token_id:
                        tau_hat = self.tau + self._sample_laplace(2.0 / self.sparse_eps)
                        if not self.accountant.account_svt_step(self.sparse_eps):
                            exit_status = 2
                            if self.verbose:
                                print("    budget exhausted on SVT resample -> stop (exit_status=2)")
                            break
                else:
                    # High agreement: accept public token for free
                    y_t = y_nonrag
                    if self.verbose:
                        print("    branch: HIGH agreement (a_t_noisy > tau_hat) -> accept public token, FREE")

                if self.verbose:
                    print(
                        f"    SELECTED: {repr(self._decode_token(y_t))} [id={int(y_t)}]"
                        f" ({'private/LD-Gumbel' if chose_private else 'public'})"
                    )

                if int(y_t) == self.eos_token_id:
                    # EOS is not appended to the output, but if it was produced by a
                    # private LD-Gumbel step its budget was already charged via
                    # account_token() above — count it so n_private_tokens reflects
                    # the budget actually spent (a free public/excluded EOS costs
                    # nothing and is not counted).
                    if chose_private:
                        self.last_run_stats["n_private_tokens"] += 1
                    if self.verbose:
                        print(f"  --- token step {step}: SELECTED EOS, stopping ---")
                    break

                generated_tokens.append(int(y_t))
                dp_flags_per_token.append(flag)
                voter_prompt_ids = [ids + [int(y_t)] for ids in voter_prompt_ids]
                public_prompt_ids = public_prompt_ids + [int(y_t)]

                if chose_private:
                    self.last_run_stats["n_private_tokens"] += 1
                    c -= 1
                    if c <= 0:
                        # paper Alg.3 line 25: c == 0 -> return. Stop instead of
                        # continuing all-public until tmax_absolute.
                        if self.verbose:
                            print(f"  --- token step {step}: c == 0 (private budget spent), stopping ---")
                        break
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

            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # C3 — optional truncation at configured stop strings (OFF by default).
            answer = trim_with_stop_strings(answer, self.stop_strings)
            if self.verbose:
                n_priv = self.last_run_stats["n_private_tokens"]
                n_free = self.last_run_stats["n_nonprivate_tokens"]
                print(
                    f"  FINAL ANSWER (decoded, {len(generated_tokens)} tokens; "
                    f"{n_priv} private / {n_free} free): {repr(answer)}"
                )
                print("=" * 30)
            return answer
        finally:
            self.tau = original_tau
