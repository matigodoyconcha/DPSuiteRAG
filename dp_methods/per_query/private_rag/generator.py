"""Pure-DP single-query generator for Private-RAG (paper Algorithm 7).

Reference: Wu, Wang, Zhang & Wang, "Private-RAG: Answering Multiple Queries with
LLMs while Keeping Your Data Private", arXiv:2511.07637 (2025), Algorithm 7.

Algorithm 7 is a **pure-DP** variant of the Koga et al. (2024, arXiv:2412.04697)
voting mechanism: the Limited-Domain Gumbel token step is replaced by the plain
**exponential mechanism over the full vocabulary** (Algorithm 6), yielding an
ε-DP guarantee (Lemma 1) with δ = 0 — no online (ε,δ) accountant needed.

Per query (``DP-RAG(x, D, LLM, ε)``):
    Set  ε_Lap = ε_Expo = ε_0 / 2 ;  c = ⌊ε / ε_0⌋              # private-release cap
    θ̂ = θ + Lap(2/ε_Lap)                                        # noisy threshold
    D_x = retrieve mk docs; split into m voters of k docs each
    for t = 1 .. T_max:
        b   = LLM(x, ∅ | y_<t)                                  # baseline (no RAG)
        v_i = LLM(x, D_x^(i) | y_<t)  for each voter i          # voter tokens
        u   = Count(votes, V) ;  s = H[b]                       # vote histogram; agreement
        if s + Lap(4/ε_Lap) ≤ θ̂:                               # voters disagree with baseline
            y_t = ExpoMech(u, V, ε_Expo) ; c -= 1 ; θ̂ resampled  # private token
        else:
            y_t = b                                             # keep baseline (free)
        stop if y_t = <EOS> or c = 0

Sensitivity (Lemma 1): add/remove of one document changes at most one voter's
token, so the agreement score has |Δs| ≤ 1 and every vocabulary count has unit
sensitivity Δu = 1. The whole procedure is an instance of AboveThreshold with at
most ``c`` discoveries, each ε_0-DP, hence ε-DP overall.

This generator plugs into ``DPVoteRAGPipeline`` (it exposes ``.m``, ``.k`` and
``generate_vote(query, docs)``), so no dedicated pipeline class is needed —
only the *retrieval* differs between MURAG and MURAG-ADA.
"""

from __future__ import annotations

import hashlib
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.generator import BaseGenerator
from core.text_gen import to_model_prompt, trim_with_stop_strings

# Utility (vote-count) sensitivity of the exponential mechanism, Δu = 1 (Lemma 1).
_DELTA_U = 1.0

# Max prompts per forward pass in the m+1-voter vote. Sub-batching bounds the peak
# GPU memory (dominated by the O(batch * seq^2) attention scores); each prompt is an
# independent causal-forward row, so the result is identical to one big batch.
_VOTER_SUB_BATCH = 16


class PrivateRAGGenerator(BaseGenerator):
    """Algorithm 7 pure-DP generator (voting + SVT + full-vocab ExpoMech).

    Args:
        model, tokenizer: a HuggingFace causal LM and its tokenizer.
        m: number of voters. k: documents per voter (retrieved = m*k).
        epsilon: total per-query budget ε (= eps_q for MURAG, eps_RAG for MURAG-ADA).
        epsilon_0: per-token budget ε_0 (ε_Lap = ε_Expo = ε_0/2; c = ⌊ε/ε_0⌋).
        vote_threshold: SVT threshold θ (default m/2 = "half the voters"; the
            agreement score s counts votes among the m voters, so θ lives on the
            0..m scale — paper τ_svt = k/2 with k = #retrieved = m when 1 doc/voter).
        t_max: maximum decoding steps.
        random_seed: base seed; per-query RNG is ``seed + hash(query)``.
        exclude_stopwords/exclude_question_words/stopwords_path: optional public
            token exclusions from the ExpoMech candidate set (privacy-neutral; OFF
            by default = paper-faithful full vocabulary).
        task_instruction, use_chat_template, stop_strings: prompt/text plumbing,
            identical to the vote generators.
    """

    #: Cut each context document at its first ``//``, as the reference implementation
    #: does. True is paper-faithful; the shared protocol turns it off so this method
    #: reads the same document text as the six that never inherited the behaviour.
    truncate_at_double_slash: bool = True

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        m: int = 50,
        k: int = 1,
        epsilon: float = 10.0,
        epsilon_0: float = 1.0,
        vote_threshold: float | None = None,
        t_max: int = 50,
        random_seed: int = 42,
        exclude_stopwords: bool = False,
        exclude_question_words: bool = False,
        stopwords_path: str | None = None,
        task_instruction: str | None = None,
        use_chat_template: bool | str = "auto",
        stop_strings: list[str] | None = None,
        verbose: bool = False,
    ):
        if epsilon <= 0 or epsilon_0 <= 0:
            raise ValueError(f"epsilon and epsilon_0 must be > 0, got {epsilon}, {epsilon_0}")
        self.model = model
        self.tokenizer = tokenizer
        self.m = int(m)
        self.k = int(k)
        self.epsilon = float(epsilon)
        self.epsilon_0 = float(epsilon_0)
        # Budget split (Alg. 7 "Set" line) and private-release cap c = ⌊ε/ε_0⌋.
        self.epsilon_lap = self.epsilon_0 / 2.0
        self.epsilon_expo = self.epsilon_0 / 2.0
        self.max_releases = int(self.epsilon // self.epsilon_0)
        # θ default = m/2 ("half the voters"): the agreement score s is a count over
        # the m voters (0..m), so the threshold lives on the voter scale, not the
        # retrieved-doc scale. Equals the paper's τ_svt = k/2 when k = #retrieved = m
        # (1 doc/voter). Using m*k/2 would exceed m for k>1 and fire every step.
        self.vote_threshold = (
            float(vote_threshold) if vote_threshold is not None else float(self.m) / 2.0
        )
        self.t_max = int(t_max)
        self.random_seed = int(random_seed)
        self.exclude_stopwords = bool(exclude_stopwords)
        self.exclude_question_words = bool(exclude_question_words)
        self.stopwords_path = stopwords_path
        self.task_instruction = task_instruction
        self.use_chat_template = use_chat_template
        self.stop_strings = stop_strings
        self.verbose = bool(verbose)
        self.verbose_label = ""

        self._py_rng = random.Random(self.random_seed)
        self._np_rng = np.random.default_rng(self.random_seed)

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        self.eos_token_id = int(eos_id) if eos_id is not None else 0
        self.vocab_size = int(self.tokenizer.vocab_size)

        self._stopword_token_ids: set[int] = (
            self._build_stopword_token_ids() if self.exclude_stopwords else set()
        )
        self.last_run_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Tokenisation / model helpers (generic; mirror the vote generator)
    # ------------------------------------------------------------------

    def _load_stopwords(self) -> list[str]:
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

    def _tokenize_to_ids(self, text: str) -> list[int]:
        raw = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        return [int(x) for x in raw]

    def _generate_next_tokens_from_ids(self, ids_list: list[list[int]]) -> list[int]:
        """Greedy next-token (argmax) for each pre-tokenised prompt.

        Prompts are forwarded in sub-batches of at most ``_VOTER_SUB_BATCH`` so the
        peak memory of the m+1-voter vote — dominated by the O(batch * seq^2)
        attention scores — stays bounded on one GPU. Each prompt is an independent
        row of a causal forward, so sub-batching is numerically identical to a
        single large batch.
        """
        if not ids_list:
            return []
        out: list[int] = []
        for start in range(0, len(ids_list), _VOTER_SUB_BATCH):
            out.extend(self._forward_last_token(ids_list[start : start + _VOTER_SUB_BATCH]))
        return out

    def _forward_last_token(self, ids_list: list[list[int]]) -> list[int]:
        """Greedy next-token (argmax) for one (sub-)batch of pre-tokenised prompts."""
        model_device = next(self.model.parameters()).device
        max_len = max(len(ids) for ids in ids_list)
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0

        padded, attention_masks = [], []
        for ids in ids_list:
            pad_len = max_len - len(ids)
            padded.append([pad_id] * pad_len + list(ids))
            attention_masks.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(padded, dtype=torch.long).to(model_device)
        attn_mask = torch.tensor(attention_masks, dtype=torch.long).to(model_device)
        with torch.no_grad():
            # logits_to_keep=1 runs the LM head on only the last position instead of
            # every position in the sequence. The vote reads just the last-token logits,
            # so the full-sequence tensor — shape [m, seq_len, vocab], several GiB for m
            # voters over a long context in fp32 — is never materialised. The batch is
            # left-padded, so the real last token sits at index -1 in every row.
            # use_cache=False: each vote step re-forwards the full (growing) sequence,
            # so a KV cache is never reused — storing it would only inflate the peak.
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    use_cache=False,
                    logits_to_keep=1,
                )
            except TypeError:
                try:
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attn_mask, use_cache=False
                    )
                except TypeError:
                    outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
            next_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        return [int(x) for x in next_ids.tolist()]

    def _partition_docs(self, docs: list[str]) -> list[list[str]]:
        docs_copy = list(docs)
        self._py_rng.shuffle(docs_copy)
        return [docs_copy[i * self.k : (i + 1) * self.k] for i in range(self.m)]

    def _context_text(self, doc: str) -> str:
        """The document as it enters the prompt.

        The reference implementation (Koga et al., eval_rag_generator.py) cuts each
        context document at its first ``//``. On the corpora it was written for that
        strips a trailing source marker; on SearchQA snippets and ChatDoctor entries,
        which contain URLs, it truncates the document mid-content. Only this method
        and MuRAG inherit the behaviour, so under the shared protocol it would feed
        two of the eight methods a shorter context than the other six read from the
        same corpus.

        Kept for the paper-faithful configuration and off for the common one.
        """
        return doc.split("//")[0] if self.truncate_at_double_slash else doc

    def _build_prompt(self, query: str, docs: list[str], generated_so_far: str) -> str:
        """Prompt template shared with the vote generators (Koga et al.)."""
        if docs:
            ctx_list = [f"{i+1}. {self._context_text(d)}" for i, d in enumerate(docs)]
            instr = self.task_instruction or "Give a simple short answer for the question based on the context"
            instruction = f"Instruction: {instr}\n"
            context = "Context: " + "; ".join(ctx_list) + ".\n"
        else:
            instr = self.task_instruction or "Give a simple short answer for the question"
            instruction = f"Instruction: {instr}\n"
            context = ""
        return instruction + context + f"Question: {query}\nAnswer:{generated_so_far}"

    def _prompt_ids(self, query: str, docs: list[str]) -> list[int]:
        return self._tokenize_to_ids(
            to_model_prompt(
                self.tokenizer,
                self._build_prompt(query, docs, ""),
                use_chat_template=self.use_chat_template,
            )
        )

    # ------------------------------------------------------------------
    # Exponential mechanism over the full vocabulary (Algorithm 6)
    # ------------------------------------------------------------------

    def _expo_mech(self, histogram: dict[int, int]) -> int:
        """Sample a token via ExpoMech(u, V, ε_Expo) with unit sensitivity Δu=1.

        Weight of token j is ``exp(ε_Expo * u_j / (2*Δu))``. Every token with a
        zero vote count shares weight 1, so instead of materialising |V| weights we
        split the categorical draw into: (i) the observed (non-zero) tokens with
        weight ``exp(ε_Expo * count / 2)``, and (ii) the ``|V| - n_observed``
        zero-count tokens each with weight 1 (sampled uniformly). This is exactly
        the full-vocabulary exponential mechanism, just computed in O(n_observed).
        """
        v = self.vocab_size
        if not histogram:
            return int(self._np_rng.integers(0, v))

        tokens = list(histogram.keys())
        counts = np.array([histogram[t] for t in tokens], dtype=np.float64)
        weights = np.exp(self.epsilon_expo * counts / (2.0 * _DELTA_U))
        sum_obs = float(weights.sum())
        n_zero = v - len(tokens)
        total = sum_obs + float(n_zero)  # zero-count tokens: weight 1 each

        if n_zero <= 0 or self._np_rng.random() < sum_obs / total:
            probs = weights / sum_obs
            j = int(self._np_rng.choice(len(tokens), p=probs))
            return int(tokens[j])
        # Uniform over the (huge) zero-count pool via rejection (observed set is tiny).
        observed = set(tokens)
        while True:
            t = int(self._np_rng.integers(0, v))
            if t not in observed:
                return t

    # ------------------------------------------------------------------
    # BaseGenerator interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        _ = prompt
        raise NotImplementedError(
            "PrivateRAGGenerator requires calling generate_vote(query, docs) directly."
        )

    def generate_with_dp(self, prompts: list[str], epsilon: float, delta: float) -> str:
        _ = (prompts, epsilon, delta)
        raise NotImplementedError(
            "PrivateRAGGenerator does not use generate_with_dp(); use generate_vote(query, docs)."
        )

    def generate_vote(self, query: str, docs: list[str]) -> str:
        """Run Algorithm 7 for one query over the (already budget-filtered) docs."""
        query_offset = int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:8], 16)
        query_seed = self.random_seed + query_offset
        self._py_rng = random.Random(query_seed)
        self._np_rng = np.random.default_rng(query_seed)

        self.last_run_stats = {
            "exit_status": 0,
            "tokens_generated": 0,
            "n_private_tokens": 0,
            "n_baseline_tokens": 0,
            "max_releases": self.max_releases,
            "epsilon": self.epsilon,
            "epsilon_0": self.epsilon_0,
        }

        # With no retrieved docs there is nothing to vote on: all m voters would get
        # an empty chunk and reduce to the baseline prompt, so every voter would vote
        # for the baseline token and the SVT branch already takes the free baseline
        # path with overwhelming probability — skip the m redundant forward passes in
        # that degenerate case. Otherwise ALWAYS build exactly m voters: Algorithm 7
        # splits D_x into a fixed number of m chunks, so m never depends on how many
        # documents were actually retrieved.
        n_voters = self.m if docs else 0

        exclude_ids: set[int] = set(self._stopword_token_ids)
        if self.exclude_question_words:
            q_ids = self.tokenizer(query, return_tensors="pt")["input_ids"][0].tolist()
            exclude_ids.update(int(t) for t in q_ids)

        # Baseline (no-RAG) prompt + m voter prompts, tokenised once and grown by y_t.
        # _partition_docs shuffles the docs and splits them into exactly m chunks of k
        # docs each; any chunk beyond the available documents is empty, so that voter's
        # prompt equals the baseline prompt (_build_prompt(query, [], "")). This keeps
        # the agreement score s on the fixed 0..m scale for every query regardless of
        # how many documents were retrieved (Lemma 1: |Δs| <= 1, θ = m/2 stays valid).
        baseline_ids = self._prompt_ids(query, [])
        partitions: list[list[str]] = self._partition_docs(docs) if n_voters else []
        voter_ids_list = [self._prompt_ids(query, vd) for vd in partitions]
        n_nonempty_voters = sum(1 for vd in partitions if vd)
        self.last_run_stats["n_voters"] = n_voters
        self.last_run_stats["n_nonempty_voters"] = n_nonempty_voters

        # SVT: initial noisy threshold θ̂ = θ + Lap(2/ε_Lap).
        theta_hat = self.vote_threshold + float(self._np_rng.laplace(0.0, 2.0 / self.epsilon_lap))
        c = self.max_releases

        generated_tokens: list[int] = []
        n_private = 0
        n_baseline = 0
        exit_status = 0  # 0=EOS, 1=t_max, 2=release budget exhausted

        if self.verbose:
            print("=" * 30)
            print(f"[Private-RAG / Alg.7] Question: {self.verbose_label}")
            print(
                f"  m={n_voters} voters ({n_nonempty_voters} non-empty), k={self.k} "
                f"docs/voter, epsilon={self.epsilon}, epsilon_0={self.epsilon_0}, "
                f"theta={self.vote_threshold}, c={c}"
            )

        for step in range(self.t_max):
            if c <= 0:
                exit_status = 2
                break

            next_tokens = self._generate_next_tokens_from_ids([baseline_ids] + voter_ids_list)
            baseline_token = next_tokens[0]
            voter_tokens = next_tokens[1:]

            # Agreement of voters with the baseline; ExpoMech histogram (public
            # exclusions applied only to the candidate set, not to the score s).
            raw_counts = Counter(voter_tokens)
            s = float(raw_counts.get(baseline_token, 0))
            histogram = {t: c_ for t, c_ in raw_counts.items() if t not in exclude_ids}

            # SVT: low agreement (voters disagree with the no-RAG baseline) => spend a
            # private release; high agreement => keep the free baseline token.
            noisy_s = s + float(self._np_rng.laplace(0.0, 4.0 / self.epsilon_lap))
            if voter_tokens and noisy_s <= theta_hat:
                y_t = self._expo_mech(histogram)
                c -= 1
                n_private += 1
                theta_hat = self.vote_threshold + float(
                    self._np_rng.laplace(0.0, 2.0 / self.epsilon_lap)
                )
                mode = "private"
            else:
                y_t = baseline_token
                n_baseline += 1
                mode = "baseline"

            if self.verbose:
                tok = self.tokenizer.decode([int(y_t)])
                print(
                    f"  step {step}: s={s:.0f} noisy_s={noisy_s:.2f} theta_hat={theta_hat:.2f} "
                    f"-> {mode} {tok!r} (c_left={c})"
                )

            if int(y_t) == self.eos_token_id:
                break

            generated_tokens.append(int(y_t))
            baseline_ids = baseline_ids + [int(y_t)]
            voter_ids_list = [ids + [int(y_t)] for ids in voter_ids_list]
        else:
            exit_status = 1  # loop completed without EOS / budget stop

        self.last_run_stats.update(
            {
                "exit_status": int(exit_status),
                "tokens_generated": len(generated_tokens),
                "n_private_tokens": int(n_private),
                "n_baseline_tokens": int(n_baseline),
                "epsilon_spent": float(min(n_private, self.max_releases) * self.epsilon_0),
            }
        )

        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        answer = trim_with_stop_strings(answer, self.stop_strings)
        if self.verbose:
            print(f"  FINAL ({len(generated_tokens)} tok, {n_private} private): {answer!r}")
            print("=" * 30)
        return answer
