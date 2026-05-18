"""Unit tests for DPVoteRAG and DPSparseVoteRAG generators."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dp_methods.per_query.dpvoterag.generator import VoteRAGGenerator
from dp_methods.per_query.dpvoterag.sparse_vote_generator import SparseVoteRAGGenerator


class _DummyModel:
    def __init__(self) -> None:
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def __call__(self, input_ids: torch.Tensor, **_kwargs):
        vocab_size = 10
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], vocab_size), dtype=torch.float32)
        logits[:, -1, 1] = 1.0
        return SimpleNamespace(logits=logits)


class _DummyTokenizer:
    eos_token_id = 0
    vocab_size = 10
    pad_token_id = 0

    def __call__(self, text, return_tensors: str = "pt", **_kwargs):
        batch = text if isinstance(text, list) else [text]
        ids = [[1, 2, 3]] * len(batch)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.ones((len(batch), 3), dtype=torch.long),
            }
        # Plain list output (no return_tensors)
        return {"input_ids": ids[0] if len(batch) == 1 else ids}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return " ".join(str(t) for t in token_ids)


def _build_dummy_generator(m: int = 10, k: int = 3) -> VoteRAGGenerator:
    return VoteRAGGenerator(
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        m=m,
        k=k,
        epsilon_token=2.0,
        delta_token=1e-5,
        epsilon_total=10.0,
        delta_total=1e-4,
        k_bar=m,
        random_seed=42,
    )


def test_partition_docs() -> None:
    generator = _build_dummy_generator(m=10, k=3)
    docs = [f"doc-{i}" for i in range(30)]
    parts = generator._partition_docs(docs)

    assert len(parts) == 10
    assert all(len(p) == 3 for p in parts)

    flat = [doc for part in parts for doc in part]
    assert len(flat) == 30
    assert len(set(flat)) == 30


def test_limited_domain_returns_valid_token() -> None:
    generator = _build_dummy_generator(m=3, k=1)
    # Use large eps and delta so the mechanism succeeds reliably (paper scale = 1/eps)
    hist = {1: 10, 2: 1, 3: 1}
    out = generator._limited_domain_or_fail(histogram=hist, eps=10.0, delta=0.1)
    assert isinstance(out, int)
    assert 0 <= out < generator.vocab_size


def test_generate_vote_runs() -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    except Exception as exc:
        pytest.skip(f"Could not load gpt2: {exc}")

    generator = VoteRAGGenerator(
        model=model,
        tokenizer=tokenizer,
        m=10,
        k=1,
        epsilon_token=2.0,
        delta_token=1e-5,
        epsilon_total=4.0,
        delta_total=1e-4,
        k_bar=10,
        random_seed=42,
    )

    query = "What is differential privacy?"
    docs = [f"Differential privacy note {i}." for i in range(10)]
    text = generator.generate_vote(query=query, docs=docs)

    assert isinstance(text, str)
    stats = dict(generator.last_run_stats)
    assert int(stats.get("exit_status", 0)) in {0, 1, 2}
    if text.strip() == "":
        assert int(stats.get("exit_status", 0)) in {1, 2}


def test_sparse_inherits_vote() -> None:
    sparse = SparseVoteRAGGenerator(
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        m=10,
        k=1,
        epsilon_token=2.0,
        delta_token=1e-5,
        epsilon_total=10.0,
        delta_total=1e-4,
        k_bar=10,
        random_seed=42,
    )
    assert isinstance(sparse, VoteRAGGenerator)


def test_tau_default() -> None:
    m = 14
    sparse = SparseVoteRAGGenerator(
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        m=m,
        k=1,
        epsilon_token=2.0,
        delta_token=1e-5,
        epsilon_total=10.0,
        delta_total=1e-4,
        k_bar=m,
        random_seed=42,
        tau=None,
    )
    assert sparse.tau == m / 2


def test_sample_laplace_nonzero() -> None:
    sparse = SparseVoteRAGGenerator(
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        m=10,
        k=1,
        epsilon_token=2.0,
        delta_token=1e-5,
        epsilon_total=10.0,
        delta_total=1e-4,
        k_bar=10,
        random_seed=42,
    )
    values = [sparse._sample_laplace(1.0) for _ in range(100)]
    assert len(set(values)) > 1


def test_sparse_eps_is_independent_from_epsilon_token() -> None:
    sparse = SparseVoteRAGGenerator(
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        m=10,
        k=1,
        epsilon_token=2.0,
        sparse_eps=0.7,
        delta_token=1e-5,
        epsilon_total=10.0,
        delta_total=1e-4,
        k_bar=10,
        random_seed=42,
    )
    assert sparse.epsilon_token == 2.0
    assert sparse.sparse_eps == 0.7


def test_generate_vote_sparse_runs() -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    except Exception as exc:
        pytest.skip(f"Could not load gpt2 for sparse test: {exc}")

    generator = SparseVoteRAGGenerator(
        model=model,
        tokenizer=tokenizer,
        m=10,
        k=1,
        epsilon_token=2.0,
        delta_token=1e-5,
        epsilon_total=4.0,
        delta_total=1e-4,
        k_bar=10,
        random_seed=42,
        tmax_absolute=20,
    )

    query = "What is differential privacy?"
    docs = [f"Differential privacy note {i}." for i in range(10)]
    text = generator.generate_vote(query=query, docs=docs)

    assert isinstance(text, str)
    stats = dict(generator.last_run_stats)
    assert int(stats.get("total_tokens", 0)) >= 0



def test_last_run_stats() -> None:
    sparse = SparseVoteRAGGenerator(
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        m=10,
        k=1,
        epsilon_token=2.0,
        delta_token=1e-5,
        epsilon_total=4.0,
        delta_total=1e-4,
        k_bar=10,
        random_seed=42,
        tau=-100.0,
        tmax_absolute=6,
    )

    _ = sparse.generate_vote(query="Explain privacy", docs=[f"doc-{i}" for i in range(10)])
    stats = sparse.last_run_stats

    assert "n_private_tokens" in stats
    assert "n_nonprivate_tokens" in stats
    assert "n_excluded_tokens" in stats
    assert "exit_status" in stats
    assert "total_tokens" in stats
    assert "dp_flags_per_token" in stats
    assert stats["n_private_tokens"] + stats["n_nonprivate_tokens"] == stats["total_tokens"]
    assert len(stats["dp_flags_per_token"]) == stats["total_tokens"]


def test_sparse_exclusion_marks_none_flags() -> None:
    sparse = SparseVoteRAGGenerator(
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        m=10,
        k=1,
        epsilon_token=2.0,
        sparse_eps=1.0,
        delta_token=1e-5,
        epsilon_total=4.0,
        delta_total=1e-4,
        k_bar=10,
        random_seed=42,
        tau=-100.0,
        tmax_absolute=6,
        exclude_stopwords=False,
        exclude_question_words=True,
    )

    _ = sparse.generate_vote(query="Explain privacy", docs=[f"doc-{i}" for i in range(10)])
    stats = sparse.last_run_stats
    flags = list(stats.get("dp_flags_per_token", []))

    assert len(flags) == int(stats.get("total_tokens", 0))
    assert all(flag is None for flag in flags)
    assert int(stats.get("n_excluded_tokens", 0)) == int(stats.get("total_tokens", 0))
