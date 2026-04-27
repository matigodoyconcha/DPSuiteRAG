from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dp_methods.dpvoterag.generator import VoteRAGGenerator
from dp_methods.dpvoterag.sparse_vote_generator import SparseVoteRAGGenerator


class _DummyModel:
    def __init__(self) -> None:
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def __call__(self, input_ids: torch.Tensor):
        vocab_size = 10
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], vocab_size), dtype=torch.float32)
        logits[:, -1, 1] = 1.0
        return SimpleNamespace(logits=logits)


class _DummyTokenizer:
    eos_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt"):
        _ = (text, return_tensors)
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
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
    hist = {1: 5, 2: 3, 3: 1}
    out = generator._limited_domain(histogram=hist, eps=2.0, delta=1e-5)
    assert out in {1, 2, 3}


def test_compute_tmax_positive() -> None:
    generator = _build_dummy_generator(m=10, k=1)
    generator.epsilon_token = 2.0
    generator.delta_token = 1e-5
    generator.epsilon_total = 10.0
    generator.delta_total = 1e-4
    assert generator._compute_tmax() > 0


def test_generate_vote_runs() -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"No se pudo cargar gpt2 para test de integracion: {exc}")

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
    assert text.strip() != ""


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


def test_generate_vote_sparse_runs() -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"No se pudo cargar gpt2 para test de integracion sparse: {exc}")

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
    assert text.strip() != ""


def _token_len_from_decoded_text(text: str) -> int:
    if not text.strip():
        return 0
    return len(text.strip().split())


def test_sparse_generates_longer() -> None:
    query = "Explain privacy"
    docs = [f"doc-{i}" for i in range(10)]

    vote_counts = []
    sparse_counts = []

    for _ in range(3):
        vote = VoteRAGGenerator(
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
        )
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
            tmax_absolute=8,
        )

        vote_text = vote.generate_vote(query=query, docs=docs)
        sparse_text = sparse.generate_vote(query=query, docs=docs)
        vote_counts.append(_token_len_from_decoded_text(vote_text))
        sparse_counts.append(_token_len_from_decoded_text(sparse_text))

    assert (sum(sparse_counts) / 3.0) >= (sum(vote_counts) / 3.0)


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
    assert "total_tokens" in stats
    assert stats["n_private_tokens"] + stats["n_nonprivate_tokens"] == stats["total_tokens"]
