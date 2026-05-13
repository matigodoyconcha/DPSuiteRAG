"""Smoke tests that exercise the runner kwarg-resolution paths.

Instantiates generators from YAML config blocks with a dummy model/tokenizer
so any kwarg conflict surfaces immediately without running inference.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from dp_methods.dpvoterag.generator import VoteRAGGenerator
from dp_methods.dpvoterag.sparse_vote_generator import SparseVoteRAGGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _DummyModel:
    def __init__(self) -> None:
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def __call__(self, input_ids: torch.Tensor, **_):
        vocab_size = 10
        logits = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], vocab_size), dtype=torch.float32
        )
        logits[:, -1, 1] = 1.0
        return SimpleNamespace(logits=logits)


class _DummyTokenizer:
    eos_token_id = 0
    vocab_size = 10
    pad_token_id = 0

    def __call__(self, text, return_tensors: str = "pt", **_):
        batch = text if isinstance(text, list) else [text]
        ids = [[1, 2, 3]] * len(batch)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids, dtype=torch.long)}
        return {"input_ids": ids[0] if len(batch) == 1 else ids}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return " ".join(str(t) for t in token_ids)


def _load_yaml(rel: str) -> dict:
    path = PROJECT_ROOT / rel
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_voterag_runner_kwargs() -> None:
    cfg = _load_yaml("experiments/configs/dpvoterag.yaml")
    block = cfg.get("dpvoterag", {})
    VoteRAGGenerator(
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        m=int(block.get("m", 30)),
        k=int(block.get("k", 1)),
        epsilon_token=float(block.get("epsilon_token", 2.0)),
        delta_token=float(block.get("delta_token", 1e-5)),
        epsilon_total=10.0,
        delta_total=float(cfg.get("delta_total", 1e-4)),
        k_bar=block.get("k_bar"),
        tmax_absolute=int(block.get("tmax_absolute", 50)),
        random_seed=42,
    )


def test_sparsevoterag_runner_kwargs() -> None:
    cfg = _load_yaml("experiments/configs/dpsparsevoterag.yaml")
    block = cfg.get("dpsparsevoterag", {})
    SparseVoteRAGGenerator(
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        m=int(block.get("m", 30)),
        k=int(block.get("k", 1)),
        epsilon_token=float(block.get("epsilon_token", 2.0)),
        sparse_eps=float(block.get("sparse_eps", 1.0)),
        delta_token=float(block.get("delta_token", 1e-5)),
        epsilon_total=10.0,
        delta_total=float(cfg.get("delta_total", 1e-4)),
        k_bar=block.get("k_bar"),
        random_seed=42,
        tau=block.get("threshold", block.get("tau")),
        tmax_absolute=int(block.get("tmax_absolute", 200)),
        exclude_stopwords=bool(block.get("exclude_stopwords", False)),
        exclude_question_words=bool(block.get("exclude_question_words", False)),
        stopwords_path=block.get("stopwords_path"),
    )
