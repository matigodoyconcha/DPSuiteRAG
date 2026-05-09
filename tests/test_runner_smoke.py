"""Smoke tests that exercise the runner kwarg-resolution paths.

These would have caught B1: the duplicate-kwargs TypeError in
run_dpsparsevoterag.py when the YAML and the preset both supply
``tmax_absolute``/``stopwords_path``.

We don't actually run inference — we just instantiate the generators with a
dummy model/tokenizer so that any kwarg conflict surfaces immediately.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from dp_methods.dpvoterag.generator import VoteRAGGenerator
from dp_methods.dpvoterag.presets import (
    resolve_dp_config,
    resolve_dp_config_for_voterag,
)
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
        _ = return_tensors
        if isinstance(text, list):
            return {"input_ids": torch.tensor([[1, 2, 3]] * len(text), dtype=torch.long)}
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return " ".join(str(t) for t in token_ids)


def _load_yaml(rel: str) -> dict:
    path = PROJECT_ROOT / rel
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@pytest.mark.parametrize(
    "config_rel",
    [
        "experiments/configs/dpvoterag.yaml",
        "experiments/configs/dpvoterag_local.yaml",
    ],
)
def test_voterag_runner_kwargs(config_rel: str) -> None:
    cfg = _load_yaml(config_rel)
    dp_flags = resolve_dp_config_for_voterag(cfg.get("dp", {}))

    block = cfg.get("dpvoterag", {})
    kwargs = {
        "model": _DummyModel(),
        "tokenizer": _DummyTokenizer(),
        "m": int(block.get("m", 30)),
        "k": int(block.get("k", 1)),
        "epsilon_token": float(block.get("epsilon_token", 2.0)),
        "delta_token": float(block.get("delta_token", 1e-5)),
        "epsilon_total": 10.0,
        "delta_total": float(cfg.get("delta_total", 1e-4)),
        "k_bar": block.get("k_bar"),
        "random_seed": 42,
    }
    kwargs.update(dp_flags)
    VoteRAGGenerator(**kwargs)


@pytest.mark.parametrize(
    "config_rel",
    [
        "experiments/configs/dpsparsevoterag.yaml",
        "experiments/configs/dpsparsevoterag_local.yaml",
    ],
)
def test_sparsevoterag_runner_kwargs(config_rel: str) -> None:
    cfg = _load_yaml(config_rel)
    dp_flags = resolve_dp_config(cfg.get("dp", {}))

    block = cfg.get("dpsparsevoterag", {})
    kwargs = {
        "model": _DummyModel(),
        "tokenizer": _DummyTokenizer(),
        "m": int(block.get("m", 30)),
        "k": int(block.get("k", 1)),
        "epsilon_token": float(block.get("epsilon_token", 2.0)),
        "sparse_eps": float(block.get("sparse_eps", 1.0)),
        "delta_token": float(block.get("delta_token", 1e-5)),
        "epsilon_total": 10.0,
        "delta_total": float(cfg.get("delta_total", 1e-4)),
        "k_bar": block.get("k_bar"),
        "random_seed": 42,
        "tau": block.get("threshold", block.get("tau")),
        "tmax_absolute": int(block.get("tmax_absolute", 200)),
        "exclude_stopwords": bool(block.get("exclude_stopwords", False)),
        "exclude_question_words": bool(block.get("exclude_question_words", False)),
        "stopwords_path": block.get("stopwords_path"),
    }
    # Same precedence the runner uses: preset overrides YAML defaults.
    kwargs.update(dp_flags)
    SparseVoteRAGGenerator(**kwargs)
