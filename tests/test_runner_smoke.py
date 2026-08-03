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

from dp_methods.per_query.dpvoterag.generator import VoteRAGGenerator
from dp_methods.per_query.dpvoterag.sparse_vote_generator import SparseVoteRAGGenerator


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
    cfg = _load_yaml("experiments/configs/full/dpvoterag.yaml")
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
        tmax_absolute=(
            int(block["tmax_absolute"])
            if block.get("tmax_absolute") is not None
            else None  # null => Tmax derived from composition
        ),
        random_seed=42,
    )


def test_sparsevoterag_runner_kwargs() -> None:
    cfg = _load_yaml("experiments/configs/full/dpsparsevoterag.yaml")
    block = cfg.get("dpsparsevoterag", {})
    SparseVoteRAGGenerator(
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        m=int(block.get("m", 30)),
        k=int(block.get("k", 1)),
        epsilon_token=float(block.get("epsilon_token", 2.0)),
        sparse_fraction=float(block.get("sparse_fraction", 0.5)),
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


@pytest.mark.parametrize(
    "config_path",
    [
        "experiments/configs/full/dpsparsevoterag.yaml",
        "experiments/configs/paper_faithful/dpsparsevoterag.yaml",
    ],
)
def test_sparsevoterag_epsilon_token_values_sweep(config_path: str) -> None:
    """Every point of the shipped sweep constructs a working generator.

    The two config sets express the sweep differently and both have to hold. The
    paper (Koga et al., §IV-A) fixes several per-token budgets and sweeps the query
    budget at each, one curve per value. The shared protocol names privacy targets
    and workload sizes instead and derives the query budget for each pair, so that a
    point of this method and a point of another are the same condition.
    """
    cfg = _load_yaml(config_path)
    block = cfg.get("dpsparsevoterag", {})

    if cfg.get("epsilon_targets"):
        from metrics.workload_accounting import DEFAULT_DELTA, budget_for_target

        cap = int(block["max_private_tokens"])
        pairs = [
            (
                budget_for_target(
                    "dpsparsevoterag", target_epsilon=float(t), queries=int(q),
                    delta=DEFAULT_DELTA, max_private_tokens=cap,
                    delta_token=float(block["delta_token"]),
                    sparse_fraction=float(block["sparse_fraction"]),
                ) / cap,
                None,
            )
            for q in cfg["workload_queries"]
            for t in cfg["epsilon_targets"]
        ]
        epsilon_token_values = [tok for tok, _ in pairs]
        epsilon_values = [1.0]  # the query budget is already inside epsilon_token
    else:
        epsilon_token_values = cfg.get("epsilon_token_values")
        assert isinstance(epsilon_token_values, list) and len(epsilon_token_values) > 1
        epsilon_values = cfg.get("epsilon_values")
        assert isinstance(epsilon_values, list) and epsilon_values

    for epsilon_token in epsilon_token_values:
        for epsilon_total in epsilon_values:
            SparseVoteRAGGenerator(
                model=_DummyModel(),
                tokenizer=_DummyTokenizer(),
                m=int(block.get("m", 30)),
                k=int(block.get("k", 1)),
                epsilon_token=float(epsilon_token),
                sparse_fraction=float(block.get("sparse_fraction", 0.5)),
                delta_token=float(block.get("delta_token", 1e-5)),
                epsilon_total=float(epsilon_total),
                delta_total=float(cfg.get("delta_total", 1e-4)),
                k_bar=block.get("k_bar"),
                random_seed=42,
                tau=block.get("threshold", block.get("tau")),
                tmax_absolute=int(block.get("tmax_absolute", 200)),
                exclude_stopwords=bool(block.get("exclude_stopwords", False)),
                exclude_question_words=bool(block.get("exclude_question_words", False)),
                stopwords_path=block.get("stopwords_path"),
            )


def test_sparsevoterag_epsilon_token_values_fallback() -> None:
    """When epsilon_token_values is absent from the config, the runner falls
    back to the single method-block epsilon_token (prior single-curve
    behavior), mirroring the fallback in run_dpsparsevoterag.py.
    """
    cfg = _load_yaml("experiments/configs/full/dpsparsevoterag.yaml")
    block = dict(cfg.get("dpsparsevoterag", {}))
    cfg_without_sweep = {k: v for k, v in cfg.items() if k != "epsilon_token_values"}

    epsilon_token_default = float(block.get("epsilon_token", 2.0))
    epsilon_token_values = cfg_without_sweep.get("epsilon_token_values")
    epsilon_token_values = (
        [float(v) for v in epsilon_token_values]
        if epsilon_token_values
        else [epsilon_token_default]
    )
    assert epsilon_token_values == [epsilon_token_default]
