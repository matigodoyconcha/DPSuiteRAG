"""Test markers, so the default run neither downloads nor loads a model.

Nine tests account for about 87% of the suite's wall time, and all of them either
download a dataset or pull model weights onto the machine running them — one loads a
2.2 GB checkpoint to assert that a function returns a string. Left unmarked they run
by default, which makes the fast feedback loop a ten-minute one and puts several
gigabytes of weights on whatever machine happens to run the tests.

Default: everything except `network`. To include them:

    pytest -m ""                 # everything
    pytest -m network            # only the downloading ones
    BENCHDP_SLOW_TESTS=1 pytest  # also the opt-in dataset comparisons

It also builds the small position-sensitive language model that the KV-cache tests
decode with; see :func:`position_sensitive_lm`.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

#: Only the tokenizer is pulled from the hub; the weights below are random.
TINY_TOKENIZER = "sshleifer/tiny-gpt2"

# Modules whose tests download data or load model weights.
_NETWORK_MODULES = {
    "test_datasets",
    "test_llms",
    "test_dp_methods",
    "test_metrics",
    "test_vanilla",
    "test_core",
    "test_embedder_factory",
    "test_seed_reproducibility",
    "test_text_gen",
}

# Individual tests that pull weights inside an otherwise cheap module.
_NETWORK_TESTS = {
    "test_private_rag.py::test_pipeline_end_to_end",
    "test_dpvoterag.py::test_generator_end_to_end",
    "test_incremental_lm.py::test_cached_matches_uncached_distilgpt2",
}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "network: downloads a dataset or loads model weights"
    )
    config.addinivalue_line("markers", "slow: takes more than a few seconds")
    if not config.option.markexpr:
        config.option.markexpr = "not network"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    marker = pytest.mark.network
    for item in items:
        module = item.nodeid.split("/")[-1].split("::")[0].removesuffix(".py")
        nodeid = item.nodeid.split("/")[-1]
        if module in _NETWORK_MODULES or any(nodeid.startswith(t) for t in _NETWORK_TESTS):
            item.add_marker(marker)


@pytest.fixture(scope="session")
def position_sensitive_lm():
    """A small randomly-initialised GPT-2 whose every input position matters.

    The incremental KV cache fails *silently*: a wrong ``position_ids`` or the
    wrong padding side yields plausible tokens and no exception, so the tests are
    the only thing standing between a subtle bug and a whole sweep of corrupted
    results. That makes the model they decode with part of the assertion.

    ``sshleifer/tiny-gpt2`` off the shelf cannot play that role: its hidden state
    is 2-dimensional and its positional table is near zero, so neither a wrong
    position offset nor a last token read out of a padding slot moves the logits
    (measured: 3e-8 between a left- and a right-padded batch). Redrawing the
    positional table at a large scale cures the first but not the second — the
    final layer norm collapses a 2-dimensional state, so a pad token and a real
    token at the same position still map to the same output.

    A 64-dimensional two-layer GPT-2 with random weights has neither problem, is
    a few million parameters, and needs no download beyond the tokenizer. Built
    once per session because that construction is the expensive part.
    """
    tokenizer = AutoTokenizer.from_pretrained(TINY_TOKENIZER)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=256,
            n_embd=64,
            n_layer=2,
            n_head=2,
        )
    )
    model.eval()
    return model, tokenizer
