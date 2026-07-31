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
"""

from __future__ import annotations

import pytest

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
