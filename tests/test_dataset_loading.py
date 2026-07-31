"""One loader path for every method, so a filter applies wherever it is declared."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench_datasets.movielens import MovielensDataset
from experiments.common import (
    dataset_loader_kwargs,
    load_dataset_from_config,
    normalise_dataset_config,
)

RUNNERS = [
    "run_dp_rag", "run_dpvoterag", "run_dpsparsevoterag", "run_dpksa",
    "run_private_rag", "run_dp_synrag",
]
CONFIGS = sorted(
    p for d in ("full", "paper_faithful")
    for p in (PROJECT_ROOT / "experiments" / "configs" / d).glob("*.yaml")
)


# ---------------------------------------------------------------------------
# Key aliases and the corpus/question split
# ---------------------------------------------------------------------------

def test_the_two_spellings_of_the_corpus_cap_resolve_to_one():
    assert normalise_dataset_config({"max_samples": 20000}) == {"max_index_docs": 20000}


def test_normalising_leaves_other_keys_alone():
    resolved = normalise_dataset_config({"name": "movielens", "top_movies": 30})
    assert resolved == {"name": "movielens", "top_movies": 30}


def test_no_config_still_uses_the_old_spelling():
    for path in CONFIGS:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = [config.get("dataset") or {}, *(config.get("datasets") or [])]
        for entry in entries:
            assert "max_samples" not in (entry or {}), (
                f"{path} still spells the corpus cap max_samples; two names for one "
                "quantity means the same corpus gets built twice"
            )


def test_the_question_count_does_not_default_to_the_corpus_size():
    # The old fallback asked for one question per indexed document: a 20000-document
    # corpus quietly became a 20000-question evaluation.
    resolved = normalise_dataset_config({"max_index_docs": 20000})
    assert "max_test_queries" not in resolved


# ---------------------------------------------------------------------------
# Filters reach the loader whichever method is running
# ---------------------------------------------------------------------------

def test_a_filter_declared_in_the_entry_reaches_the_loader():
    ds_cfg = {"name": "movielens", "top_movies": 30, "max_test_queries": 10}
    kwargs = dataset_loader_kwargs(MovielensDataset, normalise_dataset_config(ds_cfg))
    assert kwargs["top_movies"] == 30


@pytest.mark.parametrize("runner", RUNNERS)
def test_no_runner_constructs_a_loader_itself(runner):
    """Walk the AST rather than grep for a call that may sit on a path never taken.

    A substring check passes for a runner that uses the shared path in one branch and
    instantiates a loader by hand in another, which is precisely the drift this
    consolidation removes.
    """
    import ast

    source = (PROJECT_ROOT / "experiments" / f"{runner}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    loader_names = {
        "ChatDoctorDataset", "HotpotQADataset", "MedicalDirichletPhi3Dataset",
        "MovielensDataset", "NaturalQuestionsDataset", "SearchQADataset",
        "SQuADDataset", "TriviaQADataset",
    }
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in loader_names
    ]
    assert not calls, f"{runner} builds {sorted(set(calls))} directly"

    uses_shared = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_dataset_from_config"
        for node in ast.walk(tree)
    )
    assert uses_shared, f"{runner} never calls the shared loader"


@pytest.mark.parametrize("runner", RUNNERS)
def test_no_runner_keeps_a_private_loader(runner):
    import ast

    source = (PROJECT_ROOT / "experiments" / f"{runner}.py").read_text(encoding="utf-8")
    defined = {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
    }
    assert "load_dataset" not in defined
    assert "_resolve_dataset" not in defined


def test_an_unknown_dataset_is_named_in_the_error():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        load_dataset_from_config({"name": "not_a_dataset"}, 42)


# ---------------------------------------------------------------------------
# Delta: a mechanism parameter, not just a reporting figure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["dpvoterag", "dpsparsevoterag"])
def test_the_configured_deltas_leave_the_method_able_to_answer(method):
    """The invariant worth holding is that the mechanism still emits tokens.

    Delta here is not a reporting choice: `delta_token` sets the Limited-Domain Gumbel
    bottom-out threshold, and `delta_total` bounds how many tokens the accountant will
    release. Pushing either toward the workload target disables the method — measured,
    `delta_total = delta_token = 1e-5` leaves T_max = 1, one token per answer — so what
    a test can usefully pin is the outcome, not an arithmetic relation between the two.
    """
    from metrics.privacy_accounting_autodp import AutodpLDAccountant

    path = PROJECT_ROOT / "experiments" / "configs" / "full" / f"{method}.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    block = config[method]
    accountant = AutodpLDAccountant(
        target_epsilon=float(config.get("epsilon_values", [10])[-1]),
        target_delta=float(config["delta_total"]),
        epsilon_token=float(
            block.get("epsilon_token") or config.get("epsilon_token_values", [2])[-1]
        ),
        delta_token=float(block["delta_token"]),
        m=int(block.get("m", 50)),
        k=int(block.get("k", 1)),
        compose_method=str(block.get("compose_method", "rdp")),
    )
    t_max = accountant._t_max()
    assert t_max > 1, (
        f"{method} would release {t_max} token(s) per answer at the configured deltas; "
        "the answers come back empty and every text metric reads as a wrong answer"
    )


def test_the_workload_delta_is_reported_rather_than_forced():
    """The common delta lives on the workload axis, computed from real parameters.

    Forcing each method's own delta to 1e-5 does not make the benchmark comparable —
    the number means a different thing in each config (a per-token bottom-out mass, a
    per-query calibration target, a conversion slack). What is common is the composed
    workload figure, and the accountant is what produces it.
    """
    from metrics.workload_accounting import method_epsilon_total

    result = method_epsilon_total(
        "dpvoterag", queries=100, epsilon_token=2.0, max_tokens=50, delta_token=1e-11
    )
    assert result.delta == pytest.approx(1e-5)
    assert result.delta_additive == pytest.approx(100 * 50 * 1e-11)


def test_an_additive_delta_that_overruns_the_workload_is_refused():
    # The paper's per-token delta over a hundred 50-token queries spends 5e-3. The
    # accountant must say so instead of quietly reporting a budget it cannot honour.
    from metrics.workload_accounting import method_epsilon_total

    with pytest.raises(ValueError, match="exhausts the target delta"):
        method_epsilon_total(
            "dpvoterag", queries=100, epsilon_token=2.0, max_tokens=50, delta_token=1e-6
        )
