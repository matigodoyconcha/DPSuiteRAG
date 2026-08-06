"""The fields that say *how* a number was produced have to reach the analysis.

Three of them exist only so a reader can check the protocol rather than trust it:

* ``answer_tokens`` — every text metric is monotone in the answer length, so two
  methods are comparable on accuracy only if they answered at the same length.
* ``private_answer_rate`` / ``public_only`` / ``tokens_private`` — DPSparseVoteRAG's
  Sparse Vector Technique routes tokens down a budget-free public path, so a cell whose
  private mechanism provably never fires still reports a non-zero ``answer_rate`` and
  non-zero accuracy over text no mechanism touched.
* ``epsilon_retrieval_share`` — DP-RAG's realised eps_r / (eps_r + eps_g), which moves
  across the sweep although the configured ratio does not.

Dropped anywhere between the payload and the table they are worth nothing, so the whole
path is exercised here: loader, DataFrame, and the seed aggregation that groups.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.load_results import load_all
from analysis.summary_table import aggregate_seeds, build_dataframe

VANILLA_RUNNER = PROJECT_ROOT / "experiments" / "run_vanilla.py"


# ---------------------------------------------------------------------------
# The baseline carries its answer length
# ---------------------------------------------------------------------------

def _run_experiment_returns() -> list[ast.Dict]:
    """Every dict literal `run_experiment` returns, i.e. every vanilla payload."""
    tree = ast.parse(VANILLA_RUNNER.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_experiment"
    )
    return [
        node.value for node in ast.walk(fn)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    ]


def _keys(dict_node: ast.Dict) -> list[str]:
    return [k.value for k in dict_node.keys if isinstance(k, ast.Constant)]


def test_vanilla_writes_both_payloads():
    # The QA payload and the --mia-only one. If this count changes the two assertions
    # below stop covering what they claim to.
    assert len(_run_experiment_returns()) == 2


def test_every_vanilla_payload_carries_the_answer_length():
    """The control every curve is anchored against must join on the shared key.

    Vanilla runs no mechanism, so it calls no accounting block, and without this the
    baseline was the one row whose answer length lived in a nested `generation` block
    under a different key name from all seven other methods.
    """
    for payload in _run_experiment_returns():
        assert "answer_tokens" in _keys(payload), (
            "a vanilla payload has no answer_tokens; the baseline cannot be compared "
            "on any text metric without it"
        )
        value = next(
            v for k, v in zip(payload.keys, payload.values)
            if isinstance(k, ast.Constant) and k.value == "answer_tokens"
        )
        # The length the run actually generated at, not a literal that can drift from it.
        assert isinstance(value, ast.Name) and value.id == "max_new_tokens"


def test_vanilla_gains_no_privacy_fields():
    """The baseline is the ∞-epsilon reference; a privacy field on it is a false claim.

    Anything named epsilon/delta/privacy in its payload would put the non-private
    control on the privacy axis, where it does not belong and where an aggregation
    would happily group it with the DP rows.
    """
    forbidden = ("epsilon", "delta", "privacy", "dp_", "budget")
    for payload in _run_experiment_returns():
        for key in _keys(payload):
            assert not any(f in key.lower() for f in forbidden), (
                f"the non-private baseline writes a privacy-shaped field: {key!r}"
            )


def test_vanillas_fallback_answer_length_matches_the_shipped_config():
    """A fallback that disagrees with the config silently rescores the baseline.

    Every text metric is monotone in the answer length, so a run that fell back to a
    different one produced a baseline measured on a protocol the curves were not.
    """
    from experiments.run_vanilla import default_config

    shipped = yaml.safe_load(
        (PROJECT_ROOT / "experiments/configs/full/vanilla.yaml").read_text(encoding="utf-8")
    )
    assert (
        default_config()["generation"]["max_new_tokens"]
        == shipped["generation"]["max_new_tokens"]
    )


# ---------------------------------------------------------------------------
# The loader and the table
# ---------------------------------------------------------------------------

def _write(root: Path, method: str, name: str, **payload) -> None:
    out = root / method / "triviaqa" / "model"
    out.mkdir(parents=True, exist_ok=True)
    base = {
        "method": method,
        "dataset": "triviaqa",
        "llm": {"model_name": "model"},
        "epsilon_total": 10.0,
        "epsilon_target": 10.0,
        "workload_queries": 1,
        "epsilon_workload": 10.0,
        "run_seed": 1,
        "n_queries": 2,
        "metrics": {"contains_accuracy": 0.5, "answer_rate": 0.35},
        "predictions": ["a", "a"],
        "references": ["a", "a"],
        "results": [{"prediction": "a", "reference": "a"} for _ in range(2)],
    }
    base.update(payload)
    (out / name).write_text(json.dumps(base), encoding="utf-8")


@pytest.fixture
def two_methods(tmp_path: Path) -> Path:
    _write(
        tmp_path, "dpsparsevoterag", "eps10_q1.json",
        answer_tokens=30, private_answer_rate=0.0, public_only=True, tokens_private=0,
    )
    _write(
        tmp_path, "dp_rag", "eps10_q1.json",
        answer_tokens=30, epsilon_retrieval=0.5, epsilon_generation=5.0,
        epsilon_retrieval_share=0.5 / 5.5,
    )
    return tmp_path


def test_the_loader_carries_the_protocol_fields(two_methods: Path):
    records = {r.method: r for r in load_all(two_methods, recompute_vanilla=False)}
    assert set(records) == {"dpsparsevoterag", "dp_rag"}

    sparse = records["dpsparsevoterag"]
    assert sparse.answer_tokens == 30
    assert sparse.private_answer_rate == 0.0
    assert sparse.public_only is True
    assert sparse.tokens_private == 0
    # The rate that would be read instead, kept beside it and saying the opposite: the
    # cell answered 35% of its questions with no private release behind any of them.
    assert sparse.utility["answer_rate"] == 0.35

    dp_rag = records["dp_rag"]
    assert dp_rag.epsilon_retrieval_share == pytest.approx(0.5 / 5.5)


def test_a_method_with_no_public_path_reports_none_not_zero(two_methods: Path):
    """`None` is "does not apply", `0.0` is "the mechanism served nothing".

    Read as the same value, every method without a public path is reported as one whose
    privacy mechanism never fired — which is the finding the column exists to make.
    """
    dp_rag = next(r for r in load_all(two_methods, recompute_vanilla=False) if r.method == "dp_rag")
    assert dp_rag.private_answer_rate is None
    assert dp_rag.public_only is None
    assert dp_rag.tokens_private is None
    # ... and the DP-RAG-only column is likewise absent from the vote method.
    sparse = next(
        r for r in load_all(two_methods, recompute_vanilla=False)
        if r.method == "dpsparsevoterag"
    )
    assert sparse.epsilon_retrieval_share is None


def test_the_table_shows_whether_the_mechanism_served_the_cell(two_methods: Path):
    df = build_dataframe(load_all(two_methods, recompute_vanilla=False))
    for col in ("AnswerTokens", "PrivateAnswerRate", "PublicOnly", "TokensPrivate",
                "ε_retrieval_share"):
        assert col in df.columns, f"{col} never reached the table"
    row = df[df["Method"] == "dpsparsevoterag"].iloc[0]
    assert row["PrivateAnswerRate"] == 0.0
    assert row["PublicOnly"] == 1.0
    assert row["TokensPrivate"] == 0
    assert row["AnswerTokens"] == 30


def test_the_protocol_fields_survive_seed_aggregation(two_methods: Path):
    """Aggregation is where a column is silently dropped: it keeps a whitelist."""
    df = aggregate_seeds(build_dataframe(load_all(two_methods, recompute_vanilla=False)))
    for col in ("AnswerTokens", "PrivateAnswerRate", "PublicOnly", "TokensPrivate",
                "ε_retrieval_share"):
        assert col in df.columns, f"{col} was dropped by the aggregation"
    row = df[df["Method"] == "dpsparsevoterag"].iloc[0]
    assert row["PublicOnly"] == 1.0
    assert row["PrivateAnswerRate"] == 0.0


def test_two_answer_lengths_are_not_averaged_together(tmp_path: Path):
    """Answer length is part of a cell's identity, not seed noise.

    Every text metric is monotone in it, so collapsing a 30-token run and a 100-token
    run into one mean reports the length difference as a standard deviation over seeds.
    """
    _write(tmp_path, "dpksa", "a.json", answer_tokens=30, run_seed=1,
           metrics={"contains_accuracy": 0.2})
    _write(tmp_path, "dpksa", "b.json", answer_tokens=100, run_seed=2,
           metrics={"contains_accuracy": 0.8})
    df = aggregate_seeds(build_dataframe(load_all(tmp_path, recompute_vanilla=False)))
    assert len(df) == 2, f"the answer length collapsed into one row:\n{df}"
    assert set(df["N_seeds"]) == {1}
    assert sorted(df["AnswerTokens"]) == [30, 100]
