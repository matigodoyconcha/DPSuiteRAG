"""What the vote methods release privately, and what their online guard permits.

Two facts about the vote grids are invisible in the utility columns and are what these
tests hold. First, an answer being non-empty says nothing about which mechanism wrote
it: DPSparseVoteRAG routes every token the sparse test leaves below threshold down a
public, budget-free path, so a cell whose Limited-Domain vote can never fire still
returns short answers that score. `private_answer_rate` and `public_only` are the
columns that say so, and they have to mean the same thing for both methods.

Second, the generators' online (epsilon, delta) guard is a per-query ceiling. Set to
the whole workload's delta it never binds and checks nothing; set below the derived
per-release delta it refuses every release and the cells silently collapse onto
near-identical epsilons. The derivation here has to leave every shipped cell releasing
the full configured answer length.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.privacy_accounting_autodp import release_ceiling
from metrics.utility import answer_rate, private_answer_rate
from metrics.workload_accounting import (
    DEFAULT_DELTA,
    budget_for_target,
    delta_token_for_workload,
)


def _cfg(name: str) -> dict:
    path = PROJECT_ROOT / "experiments" / "configs" / "full" / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# private_answer_rate
# ---------------------------------------------------------------------------

def test_private_answer_rate_counts_answers_the_mechanism_contributed_to():
    preds = ["Paris", "Berlin", "Rome", "Madrid"]
    assert private_answer_rate(preds, [3, 0, 1, 0]) == pytest.approx(0.5)


def test_a_public_only_cell_answers_without_releasing_anything():
    """The case the column exists for: answered, scored, mechanism never consulted."""
    preds = ["Paris", "1945", "Rome"]
    assert answer_rate(preds) == 1.0
    assert private_answer_rate(preds, [0, 0, 0]) == 0.0


def test_an_empty_answer_is_not_private_however_it_is_counted():
    # A blank string cannot carry a released token; counting one would report a
    # private answer for a query the mechanism halted on.
    assert private_answer_rate(["", "  "], [2, 0]) == 0.0


def test_private_answer_rate_never_exceeds_answer_rate():
    preds = ["Paris", "", "Rome", "   "]
    counts = [4, 0, 2, 1]
    assert private_answer_rate(preds, counts) <= answer_rate(preds)


def test_private_answer_rate_requires_aligned_inputs():
    with pytest.raises(ValueError):
        private_answer_rate(["a", "b"], [1])


def test_private_answer_rate_of_no_queries_is_nan():
    import math

    assert math.isnan(private_answer_rate([], []))


# ---------------------------------------------------------------------------
# The online guard: delta_total derived per query
# ---------------------------------------------------------------------------

def _dpvoterag_cells():
    cfg = _cfg("dpvoterag")
    block = cfg["dpvoterag"]
    tokens = int(block["max_length"])
    for queries in (int(q) for q in cfg["workload_queries"]):
        delta_token = delta_token_for_workload(queries=queries, tokens=tokens)
        for target in (float(t) for t in cfg["epsilon_targets"]):
            epsilon_total = budget_for_target(
                "dpvoterag",
                target_epsilon=target,
                queries=queries,
                delta=DEFAULT_DELTA,
                max_tokens=tokens,
                delta_token=delta_token,
            )
            yield {
                "cfg": cfg, "block": block, "tokens": tokens, "queries": queries,
                "target": target, "delta_token": delta_token,
                "epsilon_total": epsilon_total,
                "epsilon_token": epsilon_total / tokens,
            }


def test_a_guard_below_one_query_s_spend_would_shut_the_grid_down():
    # The failure the derivation exists to prevent, stated as a measurement: a ceiling
    # under the mass one release costs permits nothing, and every answer in the cell is
    # the mechanism declining rather than a utility result.
    cell = next(iter(_dpvoterag_cells()))
    ceiling = release_ceiling(
        loop_cap=cell["tokens"],
        target_epsilon=float(cell["epsilon_total"]),
        target_delta=0.5 * cell["delta_token"],
        epsilon_token=cell["epsilon_token"],
        delta_token=cell["delta_token"],
        m=int(cell["block"]["m"]),
        k=int(cell["block"]["k"]),
        k_bar=cell["block"].get("k_bar"),
        compose_method=str(cell["block"].get("compose_method", "rdp")),
    )
    assert ceiling == 0


# ---------------------------------------------------------------------------
# Both grids, driven end to end with a stubbed generator
# ---------------------------------------------------------------------------

def _drive(
    monkeypatch, tmp_path, method, *, private_tokens=3, shipped_grid=False,
    log_lines=None, **overrides,
):
    """Run one vote method's sweep with stubs, returning the payloads it wrote.

    ``private_tokens`` is what the generator reports releasing per query, which is the
    only thing separating a cell the mechanism served from one the public path did.
    """
    import json
    from types import SimpleNamespace

    from tests.test_runner_sweep_smoke import _checked_ctor, _load_cfg, _args, _patch_common

    def _set_releases(stub, _kwargs):
        stub.last_run_stats["n_private_tokens"] = private_tokens

    if method == "dpvoterag":
        import experiments.run_dpvoterag as m
        from dp_methods.per_query.dpvoterag.generator import VoteRAGGenerator as real

        gen_name = "VoteRAGGenerator"
    else:
        import experiments.run_dpsparsevoterag as m
        from dp_methods.per_query.dpvoterag.sparse_vote_generator import (
            SparseVoteRAGGenerator as real,
        )

        gen_name = "SparseVoteRAGGenerator"

    _patch_common(monkeypatch, m, tmp_path)
    if log_lines is not None:
        monkeypatch.setattr(m, "log", lambda msg, *a, **kw: log_lines.append(str(msg)))
    monkeypatch.setattr(m, gen_name, _checked_ctor(real, after_init=_set_releases))
    monkeypatch.setattr(m, "DPVoteRAGPipeline", lambda **kw: SimpleNamespace(
        index=lambda d: None, query=lambda q, k=5: "stub answer"))

    m._run_for_llm(
        _load_cfg(method, tmp_path, shipped_grid=shipped_grid, **overrides),
        _args(), "stub/model", "stub__model", False,
    )
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((tmp_path / "out").rglob("*.json"))
    ]


@pytest.mark.parametrize("method", ["dpvoterag", "dpsparsevoterag"])
def test_both_vote_runners_report_what_the_mechanism_released(monkeypatch, tmp_path, method):
    payloads = _drive(monkeypatch, tmp_path, method, private_tokens=3)
    assert payloads
    for pl in payloads:
        assert pl["private_answer_rate"] == pytest.approx(1.0)
        assert pl["public_only"] is False
        assert pl["tokens_private"] > 0


@pytest.mark.parametrize("method", ["dpvoterag", "dpsparsevoterag"])
def test_a_cell_that_released_nothing_says_so_even_though_it_answered(
    monkeypatch, tmp_path, method
):
    # The measured failure: DPSparseVoteRAG's public path returns a scoring answer with
    # `tokens_private == 0`, so a non-empty prediction is not evidence the mechanism
    # ran. Both methods have to state that the same way for the column to be readable
    # across them.
    payloads = _drive(monkeypatch, tmp_path, method, private_tokens=0)
    assert payloads
    for pl in payloads:
        assert pl["predictions"] and all(p.strip() for p in pl["predictions"])
        assert answer_rate(pl["predictions"]) == 1.0
        assert pl["private_answer_rate"] == 0.0
        assert pl["public_only"] is True


@pytest.mark.parametrize("method", ["dpvoterag", "dpsparsevoterag"])
def test_a_cell_with_no_questions_is_not_reported_as_a_cell_the_mechanism_failed(
    monkeypatch, tmp_path, method
):
    """No questions asked and no answer served are different states of the column.

    ``private_answer_rate`` is undefined over zero queries, and a runner that coerced
    that NaN to 0.0 wrote the one value the column exists to distinguish: "the private
    mechanism contributed to none of the answers this cell produced". Both must come
    back null, which the CSV layer renders as a blank cell rather than a measured zero.
    """
    payloads = _drive(
        monkeypatch, tmp_path, method,
        datasets=[{"name": "squad", "max_index_docs": 3, "max_test_queries": 0}],
    )
    assert payloads, "an unservable cell must still be run and reported, never skipped"
    for pl in payloads:
        assert pl["predictions"] == []
        assert pl["private_answer_rate"] is None, (
            "a cell that asked nothing reports a rate as if it had measured one"
        )
        assert pl["public_only"] is None
        # The release count is a genuine zero: nothing was released because nothing
        # was asked, and the count is what says which of the two the null means.
        assert pl["tokens_private"] == 0


@pytest.mark.parametrize("method", ["dpvoterag", "dpsparsevoterag"])
def test_answer_tokens_is_the_bound_on_the_string_that_gets_scored(
    monkeypatch, tmp_path, method
):
    """One definition of the reported answer length, for both vote methods.

    The field exists so a reader can see two methods were compared over answers of the
    same length, which only holds if every runner reports the same quantity: the number
    of tokens its generation loop can emit before it stops. DPSparseVoteRAG's private
    ceiling `c` is not that number — the sparse test fills the rest of the answer from a
    public path `c` never touches — so the loop bound `tmax_absolute` is what it reports.
    """
    block = dict(_cfg(method)[method])
    # Pull the two ceilings apart, which the shipped grid deliberately holds equal: with
    # both at 30 a runner reporting either one is indistinguishable from the other.
    if method == "dpsparsevoterag":
        block["tmax_absolute"] = 24
        block["max_private_tokens"] = 12
        expected = 24
    else:
        block["max_length"] = 24
        block["tmax_absolute"] = None
        expected = 24

    payloads = _drive(monkeypatch, tmp_path, method, **{method: block})

    assert payloads
    for pl in payloads:
        assert pl["answer_tokens"] == expected, (
            f"{method} reports an answer length of {pl['answer_tokens']} while its "
            f"generation loop can emit {expected} tokens"
        )


#: LD-feasible cells per shipped grid. Most of both grids cannot release at all;
#: DPVoteRAG's infeasible cells come back empty, DPSparseVoteRAG's do not -- the public
#: path answers -- which is the whole reason `private_answer_rate` is recorded.
#: Both counts fell by two when the vote stopped being charged on the monotone branch
#: (`VOTE_SENSITIVITY_FACTOR`): the release costs twice its Gumbel parameter, so the
#: derivation returns half the per-token budget and the bottom-out threshold doubles.
LD_FEASIBLE_CELLS = {"dpvoterag": 3, "dpsparsevoterag": 3}


@pytest.mark.parametrize("method", ["dpvoterag", "dpsparsevoterag"])
def test_the_shipped_grid_runs_whole_at_the_full_answer_length(monkeypatch, tmp_path, method):
    """Every cell reports, lands on its target, and is charged the configured length.

    A binding guard is silent by construction: it shortens the answer the accounting
    prices, and cells calibrated to different targets then collapse onto near-identical
    epsilons with nothing in the payload saying why. Reading `release_ceiling` back out
    of all sixteen cells is what rules that out.
    """
    tokens = int(
        _cfg(method)[method]["max_length" if method == "dpvoterag" else "max_private_tokens"]
    )
    cfg = _cfg(method)
    # Derived, not pinned: the shared axis is expected to gain and lose points, and a
    # hardcoded count would fail the next time it does while saying "a cell was skipped".
    expected = len(cfg["epsilon_targets"]) * len(cfg["workload_queries"])
    payloads = _drive(monkeypatch, tmp_path, method, shipped_grid=True)

    assert len(payloads) == expected, "a cell was skipped or two cells shared a path"
    for pl in payloads:
        assert pl["release_ceiling"] == tokens, (
            f"target={pl['epsilon_target']} Q={pl['workload_queries']}: the guard bound "
            f"at {pl['release_ceiling']} releases instead of {tokens}"
        )
        # The guard the runner handed the generator, against the derivation it is
        # meant to track: twice what one query's releases actually spend. Set to the
        # whole workload's delta instead it can never bind, and stops being a check.
        expected_guard = 2.0 * tokens * delta_token_for_workload(
            queries=int(pl["workload_queries"]), tokens=tokens
        )
        assert pl["delta_total"] == pytest.approx(expected_guard)
        if pl["workload_queries"] > 1:
            assert pl["delta_total"] < DEFAULT_DELTA
        assert isinstance(pl["epsilon_workload"], float)
        assert pl["epsilon_workload"] == pytest.approx(pl["epsilon_target"], rel=1e-6)
    assert sum(pl["ld_feasible"] for pl in payloads) == LD_FEASIBLE_CELLS[method]


@pytest.mark.parametrize("method", ["dpvoterag", "dpsparsevoterag"])
def test_a_binding_guard_is_never_silent(monkeypatch, tmp_path, method):
    """Under the paper's pinned deltas the guard does bind — and it has to say so.

    A ceiling below the configured answer length is what collapses cells calibrated to
    different targets onto near-identical epsilons. It is a legitimate configuration, so
    it runs; what it must not do is happen without a line in the run log.
    """
    cfg = _cfg(method)
    block = dict(cfg[method])
    block["delta_token_from_workload"] = False
    block["delta_token"] = 1e-6
    lines: list[str] = []

    payloads = _drive(
        monkeypatch, tmp_path, method, log_lines=lines,
        **{method: block, "delta_total": 1e-5},
    )

    tokens = int(
        block["max_length" if method == "dpvoterag" else "max_private_tokens"]
    )
    assert any(pl["release_ceiling"] < tokens for pl in payloads), (
        "the pinned deltas were expected to bind the guard and did not"
    )
    assert any("BUDGET GUARD BINDS" in line for line in lines), (
        "the guard bound with nothing in the run log to say so"
    )
