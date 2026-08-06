"""The smoke gate's checks, exercised against payloads that should and should not pass.

`scripts/verify_smoke.py` is what stands between a broken protocol and a multi-day grid,
and a check that cannot fail is worse than no check: it prints [PASS] and buys the run a
confidence nothing measured. So each invariant is driven twice here — once over a run
that satisfies it, once over the same run with exactly that property broken — and the
failure is matched by the text the script would print, so a check silently reworded into
a tautology stops passing these.

The clean fixture is a whole eight-method smoke run in miniature: two seeds, two cells per
method, the fields the analysis reads. Every fault below is one mutation of it.

The seed comparison is not re-tested here; `tests/test_noise_fingerprint.py` owns it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Every method the protocol runs. `private_rag_ada` is the same runner under a variant,
#: which is how the payload names it and how the gate has to tell the two apart.
METHODS = (
    "vanilla",
    "dp_rag",
    "dp_synrag",
    "dpksa",
    "dpvoterag",
    "dpsparsevoterag",
    "private_rag",
    "private_rag_ada",
)
VOTE_METHODS = ("dpvoterag", "dpsparsevoterag")

QUESTIONS = ("q0", "q1", "q2", "q3")
ANSWER_TOKENS = 30
#: Two cells per method, at two points of the shared axis: one point could not show that
#: the budget knob moves the method along it.
CELLS = {"cell_a.json": 1.5, "cell_b.json": 4.0}


def _verify_smoke():
    """Import scripts/verify_smoke.py, which lives outside any package."""
    spec = importlib.util.spec_from_file_location(
        "verify_smoke", PROJECT_ROOT / "scripts" / "verify_smoke.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_smoke"] = module
    spec.loader.exec_module(module)
    return module


def _payload(method: str, seed: int, cell: str, epsilon: float) -> dict:
    # The baseline has no mechanism, so it repeats exactly and prices no workload;
    # everything else moves with the seed and carries the workload block.
    text = "an answer" if method == "vanilla" else f"an answer from seed {seed}"
    data = {
        "method": "private_rag" if method == "private_rag_ada" else method,
        "dataset": "chatdoctor",
        "run_seed": seed,
        "answer_tokens": ANSWER_TOKENS,
        "metrics": {"answer_rate": 1.0},
        "results": [{"query_id": q, "prediction": text} for q in QUESTIONS],
    }
    if method == "private_rag_ada":
        data["variant"] = "murag_ada"
    if method != "vanilla":
        data["workload_queries"] = 5
        data["epsilon_workload"] = epsilon
        data["workload_regime"] = "per_query"
        # A per-query method spends more of its budget as the stream grows; a flat
        # curve under that regime means the workload never reached the accountant.
        data["epsilon_workload_by_q"] = {"1": epsilon / 5.0, "5": epsilon}
    if method in VOTE_METHODS:
        # What the mechanism released, as opposed to what the generator wrote.
        data["tokens_private"] = 12
        data["public_only"] = False
        data["private_answer_rate"] = 1.0
    return data


def _write_run(root: Path, mutate=None) -> Path:
    """A clean two-seed smoke run, optionally with one payload property broken.

    `mutate(method, seed, cell, payload)` may edit the payload in place.
    """
    for seed in (1, 2):
        for method in METHODS:
            for cell, epsilon in CELLS.items():
                data = _payload(method, seed, cell, epsilon)
                if mutate is not None:
                    mutate(method, seed, cell, data)
                out = root / f"seed_{seed}" / method / "chatdoctor"
                out.mkdir(parents=True, exist_ok=True)
                (out / cell).write_text(json.dumps(data), encoding="utf-8")
    return root


def _run(root: Path):
    return _verify_smoke().verify(root)


def _failed(rep, fragment: str) -> list[str]:
    return [f for f in rep.failures if fragment in f]


# ---------------------------------------------------------------------------
# The fixture the mutations are measured against
# ---------------------------------------------------------------------------


def test_a_clean_run_passes_every_check(tmp_path):
    rep = _run(_write_run(tmp_path))
    assert rep.failures == []


def test_an_empty_results_directory_is_a_failure_not_a_pass(tmp_path):
    """The one input that would otherwise satisfy every loop by having nothing to loop."""
    rep = _run(tmp_path)
    assert _failed(rep, "results present")


# ---------------------------------------------------------------------------
# One answer length across the eight methods
# ---------------------------------------------------------------------------


def test_a_method_answering_at_another_length_fails(tmp_path):
    def mutate(method, seed, cell, data):
        if method == "dpksa":
            data["answer_tokens"] = 16

    rep = _run(_write_run(tmp_path, mutate))
    assert _failed(rep, "one answer length across methods")


def test_a_method_that_records_no_answer_length_fails(tmp_path):
    def mutate(method, seed, cell, data):
        if method == "dp_synrag":
            del data["answer_tokens"]

    rep = _run(_write_run(tmp_path, mutate))
    failures = _failed(rep, "every method records answer_tokens")
    assert failures and "dp_synrag" in failures[0]


def test_a_run_where_nothing_records_the_answer_length_fails(tmp_path):
    def mutate(method, seed, cell, data):
        del data["answer_tokens"]

    rep = _run(_write_run(tmp_path, mutate))
    assert _failed(rep, "answer_tokens recorded")


# ---------------------------------------------------------------------------
# The vote mechanisms actually released something
# ---------------------------------------------------------------------------


def test_a_vote_cell_that_answered_through_the_public_path_fails(tmp_path):
    """The fault the gate exists for, and the one `answer_rate` cannot see.

    DPSparseVoteRAG's Sparse Vector Technique routes a token down a budget-free public
    path when the private branch declines, so a cell whose mechanism never fired still
    answers every question. Its `answer_rate` is 1.0 and its text metrics look normal;
    only `tokens_private` says no DP mechanism ran.
    """

    def mutate(method, seed, cell, data):
        if method == "dpsparsevoterag":
            data["tokens_private"] = 0
            data["public_only"] = True
            data["private_answer_rate"] = 0.0

    rep = _run(_write_run(tmp_path, mutate))
    failures = _failed(rep, "the mechanism released tokens")
    assert failures and "dpsparsevoterag" in failures[0]
    # The point of the check: the run looks perfectly healthy from the answers alone.
    assert "every cell is public_only" in failures[0]
    assert not _failed(rep, "answer_rate recorded")
    assert not _failed(rep, "dpvoterag: the mechanism released tokens")


def test_one_firing_cell_is_enough_for_the_vote_check(tmp_path):
    """A smoke set need not fire at every point — the tight end legitimately releases
    nothing. It must fire somewhere, or it gated the grid without running the mechanism."""

    def mutate(method, seed, cell, data):
        if method in VOTE_METHODS and cell == "cell_a.json":
            data["tokens_private"] = 0
            data["public_only"] = True
            data["private_answer_rate"] = 0.0

    rep = _run(_write_run(tmp_path, mutate))
    assert not _failed(rep, "the mechanism released tokens")
    assert any("public_only" in n for n in rep.notes)


def test_a_vote_payload_that_never_says_whether_it_fired_fails(tmp_path):
    def mutate(method, seed, cell, data):
        if method == "dpvoterag":
            del data["tokens_private"]

    rep = _run(_write_run(tmp_path, mutate))
    assert _failed(rep, "dpvoterag: tokens_private recorded")
    # And the same absence is caught as a missing analysis field.
    assert _failed(rep, "dpvoterag: payload fields")


@pytest.mark.parametrize("field", ["private_answer_rate", "public_only"])
def test_a_vote_payload_missing_a_mechanism_field_fails(tmp_path, field):
    def mutate(method, seed, cell, data):
        if method == "dpsparsevoterag":
            del data[field]

    rep = _run(_write_run(tmp_path, mutate))
    failures = _failed(rep, "dpsparsevoterag: payload fields")
    assert failures and field in failures[0]


# ---------------------------------------------------------------------------
# The fields the analysis reads, and the axis it plots them on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["run_seed", "workload_queries", "epsilon_workload", "epsilon_workload_by_q"]
)
def test_a_missing_analysis_field_fails(tmp_path, field):
    def mutate(method, seed, cell, data):
        if method == "dp_rag":
            data.pop(field, None)

    rep = _run(_write_run(tmp_path, mutate))
    failures = _failed(rep, "dp_rag: payload fields")
    assert failures and field in failures[0]


def test_the_baseline_is_not_asked_for_a_workload_block(tmp_path):
    """`vanilla` prices no mechanism; demanding a workload epsilon of it would make the
    gate fail on a correct run, which is the other way a check goes wrong."""
    rep = _run(_write_run(tmp_path))
    assert not _failed(rep, "vanilla")


def test_a_budget_knob_that_does_not_move_the_axis_fails(tmp_path):
    def mutate(method, seed, cell, data):
        if method == "dpksa":
            data["epsilon_workload"] = 2.0

    rep = _run(_write_run(tmp_path, mutate))
    assert _failed(rep, "dpksa: distinct epsilon_workload per point")


def test_a_method_that_reached_no_point_is_a_note_not_a_failure(tmp_path):
    """A target the calibration cannot reach is reported as absent, not clamped."""

    def mutate(method, seed, cell, data):
        if method == "dpksa":
            data["epsilon_workload"] = None
            data["workload_error"] = "no budget reaches target=10"

    rep = _run(_write_run(tmp_path, mutate))
    assert not _failed(rep, "dpksa: distinct epsilon_workload")
    assert any("no point on the axis" in n for n in rep.notes)


# ---------------------------------------------------------------------------
# One question set, one workload regime, and empty answers visible as empty
# ---------------------------------------------------------------------------


def test_methods_answering_different_questions_fail(tmp_path):
    def mutate(method, seed, cell, data):
        if method == "dp_synrag":
            data["results"] = [
                {"query_id": f"other_{q}", "prediction": "an answer"} for q in QUESTIONS
            ]

    rep = _run(_write_run(tmp_path, mutate))
    # The cross-method comparison, not the across-seeds one: the shifted set is stable
    # under the run seed, so only the "method vs method" check has anything to say.
    assert _failed(rep, "vs dp_rag: same questions")
    assert not _failed(rep, "same questions across seeds")


def test_a_cell_whose_questions_move_with_the_run_seed_fails(tmp_path):
    """The corpus and the sample follow `data_seed`; only the mechanism follows the run
    seed. If the questions move too, the per-seed error bars mix two sources of variance."""

    def mutate(method, seed, cell, data):
        if method == "dp_rag" and seed == 2:
            for i, item in enumerate(data["results"]):
                item["query_id"] = f"shifted_{i}"

    rep = _run(_write_run(tmp_path, mutate))
    assert _failed(rep, "same questions across seeds")


def test_a_flat_curve_under_a_per_query_regime_fails(tmp_path):
    def mutate(method, seed, cell, data):
        if method == "private_rag":
            data["epsilon_workload_by_q"] = {"1": 1.5, "5": 1.5}

    rep = _run(_write_run(tmp_path, mutate))
    assert _failed(rep, "private_rag: per-query cost grows with Q")


def test_a_moving_curve_under_a_one_off_regime_fails(tmp_path):
    """DP-SynRAG pays once for the corpus, so its cost cannot depend on how many
    questions are then asked of it."""

    def mutate(method, seed, cell, data):
        if method == "dp_synrag":
            data["workload_regime"] = "one_shot"

    rep = _run(_write_run(tmp_path, mutate))
    assert _failed(rep, "dp_synrag: one_shot cost is flat in Q")


def test_empty_answers_without_an_answer_rate_fail(tmp_path):
    """An empty answer averaged in as a wrong one is the difference between a method
    that declined and a method that was wrong."""

    def mutate(method, seed, cell, data):
        if method == "dpksa":
            data["metrics"] = {}
            for item in data["results"]:
                item["prediction"] = "   "

    rep = _run(_write_run(tmp_path, mutate))
    assert _failed(rep, "dpksa: answer_rate recorded")


def test_empty_answers_with_an_answer_rate_are_a_note(tmp_path):
    def mutate(method, seed, cell, data):
        if method == "dpksa":
            data["metrics"] = {"answer_rate": 0.5}
            data["results"][0]["prediction"] = ""

    rep = _run(_write_run(tmp_path, mutate))
    assert not _failed(rep, "answer_rate recorded")
    assert any("empty" in n for n in rep.notes)


def test_a_baseline_that_moves_between_seeds_fails(tmp_path):
    """Greedy decoding over a fixed corpus has nothing to vary; if it varies, something
    other than the mechanism is following the run seed."""

    def mutate(method, seed, cell, data):
        if method == "vanilla" and seed == 2:
            for item in data["results"]:
                item["prediction"] = "a different answer"

    rep = _run(_write_run(tmp_path, mutate))
    assert _failed(rep, "the baseline repeats exactly")
