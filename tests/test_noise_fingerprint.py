"""The noise fingerprint, and the smoke check that reads it.

`run_seed` is supposed to move the mechanism's noise and nothing else, and the
per-seed error bars in the thesis are only honest if it does. A run cannot show
that on its own: when the model collapses onto one phrase, every voter proposes
the same token, the vote histogram is a single spike, and no Gumbel draw can move
the argmax — two seeds then produce byte-identical output whether the seed reached
the sampler or not. These tests pin the fingerprint that tells the two apart, the
guarantee that recording it changes no result, and the smoke check's verdicts.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from core.noise_trace import FINGERPRINT_ENV

from dp_methods.per_query.dpvoterag.generator import VoteRAGGenerator
from dp_methods.per_query.dpvoterag.sparse_vote_generator import SparseVoteRAGGenerator
from dp_methods.per_query.private_rag.generator import PrivateRAGGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def diagnostic_mode(monkeypatch):
    """The fingerprint is a diagnostic: nothing records it unless it is asked for."""
    monkeypatch.setenv(FINGERPRINT_ENV, "1")

DOCS = [f"Document {i} about treatment number {i} for a common ailment." for i in range(12)]
QUERY = "What treats the ailment?"


def _verify_smoke():
    """Import scripts/verify_smoke.py, which lives outside any package."""
    spec = importlib.util.spec_from_file_location(
        "verify_smoke", PROJECT_ROOT / "scripts" / "verify_smoke.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_smoke"] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------
# A model that makes every voter agree: the degenerate case the check must survive
# ----------------------------------------------------------------------------


class _OneTokenModel:
    """Always proposes the same token, whatever the context — a spike histogram."""

    def __init__(self) -> None:
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def __call__(self, input_ids: torch.Tensor, **_kwargs):
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 10), dtype=torch.float32)
        logits[:, -1, 1] = 1.0
        return SimpleNamespace(logits=logits)


class _FixedTokenizer:
    eos_token_id = 0
    vocab_size = 10
    pad_token_id = 0

    def __call__(self, text, return_tensors: str = "pt", **_kwargs):
        batch = text if isinstance(text, list) else [text]
        ids = [[1, 2, 3]] * len(batch)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.ones((len(batch), 3), dtype=torch.long),
            }
        return {"input_ids": ids[0] if len(batch) == 1 else ids}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return " ".join(str(t) for t in token_ids)


def _degenerate_generator(seed: int) -> VoteRAGGenerator:
    return VoteRAGGenerator(
        model=_OneTokenModel(),
        tokenizer=_FixedTokenizer(),
        m=6,
        k=1,
        epsilon_token=20.0,
        delta_token=1e-5,
        epsilon_total=500.0,
        delta_total=1e-4,
        k_bar=6,
        random_seed=seed,
        max_length=4,
        tmax_absolute=4,
        compose_method="simple",
    )


def test_a_spike_histogram_gives_the_same_answer_but_a_different_fingerprint(diagnostic_mode):
    answers, fingerprints = [], []
    for seed in (1, 2):
        gen = _degenerate_generator(seed)
        answers.append(gen.generate_vote(QUERY, DOCS[:6]))
        fingerprints.append(gen.last_run_stats["noise_fingerprint"])

    assert answers[0] == answers[1], "the fixture is meant to be the degenerate case"
    assert all(f is not None for f in fingerprints)
    assert fingerprints[0] != fingerprints[1], "run_seed did not reach the sampler"


def test_the_fingerprint_repeats_for_a_repeated_seed(diagnostic_mode):
    first = _degenerate_generator(3)
    second = _degenerate_generator(3)
    first.generate_vote(QUERY, DOCS[:6])
    second.generate_vote(QUERY, DOCS[:6])
    assert (
        first.last_run_stats["noise_fingerprint"]
        == second.last_run_stats["noise_fingerprint"]
    )


def test_the_fingerprint_follows_the_query_within_one_seed(diagnostic_mode):
    gen = _degenerate_generator(4)
    gen.generate_vote("What treats the ailment?", DOCS[:6])
    first = gen.last_run_stats["noise_fingerprint"]
    gen.generate_vote("Which drug is prescribed?", DOCS[:6])
    assert gen.last_run_stats["noise_fingerprint"] != first


# ----------------------------------------------------------------------------
# Recording the noise must not consume any: the emitted tokens are pinned
# ----------------------------------------------------------------------------

#: Captured from the three generators before the fingerprint existed, at seed 7 on
#: the tiny random GPT-2. Recording a draw the mechanism already makes must leave
#: the noise stream exactly where it was, so these cannot move.
GOLDEN_ANSWERS = {
    "dpvoterag": ":::::: Bowen Bowen",
    "dpsparsevoterag": ":::::: Bowen Bowen",
    "private_rag": " powerfully strokeavoredGil addict Tripreatmentreatment",
}
GOLDEN_PRIVATE_TOKENS = {"dpvoterag": 8, "dpsparsevoterag": 3, "private_rag": 6}


def _tiny_generators(position_sensitive_lm):
    model, tokenizer = position_sensitive_lm
    return {
        "dpvoterag": VoteRAGGenerator(
            model=model, tokenizer=tokenizer, m=4, k=1, epsilon_token=20.0,
            delta_token=1e-5, epsilon_total=5000.0, delta_total=1e-4, k_bar=4,
            random_seed=7, max_length=8, tmax_absolute=8, compose_method="simple",
        ),
        "dpsparsevoterag": SparseVoteRAGGenerator(
            model=model, tokenizer=tokenizer, m=4, k=1, epsilon_token=40.0,
            delta_token=1e-5, epsilon_total=5000.0, delta_total=1e-4, k_bar=4,
            random_seed=7, tmax_absolute=8, max_private_tokens=4,
        ),
        "private_rag": PrivateRAGGenerator(
            model=model, tokenizer=tokenizer, m=4, k=1, epsilon=8.0, epsilon_0=1.0,
            random_seed=7, t_max=8,
        ),
    }


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("method", sorted(GOLDEN_ANSWERS))
def test_recording_the_noise_consumes_none_of_it(
    position_sensitive_lm, monkeypatch, method, enabled
):
    monkeypatch.setenv(FINGERPRINT_ENV, "1" if enabled else "")
    gen = _tiny_generators(position_sensitive_lm)[method]
    answer = gen.generate_vote(QUERY, DOCS)
    assert answer == GOLDEN_ANSWERS[method]
    assert gen.last_run_stats["n_private_tokens"] == GOLDEN_PRIVATE_TOKENS[method]


@pytest.mark.parametrize("method", sorted(GOLDEN_ANSWERS))
def test_every_vote_generator_reports_a_fingerprint(diagnostic_mode, position_sensitive_lm, method):
    gen = _tiny_generators(position_sensitive_lm)[method]
    gen.generate_vote(QUERY, DOCS)
    fingerprint = gen.last_run_stats["noise_fingerprint"]
    assert isinstance(fingerprint, str) and fingerprint


def test_a_query_that_draws_no_noise_reports_no_fingerprint(diagnostic_mode, position_sensitive_lm):
    gen = _tiny_generators(position_sensitive_lm)["dpvoterag"]
    assert gen.generate_vote(QUERY, []) == ""
    assert gen.last_run_stats["noise_fingerprint"] is None


# ----------------------------------------------------------------------------
# What the smoke check makes of it
# ----------------------------------------------------------------------------


def _write_cell(root: Path, seed: int, fingerprints: list[str | None]) -> None:
    """One method's payload at one seed, with identical text at every seed."""
    path = root / f"seed_{seed}" / "dpvoterag" / "chatdoctor"
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "dpvoterag",
        "dataset": "chatdoctor",
        "run_seed": seed,
        "workload_queries": 2,
        "epsilon_workload": 1.0 * seed,
        "epsilon_workload_by_q": {"1": 1.0, "2": 1.0},
        "results": [
            {
                "query_id": f"q{i}",
                "prediction": "I am a doctor. I am a doctor.",
                "exit_status": 0,
                "noise_fingerprint": fingerprints[i],
            }
            for i in range(len(fingerprints))
        ],
    }
    (path / "target_40.json").write_text(json.dumps(payload), encoding="utf-8")


def test_differing_fingerprints_under_identical_text_are_a_note(tmp_path):
    _write_cell(tmp_path, 1, ["aaaa1111", "bbbb2222"])
    _write_cell(tmp_path, 2, ["cccc3333", "dddd4444"])
    rep = _verify_smoke().verify(tmp_path)
    assert not [f for f in rep.failures if "varies across seeds" in f]
    assert any("noise_fingerprint" in n for n in rep.notes)


def test_identical_fingerprints_are_the_failure(tmp_path):
    _write_cell(tmp_path, 1, ["aaaa1111", "bbbb2222"])
    _write_cell(tmp_path, 2, ["aaaa1111", "bbbb2222"])
    rep = _verify_smoke().verify(tmp_path)
    assert any("run_seed is not reaching the noise" in f for f in rep.failures)


def test_a_payload_without_a_fingerprint_says_it_cannot_decide(tmp_path):
    _write_cell(tmp_path, 1, [None, None])
    _write_cell(tmp_path, 2, [None, None])
    rep = _verify_smoke().verify(tmp_path)
    undecided = [f for f in rep.failures if "seeds are distinguishable" in f]
    assert undecided, rep.failures
    assert "BENCHDP_NOISE_FINGERPRINT=1" in undecided[0]
    assert "is not reaching" not in undecided[0], "an absent fingerprint is not a verdict"


# ----------------------------------------------------------------------------
# The diagnostic is off unless asked for
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("method", sorted(GOLDEN_ANSWERS))
def test_nothing_derived_from_the_noise_is_recorded_by_default(
    position_sensitive_lm, monkeypatch, method
):
    """A measurement run must carry no trace of the noise stream. See core/noise_trace."""
    monkeypatch.delenv(FINGERPRINT_ENV, raising=False)
    gen = _tiny_generators(position_sensitive_lm)[method]
    gen.generate_vote(QUERY, DOCS)
    assert gen.last_run_stats["noise_fingerprint"] is None


def test_the_smoke_run_turns_the_diagnostic_on(monkeypatch):
    """`runner.py --smoke` is the one path that records it, and it does so for itself."""
    import experiments.runner as runner

    monkeypatch.delenv(FINGERPRINT_ENV, raising=False)
    monkeypatch.setattr(sys, "argv", ["runner.py", "--smoke", "--methods", "vanilla"])
    # Stop the run the moment the mode is settled: the flag has to be set by then,
    # because every method subprocess inherits this environment.
    def _stop(*_a, **_k):
        raise RuntimeError("stop")

    monkeypatch.setattr(runner, "setup_summary_logging", _stop)
    with pytest.raises(RuntimeError, match="stop"):
        runner.main()
    assert os.environ.get(FINGERPRINT_ENV) == "1"
