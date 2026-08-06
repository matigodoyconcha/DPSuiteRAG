"""The A/B harness must be able to fail.

``scripts/ab_dp_rag_batching.py`` decides whether the fast DP-RAG inner loop changed
the science. A harness that returns PASS on everything would certify a regression as
readily as it certifies noise, so each verdict is exercised here against synthetic
result trees in exactly the layout the two arms write:

* two identical arms                       -> PASS
* answers that changed, correctness even    -> PASS  (the case ab_kv_cache got wrong)
* answers that changed one way              -> FAIL
* an epsilon that moved                     -> FAIL
* retrieval that moved                      -> FAIL
* a control that did not reproduce itself   -> FAIL
* every answer parting at its first token   -> FAIL
* the same number of model calls both arms  -> INCONCLUSIVE
* a paired test with no power                -> INCONCLUSIVE

No model is loaded: the harness reads finished payloads, so the comparison logic is
testable on JSON alone.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.ab_dp_rag_batching import compare

#: Long enough that one changed final token is a ~7% flip rate, i.e. inside the default
#: cap. A pair of answers that differ from the first token would trip the token gate and
#: hide whichever verdict the test is actually about.
_PREFIX = "the answer to this particular question is quite plainly the following word"


def _answer(word: str) -> str:
    return f"{_PREFIX} {word}"


def _payload(
    *,
    method: str = "dp_rag",
    n: int = 20,
    correct_word: str = "alpha",
    predicted: list[str] | None = None,
    epsilon: float = 5.0,
    forwards: float = 270.0,
    retrieved: list[list[int]] | None = None,
    metrics: dict | None = None,
) -> dict:
    """One cell's result payload, shaped like the one the runners save."""
    words = predicted if predicted is not None else [correct_word] * n
    results = []
    for i in range(n):
        gold = _answer(correct_word if isinstance(correct_word, str) else correct_word[i])
        results.append({
            "query": f"q{i}",
            "query_id": f"q{i}",
            "prediction": _answer(words[i]),
            "reference": gold,
            "answers": [gold],
            "retrieved_ids": (retrieved[i] if retrieved else [i, i + 1, i + 2]),
        })
    return {
        "method": method,
        "dataset": "chatdoctor",
        "epsilon": epsilon,
        "epsilon_total": epsilon,
        "workload_queries": 100,
        "delta": 1e-3,
        "n_forwards_per_query": forwards,
        "n_docs_per_query": 128.0,
        "time_per_query_sec": 8.0,
        "metrics": metrics if metrics is not None else {"match_accuracy": 0.5},
        "results": results,
    }


def _tree(root: Path, cells: dict[str, dict]) -> Path:
    """Write ``{relative path: payload}`` as one arm's result tree."""
    for rel, payload in cells.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return root


def _arms(tmp_path: Path, optimised: dict[str, dict], baseline: dict[str, dict]):
    return (
        _tree(tmp_path / "optimised", optimised),
        _tree(tmp_path / "baseline", baseline),
    )


def _control() -> dict:
    return _payload(method="vanilla", n=10, forwards=0.0)


def _cells(dp_rag: dict, control: dict | None = None) -> dict[str, dict]:
    cells = {"dp_rag/seed_1/chatdoctor/eps_target5_q100.json": dp_rag}
    if control is not None:
        cells["vanilla/seed_1/chatdoctor/results.json"] = control
    return cells


# --------------------------------------------------------------------------- verdicts


def test_identical_arms_pass(tmp_path):
    """Nothing moved anywhere: the only verdict that may be returned is PASS."""
    fast, slow = _arms(
        tmp_path,
        _cells(_payload(forwards=270.0), _control()),
        _cells(_payload(forwards=3900.0), _control()),
    )
    summary = compare(fast, slow)
    assert summary["verdict"] == "PASS", summary["errors"] + summary["inconclusive_reasons"]
    assert summary["token_flips"] == 0
    assert summary["median_forward_ratio"] > 1.5


def test_changed_answers_that_are_equally_often_right_pass(tmp_path):
    """Every answer changed, and the changes gain as often as they lose.

    This is the case ``ab_kv_cache.py`` failed DP-RAG on: it capped the fraction of
    changed *answers* at 25%, which for autoregressive sampling measures how far one
    flipped token propagates, not how often a token flipped. Here the flips are exactly
    balanced, so the comparison must pass.
    """
    n = 20
    gold = ["alpha" if i % 2 else "beta" for i in range(n)]
    fast, slow = _arms(
        tmp_path,
        _cells(_payload(n=n, correct_word=gold,
                        predicted=["alpha"] * n, forwards=270.0), _control()),
        _cells(_payload(n=n, correct_word=gold,
                        predicted=["beta"] * n, forwards=3900.0), _control()),
    )
    summary = compare(fast, slow)
    assert summary["answers_changed"] == n
    pooled = summary["pooled_flip_balance"]["match_accuracy"]
    assert pooled["optimised_gained"] == pooled["optimised_lost"] == n // 2
    assert summary["verdict"] == "PASS", summary["errors"] + summary["inconclusive_reasons"]


def test_one_sided_flips_fail(tmp_path):
    """The change helps every time it changes anything: unbiased is exactly what that
    is not, and the two means landing close is no defence."""
    n = 20
    fast, slow = _arms(
        tmp_path,
        _cells(_payload(n=n, predicted=["alpha"] * n, forwards=270.0), _control()),
        _cells(_payload(n=n, predicted=["beta"] * n, forwards=3900.0), _control()),
    )
    summary = compare(fast, slow)
    pooled = summary["pooled_flip_balance"]["match_accuracy"]
    assert (pooled["optimised_gained"], pooled["optimised_lost"]) == (n, 0)
    assert summary["verdict"] == "FAIL"
    assert any("moved one way" in e for e in summary["errors"])


def test_an_epsilon_difference_fails(tmp_path):
    """The budget is a function of the config and the accountant; a forward pass that
    moves it has reached something it must never reach."""
    fast, slow = _arms(
        tmp_path,
        _cells(_payload(epsilon=5.0, forwards=270.0), _control()),
        _cells(_payload(epsilon=5.000001, forwards=3900.0), _control()),
    )
    summary = compare(fast, slow)
    assert summary["verdict"] == "FAIL"
    assert any("epsilon field" in e for e in summary["errors"])


def test_moved_retrieval_fails(tmp_path):
    """Different documents reached the generator, so this is not a generation A/B."""
    moved = [[9, 9, 9] for _ in range(20)]
    fast, slow = _arms(
        tmp_path,
        _cells(_payload(forwards=270.0), _control()),
        _cells(_payload(forwards=3900.0, retrieved=moved), _control()),
    )
    summary = compare(fast, slow)
    assert summary["verdict"] == "FAIL"
    assert any("retrieved document ids" in e for e in summary["errors"])


def test_a_control_that_moved_fails(tmp_path):
    """The control reads none of the switches. If it moved, the harness is measuring
    itself and nothing it says about DP-RAG is attributable to the change."""
    fast, slow = _arms(
        tmp_path,
        _cells(_payload(forwards=270.0), _payload(method="vanilla", n=10, forwards=0.0)),
        _cells(_payload(forwards=3900.0),
               _payload(method="vanilla", n=10, predicted=["beta"] * 10, forwards=0.0)),
    )
    summary = compare(fast, slow)
    assert summary["verdict"] == "FAIL"
    assert any("control changed" in e for e in summary["errors"])


def test_no_saved_forwards_is_inconclusive(tmp_path):
    """Both arms made the same number of model calls, so the switches did nothing and
    the two arms ran the same code. That is not a pass."""
    fast, slow = _arms(
        tmp_path,
        _cells(_payload(forwards=3900.0), _control()),
        _cells(_payload(forwards=3900.0), _control()),
    )
    summary = compare(fast, slow)
    assert summary["verdict"] == "INCONCLUSIVE"
    assert any("forward-count ratio" in r for r in summary["inconclusive_reasons"])


def test_a_knob_that_leaves_the_forward_count_alone_falls_back_to_the_clock(tmp_path):
    """Under --isolate sort / kv_cache both arms make the same number of model calls by
    construction, so the forward ratio cannot say whether the switch was honoured and
    the harness must not read a pass out of it."""
    fast, slow = _arms(
        tmp_path,
        _cells(_payload(forwards=270.0), _control()),
        _cells(_payload(forwards=270.0), _control()),
    )
    summary = compare(fast, slow, evidence="speed")
    assert summary["verdict"] == "INCONCLUSIVE"
    assert any("per-query speedup" in r for r in summary["inconclusive_reasons"])


def test_a_missing_control_is_inconclusive(tmp_path):
    """Without the control there is no way to tell a harness fault from a real effect."""
    fast, slow = _arms(
        tmp_path,
        _cells(_payload(forwards=270.0)),
        _cells(_payload(forwards=3900.0)),
    )
    summary = compare(fast, slow)
    assert summary["verdict"] == "INCONCLUSIVE"
    assert any("vanilla" in r for r in summary["inconclusive_reasons"])
    # ...and asking for no control is allowed, as long as it is asked for.
    assert compare(fast, slow, expect_control=False)["verdict"] == "PASS"


def test_too_few_discordant_pairs_is_inconclusive(tmp_path):
    """An exact McNemar on four discordant pairs cannot reach p<0.05 at any split, so a
    pass from it would be unearned."""
    n = 4
    fast, slow = _arms(
        tmp_path,
        _cells(_payload(n=n, predicted=["alpha"] * n, forwards=270.0), _control()),
        _cells(_payload(n=n, predicted=["beta"] * n, forwards=3900.0), _control()),
    )
    summary = compare(fast, slow)
    assert summary["verdict"] == "INCONCLUSIVE"
    assert any("discordant pairs" in r for r in summary["inconclusive_reasons"])


def _diverge_from_the_first_token(root: Path) -> None:
    """Rewrite an arm's DP-RAG answers so no answer shares its opening token."""
    cell = next(root.rglob("dp_rag/**/*.json"))
    payload = json.loads(cell.read_text(encoding="utf-8"))
    for item in payload["results"]:
        item["prediction"] = "completely different text entirely"
    cell.write_text(json.dumps(payload), encoding="utf-8")


def test_disagreement_on_the_very_first_token_fails(tmp_path):
    """Both arms invert the same seeded uniform, so a ~1e-6 perturbation cannot move the
    sampled token at nearly every step at any temperature. A per-step hazard of ~1 means
    the two paths compute different things — and unlike the flip rate, that argument does
    not depend on where the cell sits on the epsilon axis."""
    n = 20
    fast, slow = _arms(
        tmp_path,
        _cells(_payload(n=n, forwards=270.0), _control()),
        _cells(_payload(n=n, forwards=3900.0), _control()),
    )
    _diverge_from_the_first_token(slow)

    summary = compare(fast, slow)
    assert summary["first_step_divergence_rate"] == 1.0
    assert summary["verdict"] == "FAIL"
    assert any("very first token" in e for e in summary["errors"])


def test_the_token_flip_rate_is_reported_but_not_gated_by_default(tmp_path):
    """The rate a correct implementation produces has never been measured here and
    depends on the regime, so a guessed cap could only fail a correct implementation.
    It gates only when the operator names a number."""
    n = 20
    # Every answer changes at its last token: a 1/13 flip rate, no first-token divergence,
    # and the changes gain as often as they lose so only the rate is in question.
    gold = ["alpha" if i % 2 else "beta" for i in range(n)]
    fast, slow = _arms(
        tmp_path,
        _cells(_payload(n=n, correct_word=gold, predicted=["alpha"] * n,
                        forwards=270.0), _control()),
        _cells(_payload(n=n, correct_word=gold, predicted=["beta"] * n,
                        forwards=3900.0), _control()),
    )
    lenient = compare(fast, slow)
    assert lenient["verdict"] == "PASS"
    assert lenient["token_flip_rate"] > 0.05
    assert lenient["criterion"]["token_flip_rate_gated"] is False
    assert not any("decoding steps" in e for e in lenient["errors"])

    strict = compare(fast, slow, 0.01)
    assert strict["criterion"]["token_flip_rate_gated"] is True
    assert strict["verdict"] == "FAIL"
    assert any("decoding steps sampled a different token" in e for e in strict["errors"])


def test_a_metric_outside_the_band_fails(tmp_path):
    """The backstop: a shift too large to be a handful of resolved ties."""
    fast, slow = _arms(
        tmp_path,
        _cells(_payload(forwards=270.0, metrics={"match_accuracy": 0.90}), _control()),
        _cells(_payload(forwards=3900.0, metrics={"match_accuracy": 0.50}), _control()),
    )
    summary = compare(fast, slow)
    assert summary["verdict"] == "FAIL"
    assert any("match_accuracy moved" in e for e in summary["errors"])


def test_the_noise_band_caveat_travels_with_the_verdict(tmp_path):
    """The tolerances are not a measured seed-to-seed spread, and a reader of either
    half of the output must be told so without digging."""
    from scripts.ab_dp_rag_batching import NOISE_BAND_CAVEAT, render

    fast, slow = _arms(
        tmp_path,
        _cells(_payload(forwards=270.0), _control()),
        _cells(_payload(forwards=3900.0), _control()),
    )
    summary = compare(fast, slow)
    assert summary["criterion"]["metric_tolerance_caveat"] == NOISE_BAND_CAVEAT
    assert "never run repeated seeds" in render(summary, arms=None)
