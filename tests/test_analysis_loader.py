"""Smoke tests for the analysis result loader.

Loads the real results/ directory and checks that every CanonicalRecord
is well-formed, that per-method invariants hold, and that the summary
DataFrame can be built without errors.
"""
from __future__ import annotations

import math
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.load_results import (
    CanonicalRecord,
    load_all,
    pareto_front,
)

RESULTS_DIR = PROJECT_ROOT / "results"

_VALID_EPSILON_KINDS = {"input_total", "spent", "derived_zcdp", "none", "na"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_records() -> list[CanonicalRecord]:
    # Smoke tests over a local results/ tree, which a fresh clone does not have.
    records = load_all(RESULTS_DIR, recompute_vanilla=False)
    if not records:
        pytest.skip("results/ holds no result JSONs (run an experiment first)")
    return records


# ---------------------------------------------------------------------------
# Global invariants
# ---------------------------------------------------------------------------

def test_load_all_returns_at_least_ten_records(all_records):
    assert len(all_records) >= 10, (
        f"Expected ≥10 records, got {len(all_records)}.  "
        "Check that results/ is populated."
    )


def test_all_epsilon_kinds_are_valid(all_records):
    for r in all_records:
        assert r.epsilon_kind in _VALID_EPSILON_KINDS, (
            f"Unknown epsilon_kind '{r.epsilon_kind}' in {r.source_file}"
        )


def test_all_records_have_method_and_dataset(all_records):
    for r in all_records:
        assert r.method, f"Empty method in {r.source_file}"
        assert r.dataset, f"Empty dataset in {r.source_file}"


# ---------------------------------------------------------------------------
# dp_rag_quick
# ---------------------------------------------------------------------------

def test_dp_rag_quick_epsilon_kind(all_records):
    # The small dp_rag config is now the paper_faithful variant (was "quick").
    # These are smoke tests over the real results/ dir: skip when that data is absent.
    dp_rag_small = [
        r for r in all_records
        if r.method == "dp_rag" and r.variant in ("quick", "paper_faithful")
    ]
    if not dp_rag_small:
        pytest.skip("No dp_rag quick/paper_faithful records in results/")
    for r in dp_rag_small:
        assert r.epsilon_kind == "input_total", (
            f"dp_rag should have epsilon_kind='input_total', got '{r.epsilon_kind}'"
        )


def test_dp_rag_quick_epsilon_sum_approx_15(all_records):
    # eps_r=10.0, eps_g=5.0 → sum ≈ 15.0 (the old small-config headline point)
    target = [
        r for r in all_records
        if r.method == "dp_rag"
        and r.variant in ("quick", "paper_faithful")
        and r.epsilon_sum is not None
        and math.isclose(r.epsilon_sum, 15.0, abs_tol=0.1)
    ]
    if not target:
        pytest.skip("No dp_rag record with epsilon_sum≈15 in results/")


def test_dp_rag_quick_eps15_no_composed_epsilon(all_records):
    # The eps_r10+eps_g5 file has no privacy_accounting → epsilon_composed must be None.
    target = [
        r for r in all_records
        if r.method == "dp_rag"
        and r.variant in ("quick", "paper_faithful")
        and r.epsilon_sum is not None
        and abs(r.epsilon_sum - 15.0) < 0.1
    ]
    if not target:
        pytest.skip("No dp_rag record with epsilon_sum≈15 in results/")
    for r in target:
        assert r.epsilon_composed is None, (
            f"eps_r10+eps_g5 file should lack privacy_accounting, "
            f"got epsilon_composed={r.epsilon_composed}"
        )


# ---------------------------------------------------------------------------
# dpvoterag
# ---------------------------------------------------------------------------

def test_dpvoterag_epsilon_spent_is_read_from_the_payload(tmp_path):
    """Built from a payload rather than from whatever `results/` happens to hold.

    Reading the live results tree made this assert on data the repository does not
    own: it failed whenever dpvoterag had not been run, and any stray directory could
    break or silently satisfy it. What it is really checking is that the normaliser
    picks `dp_spent.epsilon_total_spent` up, and that is a property of the loader.
    """
    run = tmp_path / "run_20260101_000000" / "dpvoterag" / "triviaqa"
    run.mkdir(parents=True)
    (run / "epsilon_8.0.json").write_text(json.dumps({
        "method": "dpvoterag",
        "dataset": "triviaqa",
        "llm": "facebook/opt-1.3b",
        "epsilon_total": 8.0,
        "dp_spent": {"epsilon_total_spent": 7.25},
        "metrics": {"contains_accuracy": 0.4},
        "predictions": ["a"],
        "references": ["a"],
    }), encoding="utf-8")

    records = load_all(results_dir=tmp_path, recompute_vanilla=False)
    assert len(records) == 1
    assert records[0].method == "dpvoterag"
    assert records[0].epsilon_allocated == 8.0
    assert records[0].epsilon_spent == 7.25


def test_dpvoterag_privacy_key_is_canonical(all_records):
    dpvote = [r for r in all_records if r.method == "dpvoterag"]
    for r in dpvote:
        # The flat 'mia_auc' alias must be normalised to the canonical
        # 's2mia_auc'; the raw alias must never leak into the privacy dict.
        assert "mia_auc" not in r.privacy, (
            f"Raw 'mia_auc' key should have been normalised away in {r.source_file}"
        )
        # Privacy (AUC-MIA) is optional — only the separate --mia-only run
        # populates it — but when present it must use the canonical key.
        if r.privacy:
            assert "s2mia_auc" in r.privacy, (
                f"Expected canonical 's2mia_auc' key, got {list(r.privacy.keys())} "
                f"in {r.source_file}"
            )


# ---------------------------------------------------------------------------
# vanilla
# ---------------------------------------------------------------------------

def test_vanilla_epsilon_kind_is_none(all_records):
    vanilla = [r for r in all_records if r.method == "vanilla"]
    if not vanilla:
        pytest.skip("No vanilla records in results/")
    for r in vanilla:
        assert r.epsilon_kind == "none", (
            f"vanilla should have epsilon_kind='none', got '{r.epsilon_kind}'"
        )


def test_vanilla_epsilon_value_is_none(all_records):
    vanilla = [r for r in all_records if r.method == "vanilla"]
    for r in vanilla:
        assert r.epsilon_value is None


# ---------------------------------------------------------------------------
# pareto_front helper
# ---------------------------------------------------------------------------

def test_pareto_front_basic():
    points = [(0.6, 0.8), (0.7, 0.9), (0.5, 0.7), (0.6, 0.6)]
    front = pareto_front(points)
    # (0.7, 0.9) is dominated by nothing; (0.5, 0.7) may also be on front
    assert (0.7, 0.9) in front


def test_pareto_front_excludes_dominated():
    # (0.6, 0.6) is dominated by (0.6, 0.8)
    points = [(0.6, 0.8), (0.6, 0.6)]
    front = pareto_front(points)
    assert (0.6, 0.6) not in front


def test_pareto_front_handles_nan():
    points = [(float("nan"), 0.5), (0.6, 0.8)]
    front = pareto_front(points)
    assert all(not (math.isnan(x) or math.isnan(y)) for x, y in front)


def test_pareto_front_empty():
    assert pareto_front([]) == []


# ---------------------------------------------------------------------------
# DataFrame smoke test
# ---------------------------------------------------------------------------

def test_build_dataframe_no_exceptions(all_records):
    from analysis.summary_table import build_dataframe
    df = build_dataframe(all_records)
    assert not df.empty
    assert "Method" in df.columns
    assert "ε" in df.columns
    assert "MatchAccuracy" in df.columns
