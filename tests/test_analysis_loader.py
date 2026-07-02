"""Smoke tests for the analysis result loader.

Loads the real results/ directory and checks that every CanonicalRecord
is well-formed, that per-method invariants hold, and that the summary
DataFrame can be built without errors.
"""
from __future__ import annotations

import math
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
    return load_all(RESULTS_DIR, recompute_vanilla=False)


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

def test_dpvoterag_full_epsilon_spent_populated(all_records):
    # Full dpvoterag runs include dp_spent → epsilon_spent must be populated.
    # Quick runs may have dp_spent=null in their JSON, so they are excluded here.
    dpvote_full = [
        r for r in all_records
        if r.method == "dpvoterag" and r.variant == "full"
    ]
    assert dpvote_full, "No full dpvoterag records found"
    for r in dpvote_full:
        assert r.epsilon_spent is not None, (
            f"Full dpvoterag record should have epsilon_spent populated: {r.source_file}"
        )


def test_dpvoterag_mia_auc_bleu_key_present(all_records):
    dpvote = [r for r in all_records if r.method == "dpvoterag"]
    for r in dpvote:
        # Key must be 'mia_auc_bleu', not the raw 'mia_auc'
        assert "mia_auc_bleu" in r.privacy, (
            f"Expected 'mia_auc_bleu' key in privacy dict, got {list(r.privacy.keys())} "
            f"in {r.source_file}"
        )
        assert "mia_auc" not in r.privacy, (
            f"Raw 'mia_auc' key should have been normalised away in {r.source_file}"
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
