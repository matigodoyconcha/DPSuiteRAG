"""Unit tests for DP-KSA (Tang et al., arXiv:2602.14374)."""

from __future__ import annotations

import numpy as np
import pytest

from statistics import NormalDist

from dp_methods.per_query.dpksa.keyword_selector import (
    build_token_histogram,
    find_best_k,
    gaps,
    padded_gap,
    ptr_threshold,
    sorted_counts_and_tokens,
    top_k_with_ptr,
)
from metrics.privacy_accounting_dpksa import calibrate_dpksa, compose_dpksa_to_dp


# ---------------------------------------------------------------------------
# Histogram + keyword selection
# ---------------------------------------------------------------------------

def test_histogram_counts_per_response_set() -> None:
    responses = ["Paris is the capital", "paris France", "London"]
    hist = build_token_histogram(responses, lowercase=True)
    assert hist["paris"] == 2  # appears in two responses
    assert hist["london"] == 1
    # A token repeated within one response is counted once (set membership).
    assert build_token_histogram(["paris paris paris"])["paris"] == 1


def test_sorted_and_gaps() -> None:
    counts, tokens = sorted_counts_and_tokens({"a": 5, "b": 3, "c": 3, "d": 1})
    assert counts == [5, 3, 3, 1]
    assert tokens[0] == "a"
    assert gaps(counts) == [2, 0, 2]


def test_find_best_k_picks_largest_gap() -> None:
    counts = [10, 9, 2, 1, 1]  # gaps d_k = [1, 7, 1, 0] -> largest at k=2
    k_hat, d_khat = find_best_k(
        counts, gumbel_scale=0.01, k_min=1, k_max=4, rng=np.random.default_rng(0)
    )
    assert 1 <= k_hat <= 4
    assert k_hat == 2
    assert d_khat == 7


def test_find_best_k_range_is_data_independent() -> None:
    # The candidate range must be the fixed public window regardless of how many
    # distinct tokens appear (no crash, no clamp to the data) — fixes C1/C2.
    rng = np.random.default_rng(0)
    for counts in ([], [5], [5, 5], [9, 1]):
        k_hat, d_khat = find_best_k(
            counts, gumbel_scale=1.0, k_min=2, k_max=5, rng=rng
        )
        assert 2 <= k_hat <= 5
        assert d_khat >= 0  # zero-padded gaps are non-negative


def test_padded_gap_zero_pads() -> None:
    assert padded_gap([10, 4], 1) == 6
    assert padded_gap([10, 4], 2) == 4   # H_(2) - H_(3)=0
    assert padded_gap([10, 4], 5) == 0   # beyond observed tokens
    assert padded_gap([], 1) == 0


def test_ptr_threshold_is_exact_quantile() -> None:
    sigma, delta_ptr = 1.3, 5e-5
    t = ptr_threshold(sigma, delta_ptr)
    # Exact (1 - delta_ptr) quantile of N(0, (2*sigma)^2).
    assert abs(NormalDist(0.0, 2.0 * sigma).cdf(t) - (1.0 - delta_ptr)) < 1e-9


def test_ptr_passes_on_large_gap() -> None:
    rng = np.random.default_rng(0)
    assert top_k_with_ptr(50, sigma=0.1, delta_ptr=5e-5, rng=rng) is True


def test_ptr_usually_fails_on_zero_gap() -> None:
    rng = np.random.default_rng(0)
    passes = [top_k_with_ptr(0, sigma=5.0, delta_ptr=5e-5, rng=rng) for _ in range(50)]
    assert sum(passes) < 50  # an unstable (small) gap should rarely pass


# ---------------------------------------------------------------------------
# Privacy accounting / calibration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", [1.0, 2.0, 3.0, 5.0, 8.0])
def test_calibration_is_sound(target: float) -> None:
    cal = calibrate_dpksa(target_epsilon=target, target_delta=1e-4)
    assert cal["achieved_epsilon"] <= target + 1e-6  # never exceeds the budget
    assert cal["gumbel_scale"] > 0.0
    assert cal["sigma"] > 0.0
    assert abs((cal["delta_ptr"] + cal["delta_conv"]) - 1e-4) < 1e-12


@pytest.mark.parametrize("coupling", [0.1, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("target", [1.0, 3.0, 8.0])
def test_calibration_sound_across_coupling(coupling: float, target: float) -> None:
    # DPKSA-05 ships ptr_noise_coupling=0.25; the (eps, delta) budget must hold for
    # any coupling (smaller sigma just reallocates noise to the Gumbel scale).
    cal = calibrate_dpksa(
        target_epsilon=target, target_delta=1e-4, ptr_noise_coupling=coupling
    )
    assert cal["achieved_epsilon"] <= target + 1e-9
    assert cal["sigma"] == pytest.approx(coupling * cal["gumbel_scale"])


def test_calibration_monotone_in_epsilon() -> None:
    c_small = calibrate_dpksa(target_epsilon=1.0, target_delta=1e-4)
    c_large = calibrate_dpksa(target_epsilon=8.0, target_delta=1e-4)
    # A larger epsilon budget allows less noise.
    assert c_large["gumbel_scale"] < c_small["gumbel_scale"]
    assert c_large["sigma"] < c_small["sigma"]


def test_compose_matches_calibration() -> None:
    cal = calibrate_dpksa(target_epsilon=3.0, target_delta=1e-4)
    eps, delta = compose_dpksa_to_dp(
        gumbel_scale=cal["gumbel_scale"],
        sigma=cal["sigma"],
        delta_ptr=cal["delta_ptr"],
        delta_conv=cal["delta_conv"],
    )
    assert eps <= 3.0 + 1e-6
    assert abs(delta - 1e-4) < 1e-12


@pytest.mark.parametrize("target", [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 20.0, 50.0])
@pytest.mark.parametrize("delta", [1e-6, 1e-4, 1e-2])
def test_calibration_sound_wide_grid(target: float, delta: float) -> None:
    cal = calibrate_dpksa(target_epsilon=target, target_delta=delta)
    assert cal["achieved_epsilon"] <= target + 1e-9
    eps, d = compose_dpksa_to_dp(
        gumbel_scale=cal["gumbel_scale"],
        sigma=cal["sigma"],
        delta_ptr=cal["delta_ptr"],
        delta_conv=cal["delta_conv"],
    )
    assert eps <= target + 1e-9          # composed epsilon never exceeds the budget
    assert abs(d - delta) < 1e-12


def test_histogram_neighbor_sensitivity() -> None:
    # Lemma KSA-1: replacing one of the N responses (top-N refill) moves each token
    # count by <= 1 and each gap by <= 2.
    base = ["paris france capital", "paris", "london uk", "rome italy"]
    neighbor = ["paris france capital", "berlin germany", "london uk", "rome italy"]
    h_base = build_token_histogram(base)
    h_neigh = build_token_histogram(neighbor)
    all_tokens = set(h_base) | set(h_neigh)
    assert max(abs(h_base.get(t, 0) - h_neigh.get(t, 0)) for t in all_tokens) <= 1
    counts_base, _ = sorted_counts_and_tokens(h_base)
    counts_neigh, _ = sorted_counts_and_tokens(h_neigh)
    k_hi = max(len(counts_base), len(counts_neigh))
    assert all(
        abs(padded_gap(counts_base, k) - padded_gap(counts_neigh, k)) <= 2
        for k in range(1, k_hi + 1)
    )


def test_ptr_false_release_rate_matches_delta() -> None:
    # Unstable gap (d_k <= 2): P(release) must equal delta_ptr (exact via inv_cdf).
    rng = np.random.default_rng(0)
    delta_ptr, sigma, n = 0.1, 1.0, 20000
    passes = sum(
        top_k_with_ptr(0, sigma=sigma, delta_ptr=delta_ptr, rng=rng) for _ in range(n)
    )
    assert abs(passes / n - delta_ptr) < 0.02


# ---------------------------------------------------------------------------
# End-to-end generator (needs a small real model)
# ---------------------------------------------------------------------------

def test_generator_end_to_end() -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    except Exception as exc:  # pragma: no cover - network/model availability
        pytest.skip(f"Could not load gpt2: {exc}")

    from dp_methods.per_query.dpksa.generator import DPKSAGenerator

    generator = DPKSAGenerator(
        model=model,
        tokenizer=tokenizer,
        target_epsilon=8.0,
        target_delta=1e-4,
        n_docs=4,
        max_answer_tokens=8,
        max_final_tokens=12,
        k_max=3,
        random_seed=42,
    )
    docs = [f"Paris is the capital of France. Note {i}." for i in range(4)]
    out = generator.generate_ksa("What is the capital of France?", docs)

    assert isinstance(out, str)
    stats = generator.last_run_stats
    assert stats["n_docs"] == 4
    assert "ptr_passed" in stats
    assert stats["achieved_epsilon"] <= 8.0 + 1e-6
