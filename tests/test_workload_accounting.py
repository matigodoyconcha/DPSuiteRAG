"""Tests for the common workload accountant."""

from __future__ import annotations

import math

import pytest

from metrics.privacy_accounting_zcdp import zcdp_dp_synrag
from metrics.workload_accounting import (
    DEFAULT_DELTA,
    METHOD_REGIMES,
    Mechanism,
    epsilon_query_for_workload,
    epsilon_total,
    exponential_mechanism,
    gaussian,
    group_privacy,
    method_epsilon_total,
    pure_dp,
    zcdp,
)


# ---------------------------------------------------------------------------
# Mechanism curves
# ---------------------------------------------------------------------------

def test_gaussian_rdp_curve_is_exact():
    mech = gaussian("gen", sigma=2.0, sensitivity=1.0)
    for alpha in (2.0, 8.0, 32.0):
        assert mech.rdp(alpha) == pytest.approx(alpha / (2.0 * 4.0))


def test_gaussian_has_no_pure_epsilon():
    assert math.isinf(gaussian("gen", sigma=1.0).pure_epsilon)


def test_pure_dp_curve_never_exceeds_its_own_epsilon():
    mech = pure_dp("svt", epsilon=0.5)
    for alpha in (1.5, 2.0, 100.0, 256.0):
        assert mech.rdp(alpha) <= 0.5 + 1e-12


def test_exponential_curve_is_tighter_than_the_generic_pure_bound():
    # The sinh bound is why composing many token-level votes stays affordable.
    eps = 0.5
    em = exponential_mechanism("vote", eps, count=50)
    generic = pure_dp("vote", eps, count=50)
    alpha = 16.0
    assert em.rdp(alpha) < generic.rdp(alpha)


def test_count_scales_the_curve_and_the_additive_delta():
    single = gaussian("g", sigma=1.0, delta_add=1e-9)
    many = gaussian("g", sigma=1.0, count=10, delta_add=1e-9)
    assert many.rdp(4.0) == pytest.approx(10 * single.rdp(4.0))
    assert many.total_delta_add == pytest.approx(1e-8)


def test_rejects_unknown_kind_and_bad_parameters():
    with pytest.raises(ValueError):
        Mechanism(name="x", kind="laplace", parameter=1.0)
    with pytest.raises(ValueError):
        Mechanism(name="x", kind="pure", parameter=0.0)
    with pytest.raises(ValueError):
        Mechanism(name="x", kind="pure", parameter=1.0, delta_add=1.5)


# ---------------------------------------------------------------------------
# Workload composition
# ---------------------------------------------------------------------------

def test_epsilon_grows_with_the_workload():
    mechs = [gaussian("gen", sigma=4.0, count=50)]
    values = [epsilon_total(mechs, queries=q).epsilon for q in (1, 5, 10, 50, 100)]
    assert values == sorted(values)
    assert values[0] < values[-1]


def test_epsilon_grows_as_noise_shrinks():
    quiet = epsilon_total([gaussian("gen", sigma=8.0)], queries=100).epsilon
    loud = epsilon_total([gaussian("gen", sigma=2.0)], queries=100).epsilon
    assert quiet < loud


def test_rdp_composition_beats_basic_composition_at_scale():
    # The asymmetry the benchmark corrects: composing per-query (eps, delta) boxes
    # naively is far looser than composing the mechanisms themselves.
    result = epsilon_total([gaussian("gen", sigma=4.0, count=50)], queries=100)
    assert result.epsilon < result.epsilon_basic


def test_pure_dp_workload_stays_pure_and_additive():
    # A pure-DP query must not be routed through a conversion that inflates it.
    result = epsilon_total([pure_dp("filter", 0.5)], queries=20)
    assert result.epsilon == pytest.approx(10.0)
    assert result.delta_additive == 0.0
    assert math.isinf(result.best_alpha)


def test_pure_dp_is_not_degraded_by_the_common_delta():
    # eps0-DP implies (eps0, delta)-DP for every delta; a zCDP round trip would turn
    # eps0 = 10 into roughly 98, which is the artefact this guards against.
    result = epsilon_total([pure_dp("murag", 10.0)], queries=100, regime="corpus_total")
    assert result.epsilon == pytest.approx(10.0)


def test_corpus_total_and_offline_regimes_are_flat_in_Q():
    for regime in ("corpus_total", "offline"):
        values = [
            epsilon_total([pure_dp("m", 3.0)], queries=q, regime=regime).epsilon
            for q in (1, 10, 100)
        ]
        assert values[0] == pytest.approx(values[-1])


def test_additive_delta_is_subtracted_from_the_target():
    mechs = [exponential_mechanism("vote", 0.5, count=50, delta_add=1e-9)]
    result = epsilon_total(mechs, queries=100, delta=DEFAULT_DELTA)
    assert result.delta_additive == pytest.approx(100 * 50 * 1e-9)
    assert result.delta_conversion == pytest.approx(DEFAULT_DELTA - result.delta_additive)


def test_additive_delta_over_budget_fails_loudly():
    # The per-token delta of the vote-based configs: 1e-6 over T=50 and Q=100 is 5e-3,
    # five hundred times a 1e-5 target. It must not pass silently.
    mechs = [exponential_mechanism("vote", 2.0, count=50, delta_add=1e-6)]
    with pytest.raises(ValueError, match="exhausts the target delta"):
        epsilon_total(mechs, queries=100, delta=DEFAULT_DELTA)


def test_components_are_reported_per_mechanism():
    mechs = [exponential_mechanism("retrieval", 0.5), gaussian("gen", sigma=4.0, count=50)]
    result = epsilon_total(mechs, queries=10)
    assert set(result.components) == {"retrieval", "gen"}


def test_rejects_invalid_workload_and_delta():
    with pytest.raises(ValueError):
        epsilon_total([pure_dp("m", 1.0)], queries=0)
    with pytest.raises(ValueError):
        epsilon_total([pure_dp("m", 1.0)], queries=1, delta=0.0)
    with pytest.raises(ValueError):
        epsilon_total([pure_dp("m", 1.0)], queries=1, regime="whatever")


# ---------------------------------------------------------------------------
# Inverse map
# ---------------------------------------------------------------------------

def _build(eps_query: float):
    return [gaussian("gen", sigma=1.0 / eps_query, count=50)]


def test_inversion_round_trips():
    for target in (1.0, 3.0, 10.0, 30.0):
        eps_q = epsilon_query_for_workload(_build, target_epsilon=target, queries=100)
        achieved = epsilon_total(_build(eps_q), queries=100).epsilon
        assert achieved == pytest.approx(target, rel=1e-3)


def test_inversion_returns_from_the_feasible_side():
    target = 10.0
    eps_q = epsilon_query_for_workload(_build, target_epsilon=target, queries=100)
    assert epsilon_total(_build(eps_q), queries=100).epsilon <= target + 1e-9


def test_inversion_tightens_as_the_workload_grows():
    budgets = [
        epsilon_query_for_workload(_build, target_epsilon=10.0, queries=q)
        for q in (1, 5, 25, 100)
    ]
    assert budgets == sorted(budgets, reverse=True)


def test_inversion_beats_the_naive_split():
    # Dividing the workload budget by Q is what the papers do to their rivals when
    # they compose a per-query (eps, delta) box naively. Composing the mechanism
    # itself must hand the method a strictly larger per-query budget for the same
    # workload guarantee — that gap is the accounting asymmetry the benchmark fixes.
    def build_per_query(eps_query: float):
        return [exponential_mechanism("query", eps_query)]

    target, queries = 10.0, 100
    eps_q = epsilon_query_for_workload(build_per_query, target_epsilon=target, queries=queries)
    assert eps_q > target / queries


def test_inversion_reports_an_infeasible_target():
    def build_with_fixed_delta(eps_query: float):
        return [exponential_mechanism("vote", eps_query, count=50, delta_add=1e-6)]

    with pytest.raises(ValueError, match="binding constraint"):
        epsilon_query_for_workload(
            build_with_fixed_delta, target_epsilon=10.0, queries=100
        )


# ---------------------------------------------------------------------------
# Per-method inventories
# ---------------------------------------------------------------------------

def test_zcdp_matches_the_gaussian_curve_it_stands_for():
    rho = 0.75
    assert zcdp("build", rho).rdp(8.0) == pytest.approx(rho * 8.0)


def test_zcdp_conversion_is_tighter_than_the_closed_form():
    # rho + 2*sqrt(rho*ln(1/delta)) is the bound DP-SynRAG prints; optimising over the
    # Renyi order can only improve on it.
    rho, delta = 1.5, DEFAULT_DELTA
    closed_form = rho + math.sqrt(4.0 * rho * math.log(1.0 / delta))
    shared = epsilon_total([zcdp("build", rho)], queries=1, delta=delta, regime="offline")
    assert shared.epsilon <= closed_form


def test_dp_synrag_joins_the_axis_from_its_own_accountant():
    native = zcdp_dp_synrag(
        K=5, sigma_h=1.0, L=5, epsilon_theta_s=0.4, sigma_mu=1.0, T=30, c=0.1057,
        tau=1.0, delta=1e-3,
    )
    shared = method_epsilon_total("dp_synrag", rho=native.rho, queries=100)
    assert shared.epsilon > 0
    assert shared.regime == "offline"


def test_the_two_flat_regimes_do_not_grow_with_the_workload():
    for method, params in (
        ("private_rag", {"m_retrievals": 1, "epsilon_q": 10.0}),
        ("dp_synrag", {"rho": 1.5}),
    ):
        values = [
            method_epsilon_total(method, queries=q, **params).epsilon for q in (1, 10, 100)
        ]
        assert values[0] == pytest.approx(values[-1])
        assert METHOD_REGIMES[method] != "per_query"


def test_the_four_per_query_methods_grow_with_the_workload():
    cases = {
        "dp_rag": {"epsilon_retrieval": 0.5, "token_epsilon": 0.05, "num_tokens": 70},
        "dpvoterag": {"epsilon_token": 0.05, "num_tokens": 50, "delta_token": 1e-11},
        "dpsparsevoterag": {
            "epsilon_token": 0.05, "num_tokens": 20, "delta_token": 1e-11,
            "sparse_epsilon": 0.05, "n_epochs": 2,
        },
        "dpksa": {"gumbel_scale": 8.0, "sigma": 2.0, "delta_ptr": 1e-9},
    }
    for method, params in cases.items():
        assert METHOD_REGIMES[method] == "per_query"
        low = method_epsilon_total(method, queries=1, **params).epsilon
        high = method_epsilon_total(method, queries=100, **params).epsilon
        assert high > low


def test_private_rag_reports_M_times_eps_q_unchanged():
    result = method_epsilon_total("private_rag", m_retrievals=1, epsilon_q=10.0, queries=100)
    assert result.epsilon == pytest.approx(10.0)


def test_dpksa_ptr_delta_is_charged_once_per_query():
    result = method_epsilon_total(
        "dpksa", gumbel_scale=8.0, sigma=2.0, delta_ptr=1e-9, queries=100
    )
    assert result.delta_additive == pytest.approx(100 * 1e-9)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        method_epsilon_total("dp_nonexistent", queries=10)


def test_every_registered_method_has_a_regime():
    from metrics.workload_accounting import METHOD_MECHANISMS

    assert set(METHOD_MECHANISMS) == set(METHOD_REGIMES)


# ---------------------------------------------------------------------------
# Group privacy
# ---------------------------------------------------------------------------

def test_group_privacy_is_the_identity_for_one_unit():
    assert group_privacy(1.0, 1e-5, 1) == (1.0, 1e-5)


def test_chunking_blows_up_the_delta():
    # The argument for indexing whole documents: delta grows exponentially in the
    # number of chunks, so a chunk-level guarantee does not survive the lift.
    _, delta_doc = group_privacy(1.0, 1e-5, 5)
    assert delta_doc > 1e-3


def test_group_privacy_delta_is_capped_at_one():
    _, delta_doc = group_privacy(5.0, 1e-2, 20)
    assert delta_doc == 1.0
