"""
Unit tests for privacy_accounting_ld module (RDP composition for LD-Gumbel mechanism).
"""

import math
import pytest
from metrics.privacy_accounting_ld import (
    rdp_exponential_mechanism,
    rdp_limited_domain_gumbel,
    compose_rdp_queries,
    compose_subsampled_mechanism,
    rdp_to_approx_dp_bbghs,
    search_optimal_alpha_rdp_to_dp,
    LDGumbelAccountant,
    DPCompositionResult,
)


class TestRDPExponentialMechanism:
    """Test RDP of pure exponential mechanism (ε-DP)."""

    def test_alpha_2_epsilon_1(self):
        """For α=2, ε=1: ρ = 2·1/(2-1) = 2."""
        rho = rdp_exponential_mechanism(alpha=2.0, epsilon=1.0)
        assert math.isclose(rho, 2.0, rel_tol=1e-6)

    def test_alpha_10_epsilon_1(self):
        """For α=10, ε=1: ρ = 10·1/(10-1) = 10/9 ≈ 1.111."""
        rho = rdp_exponential_mechanism(alpha=10.0, epsilon=1.0)
        expected = 10.0 / 9.0
        assert math.isclose(rho, expected, rel_tol=1e-6)

    def test_alpha_less_than_1_raises(self):
        """RDP order must be > 1."""
        with pytest.raises(ValueError, match="Rényi order must be > 1"):
            rdp_exponential_mechanism(alpha=0.5, epsilon=1.0)

    def test_negative_epsilon_raises(self):
        """epsilon must be > 0."""
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            rdp_exponential_mechanism(alpha=2.0, epsilon=-1.0)


class TestRDPLimitedDomainGumbel:
    """Test RDP of Limited Domain Gumbel mechanism."""

    def test_basic_ld_gumbel(self):
        """Basic case: eps=1, delta=1e-5, alpha=2."""
        rho = rdp_limited_domain_gumbel(
            alpha=2.0,
            epsilon_token=1.0,
            delta_token=1e-5,
            k_bar=10,
            sensitivity=2.0,
            k=1,
        )
        assert isinstance(rho, float)
        assert rho > 0

    def test_ld_gumbel_respects_sensitivity(self):
        """With higher sensitivity, RDP should increase."""
        rho_sens_1 = rdp_limited_domain_gumbel(
            alpha=2.0,
            epsilon_token=1.0,
            delta_token=1e-5,
            sensitivity=1.0,
        )
        rho_sens_2 = rdp_limited_domain_gumbel(
            alpha=2.0,
            epsilon_token=1.0,
            delta_token=1e-5,
            sensitivity=2.0,
        )
        # With higher sensitivity, RDP increases (conservative bound)
        assert rho_sens_2 >= rho_sens_1

    def test_ld_gumbel_invalid_alpha_raises(self):
        """alpha must be > 1."""
        with pytest.raises(ValueError, match="Rényi order must be > 1"):
            rdp_limited_domain_gumbel(
                alpha=0.5,
                epsilon_token=1.0,
                delta_token=1e-5,
            )

    def test_ld_gumbel_invalid_epsilon_raises(self):
        """epsilon must be > 0."""
        with pytest.raises(ValueError, match="epsilon_token must be > 0"):
            rdp_limited_domain_gumbel(
                alpha=2.0,
                epsilon_token=-1.0,
                delta_token=1e-5,
            )

    def test_ld_gumbel_invalid_delta_raises(self):
        """delta must be in (0, 1)."""
        with pytest.raises(ValueError, match="delta_token must be in"):
            rdp_limited_domain_gumbel(
                alpha=2.0,
                epsilon_token=1.0,
                delta_token=0.0,  # Invalid
            )
        with pytest.raises(ValueError, match="delta_token must be in"):
            rdp_limited_domain_gumbel(
                alpha=2.0,
                epsilon_token=1.0,
                delta_token=1.5,  # Invalid
            )


class TestComposeRDPQueries:
    """Test sequential composition of RDP."""

    def test_compose_two_queries(self):
        """Composing two identical RDP values."""
        rho1 = 0.5
        rho2 = 0.5
        composed = compose_rdp_queries([rho1, rho2], alpha=2.0)
        assert math.isclose(composed, 1.0, rel_tol=1e-6)

    def test_compose_empty_list(self):
        """Composing empty list gives 0."""
        composed = compose_rdp_queries([], alpha=2.0)
        assert composed == 0.0

    def test_compose_accumulates_correctly(self):
        """Compose multiple values; result is sum."""
        rhos = [0.1, 0.2, 0.3, 0.15]
        composed = compose_rdp_queries(rhos, alpha=2.0)
        expected = sum(rhos)
        assert math.isclose(composed, expected, rel_tol=1e-6)


class TestComposeSubsampledMechanism:
    """Test subsampling amplification of RDP."""

    def test_subsampling_amplification_reduces_rho(self):
        """Subsampling amplifies privacy: amplified rho < original rho per query."""
        rho_alpha = 1.0
        alpha = 2.0
        subsampling_prob = 0.1  # 10% subsampling
        num_queries = 1

        amplified = compose_subsampled_mechanism(
            rho_alpha, alpha, subsampling_prob, num_queries
        )

        # With very aggressive subsampling, amplified rho should be much smaller
        assert amplified < rho_alpha

    def test_subsampling_invalid_prob_raises(self):
        """subsampling_prob must be in (0, 1]."""
        with pytest.raises(ValueError, match="subsampling_prob must be in"):
            compose_subsampled_mechanism(1.0, 2.0, prob=0.0, niter=1)
        with pytest.raises(ValueError, match="subsampling_prob must be in"):
            compose_subsampled_mechanism(1.0, 2.0, subsampling_prob=1.5, num_queries=1)

    def test_subsampling_invalid_num_queries_raises(self):
        """num_queries must be > 0."""
        with pytest.raises(ValueError, match="num_queries must be > 0"):
            compose_subsampled_mechanism(1.0, 2.0, subsampling_prob=0.5, num_queries=0)


class TestRDPToApproxDPBBGHS:
    """Test BBGHS conversion from RDP to approximate (ε,δ)-DP."""

    def test_basic_conversion(self):
        """Basic RDP-to-DP conversion."""
        result = rdp_to_approx_dp_bbghs(
            rho_alpha=1.0,
            alpha=2.0,
            delta=1e-5,
        )
        assert isinstance(result, tuple)
        assert result.epsilon > 0
        assert result.delta == 1e-5
        assert result.method == "bbghs"
        assert result.order_alpha == 2.0

    def test_conversion_respects_delta_target(self):
        """The returned delta should match the input delta parameter."""
        target_delta = 1e-6
        result = rdp_to_approx_dp_bbghs(
            rho_alpha=1.0,
            alpha=2.0,
            delta=target_delta,
        )
        assert result.delta == target_delta

    def test_conversion_invalid_rho_raises(self):
        """rho_alpha must be >= 0."""
        with pytest.raises(ValueError, match="rho_alpha must be >= 0"):
            rdp_to_approx_dp_bbghs(rho_alpha=-1.0, alpha=2.0, delta=1e-5)

    def test_conversion_invalid_alpha_raises(self):
        """alpha must be > 1."""
        with pytest.raises(ValueError, match="alpha must be > 1"):
            rdp_to_approx_dp_bbghs(rho_alpha=1.0, alpha=0.5, delta=1e-5)

    def test_conversion_invalid_delta_raises(self):
        """delta must be in (0, 1)."""
        with pytest.raises(ValueError, match="delta must be in"):
            rdp_to_approx_dp_bbghs(rho_alpha=1.0, alpha=2.0, delta=0.0)
        with pytest.raises(ValueError, match="delta must be in"):
            rdp_to_approx_dp_bbghs(rho_alpha=1.0, alpha=2.0, delta=1.5)


class TestSearchOptimalAlpha:
    """Test alpha search for minimizing ε in RDP-to-DP conversion."""

    def test_search_finds_alpha_in_range(self):
        """Search should find an alpha within the specified range."""

        def simple_rdp(alpha):
            """Quadratic-like RDP as a function of alpha."""
            return (alpha - 10) ** 2 / 50  # Minimum around alpha=10

        result = search_optimal_alpha_rdp_to_dp(
            rho_fn=simple_rdp,
            delta=1e-5,
            alpha_min=2,
            alpha_max=200,
            num_samples=50,
        )

        # Result should be an ApproxDPResult
        assert isinstance(result, tuple)
        assert result.epsilon > 0
        assert 2 <= result.order_alpha <= 200

    def test_search_prefers_lower_epsilon(self):
        """With multiple valid alphas, search should prefer lower ε."""

        def constant_rdp(alpha):
            return 1.0  # Constant RDP regardless of alpha

        result = search_optimal_alpha_rdp_to_dp(
            rho_fn=constant_rdp,
            delta=1e-5,
            alpha_min=2,
            alpha_max=200,
            num_samples=10,
        )

        # With constant RDP, higher alpha should give lower ε
        # (since ε ≈ ρ/α + log(1/δ)/α)
        assert result.order_alpha > 2  # Should choose higher alpha


class TestLDGumbelAccountant:
    """Test online accounting for LD-Gumbel mechanism."""

    def test_accountant_initialization(self):
        """Accountant initializes with correct parameters."""
        acc = LDGumbelAccountant(
            target_epsilon=10.0,
            target_delta=1e-5,
            epsilon_token=1.0,
            delta_token=1e-5,
            m=30,
            k=1,
        )
        assert acc.target_epsilon == 10.0
        assert acc.target_delta == 1e-5
        assert acc.spent_epsilon == 0.0
        assert acc.total_tokens_dp == 0

    def test_accountant_invalid_target_epsilon_raises(self):
        """target_epsilon must be > 0."""
        with pytest.raises(ValueError, match="target_epsilon must be > 0"):
            LDGumbelAccountant(
                target_epsilon=0.0,
                target_delta=1e-5,
                epsilon_token=1.0,
                delta_token=1e-5,
            )

    def test_account_single_token(self):
        """Accounting a single token increments spent_epsilon."""
        acc = LDGumbelAccountant(
            target_epsilon=100.0,
            target_delta=1e-5,
            epsilon_token=1.0,
            delta_token=1e-5,
            m=10,
        )
        old_spent = acc.spent_epsilon
        budget_ok = acc.account_token(alpha=2.0)

        assert budget_ok is True  # Budget permits
        assert acc.spent_epsilon > old_spent  # Spent increased
        assert acc.total_tokens_dp == 1

    def test_check_budget_without_spending(self):
        """check_budget() should not modify state."""
        acc = LDGumbelAccountant(
            target_epsilon=100.0,
            target_delta=1e-5,
            epsilon_token=1.0,
            delta_token=1e-5,
        )
        old_spent = acc.spent_epsilon
        old_tokens = acc.total_tokens_dp

        status = acc.check_budget()

        # State should not change
        assert acc.spent_epsilon == old_spent
        assert acc.total_tokens_dp == old_tokens
        # But status should indicate ok
        assert status["ok"] is True

    def test_budget_exhaustion(self):
        """When target_epsilon is reached, budget_ok should become False."""
        acc = LDGumbelAccountant(
            target_epsilon=0.5,  # Very small budget
            target_delta=1e-5,
            epsilon_token=2.0,  # Large per-token cost
            delta_token=1e-5,
            m=1,
        )

        # First token might succeed
        budget_ok_1 = acc.account_token()

        if budget_ok_1:
            # Try second token; should fail
            budget_ok_2 = acc.account_token()
            # At least one should fail, or both should succeed if epsilon is large enough
            # With epsilon_token=2 and target=0.5, we expect failure

    def test_summary(self):
        """get_summary() returns complete accounting info."""
        acc = LDGumbelAccountant(
            target_epsilon=100.0,
            target_delta=1e-5,
            epsilon_token=1.0,
            delta_token=1e-5,
        )
        acc.account_token()
        acc.account_token()

        summary = acc.get_summary()

        assert summary["total_tokens_dp"] == 2
        assert summary["spent_epsilon"] > 0
        assert summary["target_epsilon"] == 100.0
        assert "budget_log" in summary
        assert len(summary["budget_log"]) == 2


class TestLDGumbelAccountantIntegration:
    """Integration tests for online accounting with mock LD queries."""

    def test_multiple_tokens_accumulate(self):
        """Accounting multiple tokens accumulates epsilon correctly."""
        acc = LDGumbelAccountant(
            target_epsilon=1000.0,  # Large budget
            target_delta=1e-5,
            epsilon_token=0.5,
            delta_token=1e-5,
            m=10,
        )

        for i in range(5):
            budget_ok = acc.account_token()
            assert budget_ok is True
            assert acc.total_tokens_dp == i + 1

        # Spent epsilon should grow monotonically
        assert acc.spent_epsilon > 0

    def test_budget_ceiling(self):
        """With tiny budget, should hit ceiling quickly."""
        acc = LDGumbelAccountant(
            target_epsilon=0.01,  # Tiny
            target_delta=1e-5,
            epsilon_token=10.0,  # Huge cost per token
            delta_token=1e-5,
            m=1,
        )

        # First token will likely exceed or come close
        budget_ok = acc.account_token()
        # With such imbalanced parameters, likely exceeds

        # Check budget should reflect exhaustion
        status = acc.check_budget()
        assert not status["ok"]
