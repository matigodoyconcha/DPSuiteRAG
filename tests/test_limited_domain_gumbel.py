"""Unit tests for the Durfee-Rogers Limited Domain Gumbel mechanism.

These tests verify the formal correctness of the LD-Gumbel implementation,
including success paths and formal failure modes.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from dp_methods.dpvoterag.generator import VoteRAGGenerator, NotFoundLDTop1


class MockModel:
	"""Minimal mock model for testing VoteRAGGenerator."""

	def __init__(self):
		self.device = "cpu"

	def parameters(self):
		"""Return a minimal parameter for device detection."""
		import torch
		return [torch.tensor(0.0)]


class MockTokenizer:
	"""Minimal mock tokenizer for testing."""

	def __init__(self):
		self.eos_token_id = 50256
		self.vocab_size = 50257  # GPT-2 vocab size for testing

	def decode(self, tokens, skip_special_tokens=False):
		return " ".join(str(t) for t in tokens)


@pytest.fixture
def generator():
	"""Create a VoteRAGGenerator instance with mock model/tokenizer."""
	model = MockModel()
	tokenizer = MockTokenizer()
	return VoteRAGGenerator(
		model=model,
		tokenizer=tokenizer,
		m=10,
		k=1,
		epsilon_token=1.0,
		delta_token=1e-5,
		epsilon_total=100.0,
		delta_total=1e-4,
		k_bar=10,
		random_seed=42,
	)


class TestLimitedDomainSuccessCase:
	"""Test cases where the LD mechanism should succeed."""

	def test_skewed_histogram_succeeds(self, generator):
		"""Test that a skewed histogram (one dominant token) succeeds.
		
		With a clear winner (high count) and reasonable privacy parameters,
		the mechanism should succeed with v_max >= v_perp.
		"""
		histogram = {
			100: 8,   # Token 100 appears 8 times (dominant)
			101: 1,   # Token 101 appears 1 time
			102: 1,   # Token 102 appears 1 time
		}
		
		# This should succeed without raising NotFoundLDTop1
		# Using larger epsilon (10.0) and larger delta (1e-3) to make success likely
		selected_token = generator._limited_domain_or_fail(
			histogram=histogram,
			eps=10.0,
			delta=1e-3,
			k_bar=3,
			sens=2.0,
		)
		assert selected_token in [100, 101, 102], "Selected token must be from histogram."

	def test_uniform_histogram_with_large_delta(self, generator):
		"""Test that even a uniform histogram succeeds with large delta.
		
		With delta close to 1 (weak privacy), the threshold h_perp becomes small,
		making success more likely even with uniform distributions.
		"""
		histogram = {
			200: 2,
			201: 2,
			202: 2,
			203: 2,
			204: 2,
		}
		
		# Large delta weakens the privacy-paying term
		selected_token = generator._limited_domain_or_fail(
			histogram=histogram,
			eps=1.0,
			delta=0.99,  # Very weak privacy
			k_bar=5,
			sens=2.0,
		)
		assert selected_token in [200, 201, 202, 203, 204]

	def test_k_bar_respects_histogram_size(self, generator):
		"""Test that k_bar is clamped to histogram size."""
		histogram = {300: 5, 301: 3}
		
		# Request k_bar=100 but histogram only has 2 unique tokens
		# Using large epsilon for successful runs
		selected_token = generator._limited_domain_or_fail(
			histogram=histogram,
			eps=10.0,
			delta=1e-3,
			k_bar=100,  # Larger than histogram
			sens=2.0,
		)
		assert selected_token in [300, 301]


class TestLimitedDomainFailureMode:
	"""Test cases where the LD mechanism should formally fail."""

	def test_uniform_histogram_fails_with_tiny_delta(self, generator):
		"""Test that uniform histogram with tiny delta causes mechanism failure.
		
		With delta very small (strong privacy) and uniform counts, the threshold
		h_perp becomes large, and it's likely that all Gumbel-perturbed counts
		fall below v_perp, causing a formal privacy failure.
		
		Note: Due to randomness, this is a probabilistic test. We run it multiple
		times and expect at least one failure.
		"""
		histogram = {
			400: 1,
			401: 1,
			402: 1,
			403: 1,
			404: 1,
		}
		
		failure_count = 0
		n_trials = 20
		
		for _ in range(n_trials):
			try:
				generator._limited_domain_or_fail(
					histogram=histogram,
					eps=0.1,  # Very small epsilon (strong privacy)
					delta=1e-10,  # Extremely small delta (very strong privacy)
					k_bar=5,
					sens=2.0,
				)
			except NotFoundLDTop1:
				failure_count += 1
		
		# Expect at least one failure in the trials (probabilistic)
		assert failure_count > 0, (
			f"Mechanism should fail with uniform histogram and tiny delta; "
			f"failed {failure_count}/{n_trials} times."
		)

	def test_empty_histogram_raises(self, generator):
		"""Test that empty histogram raises NotFoundLDTop1."""
		with pytest.raises(NotFoundLDTop1, match="Empty histogram"):
			generator._limited_domain_or_fail(
				histogram={},
				eps=1.0,
				delta=1e-5,
				k_bar=5,
				sens=2.0,
			)


class TestLimitedDomainParameterValidation:
	"""Test parameter validation and edge cases."""

	def test_invalid_epsilon_negative(self, generator):
		"""Test that negative epsilon raises ValueError."""
		histogram = {500: 3, 501: 2}
		
		with pytest.raises(ValueError, match="Invalid privacy parameters"):
			generator._limited_domain_or_fail(
				histogram=histogram,
				eps=-1.0,
				delta=1e-5,
				k_bar=2,
				sens=2.0,
			)

	def test_invalid_delta_zero(self, generator):
		"""Test that delta=0 raises ValueError."""
		histogram = {500: 3, 501: 2}
		
		with pytest.raises(ValueError, match="Invalid privacy parameters"):
			generator._limited_domain_or_fail(
				histogram=histogram,
				eps=1.0,
				delta=0.0,
				k_bar=2,
				sens=2.0,
			)

	def test_invalid_delta_too_large(self, generator):
		"""Test that delta >= 1 raises ValueError."""
		histogram = {500: 3, 501: 2}
		
		with pytest.raises(ValueError, match="Invalid privacy parameters"):
			generator._limited_domain_or_fail(
				histogram=histogram,
				eps=1.0,
				delta=1.0,
				k_bar=2,
				sens=2.0,
			)

	def test_k_bar_zero_raises(self, generator):
		"""Test that k_bar=0 raises ValueError."""
		histogram = {500: 3, 501: 2}
		
		with pytest.raises(ValueError, match="k_bar_real must be >= 1"):
			generator._limited_domain_or_fail(
				histogram=histogram,
				eps=1.0,
				delta=1e-5,
				k_bar=0,
				sens=2.0,
			)


class TestLimitedDomainSensitivityParameter:
	"""Test the sensitivity parameter (default sens=2 for l0-neighboring)."""

	def test_sensitivity_two_default(self, generator):
		"""Test that default sensitivity is 2 (l0-neighboring)."""
		histogram = {600: 5, 601: 3, 602: 2}
		
		# Should use sens=2 by default and succeed
		# Using large epsilon for success
		selected_token = generator._limited_domain_or_fail(
			histogram=histogram,
			eps=10.0,
			delta=1e-3,
			k_bar=3,
			# sens not specified, should default to 2.0
		)
		assert selected_token in [600, 601, 602]

	def test_sensitivity_one_alternative(self, generator):
		"""Test that sens=1 (l1-neighboring) can be specified."""
		histogram = {700: 4, 701: 3, 702: 2, 703: 1}
		
		# With sens=1, the threshold h_perp includes log(min(1, k_bar, dim-k_bar))
		# This tests that the parameter is properly threaded
		# Using large epsilon for success
		selected_token = generator._limited_domain_or_fail(
			histogram=histogram,
			eps=10.0,
			delta=1e-3,
			k_bar=4,
			sens=1.0,
		)
		assert selected_token in [700, 701, 702, 703]


class TestNotFoundLDTop1Exception:
	"""Test the NotFoundLDTop1 exception class."""

	def test_exception_is_catchable(self):
		"""Test that NotFoundLDTop1 can be caught and contains message."""
		try:
			raise NotFoundLDTop1("Test message")
		except NotFoundLDTop1 as e:
			assert "Test message" in str(e)

	def test_exception_inherits_from_exception(self):
		"""Test that NotFoundLDTop1 is a proper Exception."""
		assert issubclass(NotFoundLDTop1, Exception)


if __name__ == "__main__":
	pytest.main([__file__, "-v"])
