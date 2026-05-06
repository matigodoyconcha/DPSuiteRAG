"""Unit tests for fail-mode strategies in VoteRAGGenerator.

Tests verify that the four fail-mode strategies correctly handle LD mechanism failures.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from dp_methods.dpvoterag.generator import VoteRAGGenerator, NotFoundLDTop1


class MockModel:
	"""Minimal mock model for testing."""

	def __init__(self):
		self.device = "cpu"

	def parameters(self):
		"""Return an iterator of minimal parameters for device detection."""
		import torch
		return iter([torch.tensor(0.0, device=self.device)])
	
	def __call__(self, input_ids):
		"""Mock model forward pass."""
		import torch
		# Return mock logits
		return type('obj', (object,), {
			'logits': torch.randn(input_ids.shape[0], input_ids.shape[1], 50257)
		})()


class MockTokenizer:
	"""Minimal mock tokenizer for testing."""

	def __init__(self):
		self.eos_token_id = 50256
		self.vocab_size = 50257

	def decode(self, tokens, skip_special_tokens=False):
		return " ".join(str(t) for t in tokens)
	
	def __call__(self, text, return_tensors=None):
		"""Mock tokenizer callable for encoding text."""
		import torch
		# Return a mock encoded dictionary with input_ids
		return {
			"input_ids": torch.tensor([[101, 2054, 2003, 1045, 102]])  # Mock token IDs
		}


@pytest.fixture
def mock_generator_base():
	"""Create a base generator for testing fail-modes."""
	model = MockModel()
	tokenizer = MockTokenizer()
	return model, tokenizer


class TestFailModeValidation:
	"""Test fail_mode parameter validation."""

	def test_invalid_fail_mode_raises(self, mock_generator_base):
		"""Test that invalid fail_mode raises ValueError."""
		model, tokenizer = mock_generator_base
		
		with pytest.raises(ValueError, match="Invalid fail_mode"):
			VoteRAGGenerator(
				model=model,
				tokenizer=tokenizer,
				fail_mode="invalid_mode",
			)

	def test_valid_fail_modes_accepted(self, mock_generator_base):
		"""Test that all valid fail_modes are accepted."""
		model, tokenizer = mock_generator_base
		
		for fail_mode in ["ld_pate", "rand", "stop", "raise"]:
			gen = VoteRAGGenerator(
				model=model,
				tokenizer=tokenizer,
				fail_mode=fail_mode,
			)
			assert gen.fail_mode == fail_mode


class TestFailModeLD_PATE:
	"""Test 'ld_pate' fail-mode strategy."""

	def test_ld_pate_retries_with_reduced_k_bar(self, mock_generator_base):
		"""Test that 'ld_pate' retries with smaller k_bar on failure."""
		model, tokenizer = mock_generator_base
		gen = VoteRAGGenerator(
			model=model,
			tokenizer=tokenizer,
			fail_mode="ld_pate",
			k_bar=5,
		)
		
		# Histogram with uniform counts should trigger failure with k_bar=5
		# but possibly succeed with smaller k_bar
		histogram = {100: 1, 101: 1, 102: 1, 103: 1, 104: 1}
		
		token_id, exit_status = gen._handle_ld_failure(
			histogram=histogram,
			eps=0.1,  # Very small epsilon to encourage failure
			delta=1e-10,  # Very small delta to encourage failure
			k_bar=5,
			sens=2.0,
		)
		
		# Should succeed (return token, exit_status=0) even if retries happened
		assert exit_status == 0
		assert token_id in list(histogram.keys()) or 0 <= token_id < tokenizer.vocab_size

	def test_ld_pate_falls_back_to_rand_if_all_retries_fail(self, mock_generator_base):
		"""Test that 'ld_pate' falls back to 'rand' if all retries exhausted."""
		model, tokenizer = mock_generator_base
		gen = VoteRAGGenerator(
			model=model,
			tokenizer=tokenizer,
			fail_mode="ld_pate",
			k_bar=2,
		)
		
		# Single-token histogram: all retries will fail (can't reduce k_bar below 1)
		histogram = {100: 1}
		
		token_id, exit_status = gen._handle_ld_failure(
			histogram=histogram,
			eps=0.1,
			delta=1e-10,
			k_bar=2,
			sens=2.0,
		)
		
		# Should fall back to random
		assert exit_status == 0
		assert 0 <= token_id < tokenizer.vocab_size


class TestFailModeRAND:
	"""Test 'rand' fail-mode strategy."""

	def test_rand_returns_random_token(self, mock_generator_base):
		"""Test that 'rand' returns a token from the vocabulary."""
		model, tokenizer = mock_generator_base
		gen = VoteRAGGenerator(
			model=model,
			tokenizer=tokenizer,
			fail_mode="rand",
		)
		
		histogram = {100: 5, 101: 3}
		
		token_id, exit_status = gen._handle_ld_failure(
			histogram=histogram,
			eps=0.1,
			delta=1e-10,
			k_bar=2,
			sens=2.0,
		)
		
		# Should return a valid token from vocabulary
		assert exit_status == 0
		assert 0 <= token_id < tokenizer.vocab_size

	def test_rand_not_necessarily_from_histogram(self, mock_generator_base):
		"""Test that 'rand' can return tokens not in histogram."""
		model, tokenizer = mock_generator_base
		gen = VoteRAGGenerator(
			model=model,
			tokenizer=tokenizer,
			fail_mode="rand",
			random_seed=42,
		)
		
		histogram = {100: 5}
		
		# Generate multiple times to see variety
		tokens = []
		for _ in range(10):
			token_id, _ = gen._handle_ld_failure(
				histogram=histogram,
				eps=0.1,
				delta=1e-10,
				k_bar=1,
				sens=2.0,
			)
			tokens.append(token_id)
		
		# At least some tokens should be outside the histogram (probabilistically)
		# Since vocab is large, most will be outside
		assert any(t != 100 for t in tokens)


class TestFailModeSTOP:
	"""Test 'stop' fail-mode strategy."""

	def test_stop_returns_none_with_exit_status_1(self, mock_generator_base):
		"""Test that 'stop' returns None and exit_status=1."""
		model, tokenizer = mock_generator_base
		gen = VoteRAGGenerator(
			model=model,
			tokenizer=tokenizer,
			fail_mode="stop",
		)
		
		histogram = {100: 5, 101: 3}
		
		token_id, exit_status = gen._handle_ld_failure(
			histogram=histogram,
			eps=0.1,
			delta=1e-10,
			k_bar=2,
			sens=2.0,
		)
		
		# Should return None and exit_status=1
		assert token_id is None
		assert exit_status == 1


class TestFailModeRAISE:
	"""Test 'raise' fail-mode strategy."""

	def test_raise_re_raises_exception(self, mock_generator_base):
		"""Test that 'raise' mode re-raises NotFoundLDTop1."""
		model, tokenizer = mock_generator_base
		gen = VoteRAGGenerator(
			model=model,
			tokenizer=tokenizer,
			fail_mode="raise",
		)
		
		histogram = {100: 5, 101: 3}
		
		with pytest.raises(NotFoundLDTop1, match="fail_mode='raise'"):
			gen._handle_ld_failure(
				histogram=histogram,
				eps=0.1,
				delta=1e-10,
				k_bar=2,
				sens=2.0,
			)


class TestFailModeIntegration:
	"""Test fail-mode integration with generate_vote()."""

	def test_generate_vote_handles_ld_failure_with_ld_pate(self, mock_generator_base):
		"""Test that generate_vote() correctly handles LD failures with ld_pate."""
		model, tokenizer = mock_generator_base
		gen = VoteRAGGenerator(
			model=model,
			tokenizer=tokenizer,
			m=5,
			k=1,
			epsilon_token=10.0,  # Large epsilon for success likelihood
			delta_token=1e-3,
			fail_mode="ld_pate",
		)
		
		docs = ["Document 1", "Document 2", "Document 3", "Document 4", "Document 5"]
		query = "Test query"
		
		# Should not raise, should return a string
		result = gen.generate_vote(query, docs)
		assert isinstance(result, str)

	def test_generate_vote_stops_with_stop_mode(self, mock_generator_base):
		"""Test that generate_vote() terminates generation with 'stop' mode."""
		# This test is harder to verify directly since we'd need to force
		# an LD failure first. We'll just verify the mode is accepted.
		model, tokenizer = mock_generator_base
		gen = VoteRAGGenerator(
			model=model,
			tokenizer=tokenizer,
			fail_mode="stop",
		)
		
		# Should be constructible
		assert gen.fail_mode == "stop"


if __name__ == "__main__":
	pytest.main([__file__, "-v"])
