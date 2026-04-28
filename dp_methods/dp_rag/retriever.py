from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from core.retriever import BaseRetriever


class DPRetriever(BaseRetriever):
	"""
	DP-RAG Retriever: Implements privacy-preserving top-k document selection.
	
	Reference: Grislain et al., "RAG with Differential Privacy", arXiv:2412.19291v2 (2024)
	
	Privacy mechanism:
	- Uses exponential mechanism to sample threshold τ_top-k
	- Utility function: U_top-k(τ) = -|#{documents with score ≥ τ} - k|
	- Threshold sampled from: τ ∝ exp(ε · U_top-k(τ) / 2)
	- Sensitivity: Δ = 1 (changing one document changes U by at most 1)
	
	The factor 1/2 in the exponential is critical for the DP proof.
	"""
	
	def __init__(
		self,
		index,
		epsilon: float,
		k: int = 5,
		max_docs: int = 20,
		sensitivity: float = 1.0,
		seed: int = 42,
	):
		if epsilon <= 0:
			raise ValueError("epsilon must be > 0 for DP retrieval.")
		if k <= 0:
			raise ValueError("k must be >= 1.")
		if max_docs <= 0:
			raise ValueError("max_docs must be >= 1.")
		if sensitivity <= 0:
			raise ValueError("sensitivity must be > 0.")

		self.indexer = index
		self.epsilon = float(epsilon)
		self.k = int(k)
		self.max_docs = int(max_docs)
		self.sensitivity = float(sensitivity)
		self.seed = int(seed)

	def _get_index(self) -> Any:
		idx = getattr(self.indexer, "index", None)
		if idx is None and hasattr(self.indexer, "get_index"):
			idx = self.indexer.get_index()
		return idx

	def _utility_top_k(self, scores: list[float], tau: float, k: int) -> float:
		"""
		Compute utility function for top-k mechanism.
		
		Args:
			scores: List of document similarity scores
			tau: Threshold value
			k: Target number of documents
			
		Returns:
			U_top-k(τ) = -|#{documents with score ≥ τ} - k|
			
		This function measures how far the count of documents above threshold
		is from the target k. The negative sign follows the convention that
		utility increases as the goal is met.
		"""
		count_above_tau = sum(1 for s in scores if s >= tau)
		utility = -abs(count_above_tau - k)
		return float(utility)

	def _sample_exponential_mechanism(
		self, 
		scores: list[float], 
		k: int, 
		epsilon: float, 
		seed: int
	) -> float:
		"""
		Sample threshold τ using the exponential mechanism.
		
		The exponential mechanism (Dwork & Roth, 2014) provides ε-DP when sampling
		from: τ ∝ exp(ε · U(τ) / (2 · Δ))
		
		where:
		- U(τ) is the utility function
		- Δ = 1 is the sensitivity (changing one document changes U by at most 1)
		- The factor 1/2 in the exponent appears because we normalize by 2·Δ
		
		Args:
			scores: List of document similarity scores
			k: Target number of documents  
			epsilon: Privacy budget
			seed: Random seed for reproducibility
			
		Returns:
			Sampled threshold τ_top-k
		"""
		if not scores:
			return 0.0
		
		# Candidate thresholds: unique scores (sorted descending)
		unique_scores = sorted(set(scores), reverse=True)
		
		if not unique_scores:
			return 0.0
		
		# Compute utility for each candidate threshold
		utilities = np.array([self._utility_top_k(scores, tau, k) for tau in unique_scores])
		
		# Compute exponential weights: exp(ε · U(τ) / 2)
		# Dividing by 2 because sensitivity Δ=1, so we use ε/(2·Δ) = ε/2
		exp_weights = np.exp((self.epsilon * utilities) / 2.0)
		
		# Normalize to get probabilities
		probabilities = exp_weights / np.sum(exp_weights)
		
		# Sample threshold from the distribution
		rng = np.random.default_rng(seed)
		sampled_threshold = float(rng.choice(unique_scores, p=probabilities))
		
		return sampled_threshold

	def retrieve(self, query_vector: np.ndarray, k: int = None) -> list[dict]:
		index = self._get_index()
		if index is None:
			raise RuntimeError("Index has not been built or loaded yet.")

		ntotal = int(getattr(index, "ntotal", 0))
		if ntotal <= 0:
			return []

		query_vec = np.array([query_vector], dtype=np.float32)
		norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
		norm[norm == 0.0] = 1.0
		query_vec = query_vec / norm

		search_k = min(ntotal, max(1, self.max_docs * 3))
		scores, indices = index.search(query_vec, search_k)

		candidates: list[dict] = []
		metadata = getattr(self.indexer, "metadata", [])
		for j, idx in enumerate(indices[0]):
			if idx == -1:
				continue
			if idx >= len(metadata):
				continue
			chunk_meta = metadata[idx]
			text = str(chunk_meta.get("text", "")) if isinstance(chunk_meta, dict) else ""
			candidates.append(
				{
					"text": text,
					"metadata": chunk_meta,
					"score": float(scores[0][j]),
				}
			)

		if not candidates:
			return []

		candidates.sort(key=lambda x: x["score"], reverse=True)

		# Extract scores for exponential mechanism
		candidate_scores = [c["score"] for c in candidates]

		# Per-query seed: same query => same noise; different queries => different noise.
		query_offset = int(hashlib.sha256(query_vec.tobytes()).hexdigest()[:8], 16)
		query_seed = self.seed + query_offset

		# Sample threshold using exponential mechanism with ε-DP guarantee
		tau_dp = self._sample_exponential_mechanism(
			candidate_scores,
			self.k,
			self.epsilon,
			query_seed,
		)

		# Select all documents with score >= sampled threshold
		selected = [c for c in candidates if c["score"] >= tau_dp]

		# Fallback: if no documents selected (rare with continuous scores), 
		# include at least the first candidate
		if len(selected) == 0:
			selected = [candidates[0]]

		# Cap at max_docs if needed
		if len(selected) > self.max_docs:
			selected = selected[: self.max_docs]

		return selected


class DPTopPRetriever(BaseRetriever):
	"""
	DP-RAG Top-P Retriever: Implements privacy-preserving weighted document selection.
	
	Reference: Grislain et al., "RAG with Differential Privacy", arXiv:2412.19291v2 (2024)
	
	Privacy mechanism:
	- Uses exponential mechanism to sample threshold τ_top-p
	- Documents weighted by relevance: w(s) = exp(α · (s − s_max) / (s_max − s_min))
	- Utility function: U_top-p(τ) = -|Σᵢ 1[0,sᵢ](τ)·w(sᵢ) − p·Σᵢ w(sᵢ)|
	- Threshold sampled from: τ ∝ exp(ε · U_top-p(τ) / 2)
	- Sensitivity: Δ = 1 (changing one document changes U by at most 1)
	
	This is more flexible than top-k because it weights documents by their score.
	"""
	
	def __init__(
		self,
		index,
		epsilon: float,
		p: float = 0.3,
		alpha: float = 1.0,
		max_docs: int = 20,
		sensitivity: float = 1.0,
		seed: int = 42,
	):
		if epsilon <= 0:
			raise ValueError("epsilon must be > 0 for DP retrieval.")
		if not (0.0 < p <= 1.0):
			raise ValueError("p must be in (0, 1] for top-p selection.")
		if alpha <= 0:
			raise ValueError("alpha must be > 0 for weight function.")
		if max_docs <= 0:
			raise ValueError("max_docs must be >= 1.")
		if sensitivity <= 0:
			raise ValueError("sensitivity must be > 0.")

		self.indexer = index
		self.epsilon = float(epsilon)
		self.p = float(p)
		self.alpha = float(alpha)
		self.max_docs = int(max_docs)
		self.sensitivity = float(sensitivity)
		self.seed = int(seed)

	def _get_index(self) -> Any:
		idx = getattr(self.indexer, "index", None)
		if idx is None and hasattr(self.indexer, "get_index"):
			idx = self.indexer.get_index()
		return idx

	def _weight_function(self, score: float, s_max: float, s_min: float) -> float:
		"""
		Compute weight function for document.
		
		w(s) = exp(α · (s − s_max) / (s_max − s_min))
		
		Args:
			score: Document similarity score
			s_max: Maximum score in the set
			s_min: Minimum score in the set
			
		Returns:
			Weight w(s) ∈ (0, 1] for α > 0
		"""
		if s_max == s_min:
			# Edge case: all scores are equal
			return 1.0
		
		normalized = (score - s_max) / (s_max - s_min)
		weight = float(np.exp(self.alpha * normalized))
		return weight

	def _utility_top_p(
		self, 
		scores: list[float], 
		tau: float, 
		p: float, 
		s_max: float, 
		s_min: float
	) -> float:
		"""
		Compute utility function for top-p mechanism.
		
		Args:
			scores: List of document similarity scores
			tau: Threshold value
			p: Target proportion of total weight
			s_max: Maximum score in the set
			s_min: Minimum score in the set
			
		Returns:
			U_top-p(τ) = -|Σᵢ 1[0,sᵢ](τ)·w(sᵢ) − p·Σᵢ w(sᵢ)|
			
		This function measures the difference between the weighted count
		of documents above threshold and the target weighted fraction p.
		"""
		# Compute weight for each score
		weights = np.array([self._weight_function(s, s_max, s_min) for s in scores])
		
		# Count documents with score >= tau, weighted
		above_tau = np.array([1.0 if s >= tau else 0.0 for s in scores])
		weighted_count_above = float(np.sum(weights * above_tau))
		
		# Total weight (sum of all weights)
		total_weight = float(np.sum(weights))
		
		# Target weight (p fraction of total)
		target_weight = p * total_weight
		
		# Utility: how far we are from the target
		utility = -abs(weighted_count_above - target_weight)
		return float(utility)

	def _sample_exponential_mechanism(
		self,
		scores: list[float],
		p: float,
		epsilon: float,
		seed: int,
		s_max: float,
		s_min: float,
	) -> float:
		"""
		Sample threshold τ using the exponential mechanism.
		
		The exponential mechanism provides ε-DP when sampling from:
		τ ∝ exp(ε · U(τ) / (2 · Δ))
		
		where:
		- U(τ) is the utility function
		- Δ = 1 is the sensitivity
		- The factor 1/2 is critical for the DP proof
		
		Args:
			scores: List of document similarity scores
			p: Target proportion of total weight
			epsilon: Privacy budget
			seed: Random seed for reproducibility
			s_max: Maximum score in the set
			s_min: Minimum score in the set
			
		Returns:
			Sampled threshold τ_top-p
		"""
		if not scores:
			return 0.0
		
		# Candidate thresholds: unique scores (sorted descending)
		unique_scores = sorted(set(scores), reverse=True)
		
		if not unique_scores:
			return 0.0
		
		# Compute utility for each candidate threshold
		utilities = np.array([
			self._utility_top_p(scores, tau, p, s_max, s_min) 
			for tau in unique_scores
		])
		
		# Compute exponential weights: exp(ε · U(τ) / 2)
		# Dividing by 2 because sensitivity Δ=1, so we use ε/(2·Δ) = ε/2
		exp_weights = np.exp((epsilon * utilities) / 2.0)
		
		# Normalize to get probabilities
		probabilities = exp_weights / np.sum(exp_weights)
		
		# Sample threshold from the distribution
		rng = np.random.default_rng(seed)
		sampled_threshold = float(rng.choice(unique_scores, p=probabilities))
		
		return sampled_threshold

	def retrieve(self, query_vector: np.ndarray, k: int = None) -> list[dict]:
		index = self._get_index()
		if index is None:
			raise RuntimeError("Index has not been built or loaded yet.")

		ntotal = int(getattr(index, "ntotal", 0))
		if ntotal <= 0:
			return []

		query_vec = np.array([query_vector], dtype=np.float32)
		norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
		norm[norm == 0.0] = 1.0
		query_vec = query_vec / norm

		search_k = min(ntotal, max(1, self.max_docs * 3))
		scores, indices = index.search(query_vec, search_k)

		candidates: list[dict] = []
		metadata = getattr(self.indexer, "metadata", [])
		for j, idx in enumerate(indices[0]):
			if idx == -1:
				continue
			if idx >= len(metadata):
				continue
			chunk_meta = metadata[idx]
			text = str(chunk_meta.get("text", "")) if isinstance(chunk_meta, dict) else ""
			candidates.append(
				{
					"text": text,
					"metadata": chunk_meta,
					"score": float(scores[0][j]),
				}
			)

		if not candidates:
			return []

		candidates.sort(key=lambda x: x["score"], reverse=True)

		# Extract scores for exponential mechanism
		candidate_scores = [c["score"] for c in candidates]
		s_max = max(candidate_scores)
		s_min = min(candidate_scores)

		# Per-query seed: same query => same noise; different queries => different noise.
		query_offset = int(hashlib.sha256(query_vec.tobytes()).hexdigest()[:8], 16)
		query_seed = self.seed + query_offset

		# Sample threshold using exponential mechanism with ε-DP guarantee
		tau_dp = self._sample_exponential_mechanism(
			candidate_scores,
			self.p,
			self.epsilon,
			query_seed,
			s_max,
			s_min,
		)

		# Select all documents with score >= sampled threshold
		selected = [c for c in candidates if c["score"] >= tau_dp]

		# Fallback: if no documents selected (rare with continuous scores), 
		# include at least the first candidate
		if len(selected) == 0:
			selected = [candidates[0]]

		# Cap at max_docs if needed
		if len(selected) > self.max_docs:
			selected = selected[: self.max_docs]

		return selected

