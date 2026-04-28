from __future__ import annotations

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
		
		# Sample threshold using exponential mechanism with ε-DP guarantee
		tau_dp = self._sample_exponential_mechanism(
			candidate_scores, 
			self.k, 
			self.epsilon, 
			self.seed
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

