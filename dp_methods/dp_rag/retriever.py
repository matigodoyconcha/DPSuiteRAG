from __future__ import annotations

from typing import Any

import numpy as np

from core.retriever import BaseRetriever


class DPRetriever(BaseRetriever):
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

		kth_idx = min(max(self.k - 1, 0), len(candidates) - 1)
		tau_base = float(candidates[kth_idx]["score"])

		rng = np.random.default_rng(self.seed)
		tau_noisy = tau_base + rng.laplace(loc=0.0, scale=self.sensitivity / self.epsilon)

		selected = [c for c in candidates if c["score"] >= tau_noisy]

		if len(selected) == 0:
			selected = [candidates[0]]

		if len(selected) > self.max_docs:
			selected = selected[: self.max_docs]

		return selected
