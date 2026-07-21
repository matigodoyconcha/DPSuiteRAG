from __future__ import annotations

from typing import Iterable


def privacy_budget_consumption(
	dp_spent_per_query: list[dict[str, float]],
) -> float:
	"""Compute average effective epsilon consumed per query.

	Sums the per-query epsilon values and divides by the number of queries,
	giving the mean privacy cost under sequential composition.

	Args:
		dp_spent_per_query: List of dicts with at least an ``"epsilon"`` key,
		    one entry per query (as produced by VoteRAG ``last_run_stats``).

	Returns:
		Mean epsilon per query (0.0 if the list is empty).
	"""
	if not dp_spent_per_query:
		return 0.0
	total = sum(float(item.get("epsilon", 0.0)) for item in dp_spent_per_query)
	return total / len(dp_spent_per_query)


def compute_privacy_metrics(
	predictions: list[str],
	references: list[str],
	metric_names: Iterable[str],
	seed: int = 42,
	**kwargs,
) -> dict[str, float]:
	"""Compute privacy metrics for a set of predictions.

	Extra inputs for richer metrics are passed via kwargs:
	  - ``dp_spent_per_query``   (list[dict[str,float]]):  per-query DP accounting, for ``privacy_budget_consumption``.

	Args:
		predictions: Model predictions to evaluate.
		references: Ground truth references for comparison.
		metric_names: Names of metrics to compute.
		seed: Random seed for reproducibility.

	Returns:
		Dictionary mapping metric names to their computed values.

	Raises:
		ValueError: If an unsupported metric name is provided.
	"""
	metrics: dict[str, float] = {}
	for name in metric_names:
		key = str(name).strip().lower()
		if key == "privacy_budget_consumption":
			dp_spent = list(kwargs.get("dp_spent_per_query", []))
			metrics[key] = privacy_budget_consumption(dp_spent)
		else:
			raise ValueError(f"Privacy metric '{name}' is not supported.")
	return metrics
