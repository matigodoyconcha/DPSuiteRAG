from __future__ import annotations

from typing import Iterable

from rouge_score import rouge_scorer


def _normalize_text(text: str) -> str:
	"""Normalize text for comparison.

	Lowercases and normalizes whitespace for consistent evaluation.

	Args:
		text: Text to normalize.

	Returns:
		Normalized text with consistent formatting.
	"""
	return " ".join(str(text).strip().lower().split())


def match_accuracy(predictions: list[str], references: list[str]) -> float:
	"""Calculate exact match accuracy between predictions and references.

	Compares normalized text for exact string matching.

	Args:
		predictions: List of predicted texts.
		references: List of ground truth references.

	Returns:
		Accuracy (0.0 to 1.0) of exact matches.
	"""
	if not predictions:
		return 0.0
	n = min(len(predictions), len(references))
	if n == 0:
		return 0.0
	correct = 0
	for pred, ref in zip(predictions[:n], references[:n]):
		if _normalize_text(pred) == _normalize_text(ref):
			correct += 1
	return float(correct) / float(n)


def rouge_l(predictions: list[str], references: list[str]) -> float:
	"""Calculate ROUGE-L F1 score between predictions and references.

	ROUGE-L measures longest common subsequence overlap, capturing
	semantic and structural similarity.

	Args:
		predictions: List of predicted texts.
		references: List of ground truth references.

	Returns:
		Average ROUGE-L F1 score (0.0 to 1.0).
	"""
	if not predictions:
		return 0.0
	n = min(len(predictions), len(references))
	if n == 0:
		return 0.0

	scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
	total = 0.0
	for pred, ref in zip(predictions[:n], references[:n]):
		score = scorer.score(str(ref), str(pred))["rougeL"].fmeasure
		total += float(score)
	return total / float(n)


def compute_utility_metrics(
	predictions: list[str],
	references: list[str],
	metric_names: Iterable[str],
) -> dict[str, float]:
	"""Compute utility metrics for a set of predictions.

	Args:
		predictions: Model predictions to evaluate.
		references: Ground truth references for comparison.
		metric_names: Names of metrics to compute (e.g. "match_accuracy", "rouge_l").

	Returns:
		Dictionary mapping metric names to their computed values.

	Raises:
		ValueError: If an unsupported metric name is provided.
	"""
	metrics: dict[str, float] = {}
	for name in metric_names:
		key = str(name).strip().lower()
		if key == "match_accuracy":
			metrics[key] = match_accuracy(predictions, references)
		elif key == "rouge_l":
			metrics[key] = rouge_l(predictions, references)
		else:
			raise ValueError(f"Utility metric '{name}' is not supported.")
	return metrics
