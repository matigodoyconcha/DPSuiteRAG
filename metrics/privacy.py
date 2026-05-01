from __future__ import annotations

from typing import Iterable

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def _bleu_similarity(prediction: str, reference: str) -> float:
	"""Calculate BLEU similarity between prediction and reference.

	Args:
		prediction: Predicted text.
		reference: Reference/ground truth text.

	Returns:
		BLEU score between 0 and 1.
	"""
	pred_tokens = str(prediction).strip().split()
	ref_tokens = str(reference).strip().split()
	if not pred_tokens or not ref_tokens:
		return 0.0
	smooth = SmoothingFunction().method1
	return float(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth))


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
	"""Calculate ROC-AUC score from binary labels and scores.

	Args:
		labels: Binary labels (0 or 1).
		scores: Scores to use for ROC curve.

	Returns:
		ROC-AUC score between 0 and 1.
	"""
	pos = int((labels == 1).sum())
	neg = int((labels == 0).sum())
	if pos == 0 or neg == 0:
		return 0.5

	order = np.argsort(scores)
	ranks = np.empty_like(order, dtype=float)
	ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
	rank_sum_pos = float(ranks[labels == 1].sum())
	auc = (rank_sum_pos - (pos * (pos + 1) / 2.0)) / (pos * neg)
	return float(max(0.0, min(1.0, auc)))


def mia_auc_bleu(predictions: list[str], references: list[str], seed: int = 42) -> float:
	"""Calculate membership inference attack (MIA) AUC using BLEU similarity.

	Compares BLEU similarity with true references vs randomly shuffled references
	to measure information leakage through the model's predictions.

	Args:
		predictions: List of model predictions.
		references: List of true reference responses.
		seed: Random seed for reproducibility.

	Returns:
		ROC-AUC score indicating privacy leakage (0.5 = no leakage, 1.0 = full leakage).
	"""
	n = min(len(predictions), len(references))
	if n == 0:
		return 0.5

	preds = [str(x) for x in predictions[:n]]
	refs = [str(x) for x in references[:n]]

	rng = np.random.default_rng(seed)
	perm = rng.permutation(n)
	shuffled_refs = [refs[i] for i in perm]

	pos_scores = np.array([_bleu_similarity(p, r) for p, r in zip(preds, refs)], dtype=float)
	neg_scores = np.array([_bleu_similarity(p, r) for p, r in zip(preds, shuffled_refs)], dtype=float)

	labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
	scores = np.concatenate([pos_scores, neg_scores])
	return _roc_auc(labels, scores)


def private_entity_leakage(predictions: list[str], private_entities: list[str]) -> float:
	"""Measure leakage of private entities in predictions.

	Calculates the fraction of predictions containing any of the private entities
	that should have been protected by differential privacy.

	Args:
		predictions: List of model predictions to check.
		private_entities: List of private entities that should not appear in predictions.

	Returns:
		Leakage ratio (0.0 = no leakage, 1.0 = all predictions leaked private info).
	"""
	if not predictions:
		return 0.0
	if not private_entities:
		return 0.0

	entities = [str(e).strip().lower() for e in private_entities if str(e).strip()]
	if not entities:
		return 0.0

	leak_count = 0
	for pred in predictions:
		text = str(pred).lower()
		if any(ent in text for ent in entities):
			leak_count += 1
	return float(leak_count) / float(len(predictions))


def compute_privacy_metrics(
	predictions: list[str],
	references: list[str],
	private_entities: list[str],
	metric_names: Iterable[str],
	seed: int = 42,
) -> dict[str, float]:
	"""Compute privacy metrics for a set of predictions.

	Args:
		predictions: Model predictions to evaluate.
		references: Ground truth references for comparison.
		private_entities: Entities that should not be leaked.
		metric_names: Names of metrics to compute (e.g. "mia_auc_bleu", "private_entity_leakage").
		seed: Random seed for reproducibility.

	Returns:
		Dictionary mapping metric names to their computed values.

	Raises:
		ValueError: If an unsupported metric name is provided.
	"""
	metrics: dict[str, float] = {}
	for name in metric_names:
		key = str(name).strip().lower()
		if key == "mia_auc_bleu":
			metrics[key] = mia_auc_bleu(predictions, references, seed=seed)
		elif key == "private_entity_leakage":
			metrics[key] = private_entity_leakage(predictions, private_entities)
		else:
			raise ValueError(f"Privacy metric '{name}' is not supported.")
	return metrics
