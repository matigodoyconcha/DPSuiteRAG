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
	"""Average number of private-entity occurrences per prediction.

	Paper definition (arXiv:2510.06719v2): "how often a patient's full name
	appears in the model's responses" — averaged across all predictions.
	Each prediction contributes a count (not a binary flag), so multiple
	entities leaking in one response are counted individually.
	"""
	if not predictions:
		return 0.0
	if not private_entities:
		return 0.0

	entities = [str(e).strip().lower() for e in private_entities if str(e).strip()]
	if not entities:
		return 0.0

	total = 0
	for pred in predictions:
		text = str(pred).lower()
		total += sum(1 for ent in entities if ent in text)
	return float(total) / float(len(predictions))


def bleu_membership_score(
	predictions: list[str],
	corpus_docs: list[str],
) -> float:
	"""Measure BLEU-based membership score against private corpus documents.

	For each prediction computes the maximum BLEU similarity over all corpus
	documents. The average across predictions indicates how lexically close
	the outputs are to the private corpus, serving as a leakage proxy.

	Args:
		predictions: List of generated answers to test.
		corpus_docs: Private corpus documents to compare against.

	Returns:
		Average max-BLEU score (0.0 = no similarity, 1.0 = full match).
	"""
	if not predictions or not corpus_docs:
		return 0.0

	corpus_tokens = [
		str(d).strip().split() for d in corpus_docs if str(d).strip()
	]
	if not corpus_tokens:
		return 0.0

	smooth = SmoothingFunction().method1
	total = 0.0
	for pred in predictions:
		pred_tokens = str(pred).strip().split()
		if not pred_tokens:
			continue
		scores = [
			float(sentence_bleu([ref_toks], pred_tokens, smoothing_function=smooth))
			for ref_toks in corpus_tokens
		]
		total += max(scores)
	return total / len(predictions)


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
	private_entities: list[str],
	metric_names: Iterable[str],
	seed: int = 42,
	**kwargs,
) -> dict[str, float]:
	"""Compute privacy metrics for a set of predictions.

	Extra inputs for richer metrics are passed via kwargs:
	  - ``corpus_docs``          (list[str]):              private documents, for ``bleu_membership_score``.
	  - ``dp_spent_per_query``   (list[dict[str,float]]):  per-query DP accounting, for ``privacy_budget_consumption``.

	Args:
		predictions: Model predictions to evaluate.
		references: Ground truth references for comparison.
		private_entities: Entities that should not be leaked.
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
		if key == "mia_auc_bleu":
			metrics[key] = mia_auc_bleu(predictions, references, seed=seed)
		elif key == "private_entity_leakage":
			metrics[key] = private_entity_leakage(predictions, private_entities)
		elif key == "bleu_membership_score":
			corpus_docs = list(kwargs.get("corpus_docs", []))
			metrics[key] = bleu_membership_score(predictions, corpus_docs)
		elif key == "privacy_budget_consumption":
			dp_spent = list(kwargs.get("dp_spent_per_query", []))
			metrics[key] = privacy_budget_consumption(dp_spent)
		else:
			raise ValueError(f"Privacy metric '{name}' is not supported.")
	return metrics
