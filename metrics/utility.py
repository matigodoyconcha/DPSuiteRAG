from __future__ import annotations

from typing import Iterable

import numpy as np
from rouge_score import rouge_scorer

from core.device import cuda_oom_warning


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
	"""Exact-match accuracy: fraction of predictions equal to the reference."""
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


def contains_accuracy(predictions: list[str], references: list[str]) -> float:
	"""Containment accuracy: fraction of predictions that include the reference.

	Paper definition (arXiv:2510.06719v2): "whether the model output includes
	the gold answer / fictitious disease name / top-10 movie title".
	"""
	if not predictions:
		return 0.0
	n = min(len(predictions), len(references))
	if n == 0:
		return 0.0
	correct = 0
	for pred, ref in zip(predictions[:n], references[:n]):
		if _normalize_text(ref) in _normalize_text(pred):
			correct += 1
	return float(correct) / float(n)


def hit_rate(predictions: list[str], reference_sets: list[list[str]]) -> float:
	"""Recommendation Hit@k: fraction of predictions that contain ANY acceptable
	answer from the corresponding per-item set (e.g. a user's top-k favorite
	movie titles).

	Used for Movielens in DP-SynRAG, where the gold is a SET, not a single
	string (paper: "the output includes any of the user's top-10 favorites").
	``reference_sets[i]`` is the list of acceptable golds for prediction ``i``;
	this is a NEW, set-valued metric and does not change the single-reference
	metrics. Matching uses the same lowercase/whitespace normalization.
	"""
	if not predictions or not reference_sets:
		return 0.0
	n = min(len(predictions), len(reference_sets))
	if n == 0:
		return 0.0
	hits = 0
	for pred, refs in zip(predictions[:n], reference_sets[:n]):
		p = _normalize_text(pred)
		for ref in (refs or []):
			rv = _normalize_text(ref)
			if rv and rv in p:
				hits += 1
				break
	return float(hits) / float(n)


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


def rouge_1(predictions: list[str], references: list[str]) -> float:
	"""Average ROUGE-1 F1 (unigram overlap) between predictions and references.

	Reported by DP-KSA (Tang et al., arXiv:2602.14374, §6.1) alongside ROUGE-L.
	"""
	if not predictions:
		return 0.0
	n = min(len(predictions), len(references))
	if n == 0:
		return 0.0

	scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
	total = 0.0
	for pred, ref in zip(predictions[:n], references[:n]):
		total += float(scorer.score(str(ref), str(pred))["rouge1"].fmeasure)
	return total / float(n)


def _levenshtein_distance(a: str, b: str) -> int:
	"""Character-level edit distance (insert/delete/substitute), iterative DP."""
	if a == b:
		return 0
	if not a:
		return len(b)
	if not b:
		return len(a)
	prev = list(range(len(b) + 1))
	for i, ca in enumerate(a, start=1):
		cur = [i]
		for j, cb in enumerate(b, start=1):
			cost = 0 if ca == cb else 1
			cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
		prev = cur
	return prev[-1]


def levenshtein(predictions: list[str], references: list[str]) -> float:
	"""Average normalized Levenshtein similarity: ``1 - edit_distance / max_len``.

	Reported by DP-KSA (arXiv:2602.14374, §6.1). Returns a similarity in [0, 1]
	(1.0 = identical after normalization), so higher is better like the other
	utility metrics.
	"""
	if not predictions:
		return 0.0
	n = min(len(predictions), len(references))
	if n == 0:
		return 0.0

	total = 0.0
	for pred, ref in zip(predictions[:n], references[:n]):
		p = _normalize_text(pred)
		r = _normalize_text(ref)
		if not p and not r:
			total += 1.0
			continue
		m = max(len(p), len(r))
		total += 1.0 - _levenshtein_distance(p, r) / m if m > 0 else 0.0
	return total / float(n)


def f1_score(predictions: list[str], references: list[str]) -> float:
	"""Calculate F1 score (token-based) between predictions and references.

	F1 is computed as 2 * (precision * recall) / (precision + recall) based on
	word overlap. This is the standard metric for SQuAD and similar QA datasets.

	Args:
		predictions: List of predicted texts.
		references: List of ground truth references.

	Returns:
		Average F1 score (0.0 to 1.0).
	"""
	if not predictions:
		return 0.0
	n = min(len(predictions), len(references))
	if n == 0:
		return 0.0

	total_f1 = 0.0
	for pred, ref in zip(predictions[:n], references[:n]):
		pred_tokens = set(_normalize_text(pred).split())
		ref_tokens = set(_normalize_text(ref).split())

		# Handle empty cases
		if len(ref_tokens) == 0 and len(pred_tokens) == 0:
			total_f1 += 1.0
			continue
		if len(ref_tokens) == 0 or len(pred_tokens) == 0:
			total_f1 += 0.0
			continue

		# Compute precision, recall, and F1
		common = len(pred_tokens & ref_tokens)
		precision = float(common) / float(len(pred_tokens)) if len(pred_tokens) > 0 else 0.0
		recall = float(common) / float(len(ref_tokens)) if len(ref_tokens) > 0 else 0.0

		if precision + recall > 0:
			f1 = 2.0 * (precision * recall) / (precision + recall)
		else:
			f1 = 0.0
		total_f1 += f1

	return total_f1 / float(n)


def bert_score_f1(
	predictions: list[str],
	references: list[str],
	lang: str = "en",
) -> float:
	"""Calculate BERTScore F1 using contextual BERT embeddings.

	Args:
		predictions: List of predicted texts.
		references: List of ground truth references.
		lang: Language code for the BERTScore model.

	Returns:
		Average BERTScore F1 (0.0 to 1.0).
	"""
	if not predictions:
		return 0.0
	n = min(len(predictions), len(references))
	if n == 0:
		return 0.0

	# We request the fast (Rust) tokenizer because with recent transformers the
	# *slow* RobertaTokenizer no longer exposes ``build_inputs_with_special_tokens``,
	# which bert_score calls internally. But if the environment cannot build a fast
	# tokenizer it silently falls back to the slow one and still raises that
	# AttributeError. The import itself can also fail (e.g. transformers refusing to
	# load against an out-of-range tokenizers pin). Rather than aborting the whole
	# offline scoring pass over every result file for one finicky metric, degrade to
	# NaN ("not computed") and warn once — all other metrics are still written. The
	# import is inside the try so import-time failures degrade too.
	try:
		from bert_score import score as _bert_score

		_, _, F = _bert_score(
			predictions[:n],
			references[:n],
			lang=lang,
			verbose=False,
			use_fast_tokenizer=True,
		)
		return float(F.mean().item())
	except Exception as exc:  # noqa: BLE001 — any tokenizer/version failure
		global _BERTSCORE_WARNED
		if not _BERTSCORE_WARNED:
			print(f"  [warn] bert_score_f1 unavailable, recording NaN: {exc}")
			_BERTSCORE_WARNED = True
		return float("nan")


_BERTSCORE_WARNED = False


def recall_at_k(
	retrieved_ids: list[list],
	relevant_ids: list[list],
) -> float:
	"""Calculate Recall@k: fraction of relevant docs found in retrieved set per query.

	Args:
		retrieved_ids: For each query, the IDs of retrieved documents.
		relevant_ids: For each query, the IDs of ground-truth relevant documents.

	Returns:
		Average Recall@k (0.0 to 1.0).
	"""
	if not retrieved_ids:
		return 0.0
	n = min(len(retrieved_ids), len(relevant_ids))
	if n == 0:
		return 0.0

	total = 0.0
	counted = 0
	for ret, rel in zip(retrieved_ids[:n], relevant_ids[:n]):
		rel_set = set(rel)
		if not rel_set:
			continue
		ret_set = set(ret)
		total += len(ret_set & rel_set) / len(rel_set)
		counted += 1

	return total / counted if counted > 0 else 0.0


def answer_faithfulness(
	predictions: list[str],
	contexts: list[list[str]],
) -> float:
	"""Estimate answer faithfulness as ROUGE-1 precision of prediction over retrieved context.

	Measures how much of the generated answer is supported by the retrieved chunks
	(recall of answer tokens that appear in context).

	Args:
		predictions: List of generated answers.
		contexts: For each query, a list of retrieved context strings.

	Returns:
		Average faithfulness score (0.0 to 1.0).
	"""
	if not predictions:
		return 0.0
	n = min(len(predictions), len(contexts))
	if n == 0:
		return 0.0

	_scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
	total = 0.0
	for pred, ctx_list in zip(predictions[:n], contexts[:n]):
		if not ctx_list:
			continue
		context = " ".join(str(c) for c in ctx_list)
		# Precision: fraction of answer tokens that appear in context.
		score = _scorer.score(context, str(pred))["rouge1"].precision
		total += float(score)
	return total / n


def answer_relevancy(
	queries: list[str],
	predictions: list[str],
	model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
	"""Estimate answer relevancy as cosine similarity between query and answer embeddings.

	Args:
		queries: List of input queries.
		predictions: List of generated answers.
		model_name: Sentence-transformer model to use for encoding.

	Returns:
		Average cosine similarity (0.0 to 1.0 after clipping).
	"""
	if not queries or not predictions:
		return 0.0
	n = min(len(queries), len(predictions))
	if n == 0:
		return 0.0

	from sentence_transformers import SentenceTransformer

	with cuda_oom_warning(model_name):
		_model = SentenceTransformer(model_name)
	q_embs = _model.encode(queries[:n], convert_to_numpy=True, show_progress_bar=False)
	p_embs = _model.encode(predictions[:n], convert_to_numpy=True, show_progress_bar=False)

	q_norms = np.linalg.norm(q_embs, axis=1, keepdims=True)
	p_norms = np.linalg.norm(p_embs, axis=1, keepdims=True)
	q_embs = q_embs / np.maximum(q_norms, 1e-8)
	p_embs = p_embs / np.maximum(p_norms, 1e-8)

	cos_sims = (q_embs * p_embs).sum(axis=1)
	return float(np.clip(cos_sims, 0.0, 1.0).mean())


def perplexity(
	texts: list[str],
	model_name: str = "gpt2",
	device: str = "auto",
) -> float:
	"""Average perplexity of texts under a causal language model.

	Used in the paper (Appendix C.4) to measure synthetic text quality.
	Lower perplexity = more fluent text.
	"""
	import torch
	from transformers import AutoModelForCausalLM, AutoTokenizer

	valid = [t for t in texts if t and t.strip()]
	if not valid:
		# All predictions empty/degenerate (e.g. dpvoterag LD-Gumbel halts -> "").
		# Return NaN, not inf: inf poisons any mean across eps levels / methods,
		# while NaN is naturally excluded from aggregates.
		return float("nan")

	if device == "auto":
		if torch.cuda.is_available():
			device = "cuda"
		elif torch.backends.mps.is_available():
			device = "mps"
		else:
			device = "cpu"

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token
	with cuda_oom_warning(model_name, device):
		model = AutoModelForCausalLM.from_pretrained(model_name)
		model = model.to(device)
	model.eval()

	total_nll = 0.0
	total_tokens = 0
	with torch.no_grad():
		for text in valid:
			enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
			input_ids = enc["input_ids"].to(device)
			if input_ids.shape[1] < 2:
				continue
			loss = model(input_ids, labels=input_ids).loss
			n = input_ids.shape[1] - 1
			total_nll += loss.item() * n
			total_tokens += n

	if total_tokens == 0:
		return float("nan")
	return float(np.exp(total_nll / total_tokens))


def compute_utility_metrics(
	predictions: list[str],
	references: list[str],
	metric_names: Iterable[str],
	**kwargs,
) -> dict[str, float]:
	"""Compute utility metrics for a set of predictions.

	Extra inputs for richer metrics are passed via kwargs:
	  - ``reference_sets`` (list[list[str]]): per-item acceptable gold sets, for ``hit_rate``.
	  - ``contexts``  (list[list[str]]): retrieved chunks per query, for ``answer_faithfulness``.
	  - ``queries``   (list[str]):       input queries, for ``answer_relevancy``.
	  - ``retrieved_ids`` / ``relevant_ids`` (list[list]): for ``recall_at_k``.
	  - ``lang``      (str):             BERTScore language, default ``"en"``.
	  - ``embedder_model`` (str):        sentence-transformer name for ``answer_relevancy``.

	Args:
		predictions: Model predictions to evaluate.
		references: Ground truth references for comparison.
		metric_names: Names of metrics to compute.

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
		elif key == "contains_accuracy":
			metrics[key] = contains_accuracy(predictions, references)
		elif key == "rouge_l":
			metrics[key] = rouge_l(predictions, references)
		elif key == "hit_rate":
			reference_sets = kwargs.get("reference_sets", [])
			metrics[key] = hit_rate(predictions, reference_sets)
		elif key == "rouge_1":
			metrics[key] = rouge_1(predictions, references)
		elif key == "levenshtein":
			metrics[key] = levenshtein(predictions, references)
		elif key == "f1_score":
			metrics[key] = f1_score(predictions, references)
		elif key == "bert_score_f1":
			lang = str(kwargs.get("lang", "en"))
			metrics[key] = bert_score_f1(predictions, references, lang=lang)
		elif key == "recall_at_k":
			retrieved_ids = kwargs.get("retrieved_ids", [])
			relevant_ids = kwargs.get("relevant_ids", [])
			metrics[key] = recall_at_k(retrieved_ids, relevant_ids)
		elif key == "answer_faithfulness":
			contexts = kwargs.get("contexts", [])
			metrics[key] = answer_faithfulness(predictions, contexts)
		elif key == "answer_relevancy":
			queries = kwargs.get("queries", [])
			model_name = str(kwargs.get("embedder_model", "sentence-transformers/all-MiniLM-L6-v2"))
			metrics[key] = answer_relevancy(queries, predictions, model_name=model_name)
		elif key == "perplexity":
			# Computes perplexity over synthetic texts when provided, else predictions.
			texts_for_ppl = kwargs.get("texts", predictions)
			ppl_model = str(kwargs.get("perplexity_model", "gpt2"))
			metrics[key] = perplexity(texts_for_ppl, model_name=ppl_model)
		else:
			raise ValueError(f"Utility metric '{name}' is not supported.")
	return metrics
