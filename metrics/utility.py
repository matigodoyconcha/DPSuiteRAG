from __future__ import annotations

import re
import string
from collections import Counter
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


_ARTICLES_RE = re.compile(r"\b(?:a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _squad_normalize(text: str) -> str:
	"""SQuAD / QA-standard answer normalization (Rajpurkar et al. 2016).

	Lowercases, strips punctuation, drops the articles a/an/the, and collapses
	whitespace. This is the normalization the standard EM/F1 QA metrics use (and
	the metric the DP-KSA / Private-RAG papers we benchmark against report), so a
	prediction like ``'David Seville.'`` or ``'The Phantom of the Opera'`` is
	scored on content instead of being penalized for a trailing period or a
	leading article. Used by ``match_accuracy`` (EM) and ``f1_score`` (token-F1).
	"""
	s = str(text).lower().translate(_PUNCT_TABLE)
	s = _ARTICLES_RE.sub(" ", s)
	return " ".join(s.split())


def _as_ref_list(ref) -> list[str]:
	"""A reference may be a single gold string OR a list of acceptable golds
	(aliases, e.g. TriviaQA's answer.aliases or NQ-open's multiple answers).
	Normalize either shape to a non-empty list of strings so every metric can
	score max-over-aliases uniformly and stay backward compatible with the
	single-string callers.
	"""
	if isinstance(ref, (list, tuple, set)):
		out = [str(r) for r in ref]
		return out or [""]
	return [str(ref)]


def _has_content(text) -> bool:
	"""True when the string still carries an answer after QA normalization.

	``_squad_normalize`` strips punctuation and the articles a/an/the, so ``"The"``,
	``"a"`` and ``"!!!"`` all reduce to nothing at all. Emptiness has to be judged on
	what the metrics actually compare, not on the raw string.
	"""
	return bool(_squad_normalize(text))


def has_scoreable_reference(reference) -> bool:
	"""True when at least one acceptable gold has content.

	Accepts either shape a reference can take — a single string or an alias list —
	so loaders and the scorer decide "is there anything here to score against?" by
	the same rule.
	"""
	return any(_has_content(r) for r in _as_ref_list(reference))


def _is_scoreable(pred: str, reference) -> bool:
	"""Whether a prediction/reference pair says anything about the method.

	A pair where the prediction is empty AND no gold survives normalization compares
	nothing to nothing. Exact match and token-F1 both call that a perfect hit, which
	hands the top score of a run to a cell that answered none of its questions — and
	it hands it to precisely the mechanisms that return an empty string when their
	budget runs out, halt or decline to release. Such a pair is not scored at all.

	Only the doubly-degenerate case is excluded. A real answer against a contentless
	gold, and an empty answer against a real gold, are both scored, and both score 0:
	the item is still a question the method got wrong.
	"""
	return _has_content(pred) or has_scoreable_reference(reference)


def _scoreable_pairs(predictions: list[str], references: list) -> list[tuple[str, object]]:
	"""The aligned (prediction, reference) pairs that have something to compare.

	Every pair-comparison metric iterates this, so all of them average over the same
	items and one dropped-item count describes the whole cell.
	"""
	n = min(len(predictions), len(references))
	return [
		(p, r)
		for p, r in zip(predictions[:n], references[:n])
		if _is_scoreable(p, r)
	]


#: The metrics that compare a prediction to a reference item by item. They all average
#: over :func:`_scoreable_pairs`, so one coverage count describes the whole group.
_PAIR_COMPARISON_METRICS = {
	"match_accuracy", "contains_accuracy", "rouge_l", "rouge_1",
	"levenshtein", "f1_score", "bert_score_f1",
}

#: The denominator those metrics used, and what they dropped.
_COVERAGE_METRICS = {"n_scored", "n_unscoreable"}


def scoring_coverage(predictions: list[str], references: list) -> dict[str, int]:
	"""How many aligned items the pair-comparison metrics could score, and how many
	they had to drop.

	A mean over 37 of 40 questions is a different measurement from a mean over 40, so
	the denominator travels with the numbers rather than being reconstructed later.
	"""
	n = min(len(predictions), len(references))
	scored = len(_scoreable_pairs(predictions, references))
	return {"n_scored": scored, "n_unscoreable": n - scored}


def _em_pair(pred: str, ref: str) -> float:
	"""Exact match (SQuAD-normalized) for one prediction/reference pair."""
	if not _is_scoreable(pred, ref):
		# Nothing against nothing is not a match. The aggregates drop this pair
		# outright; the pair helper agrees so a direct caller cannot be credited either.
		return 0.0
	return 1.0 if _squad_normalize(pred) == _squad_normalize(ref) else 0.0


def _token_f1_pair(pred: str, ref: str) -> float:
	"""SQuAD token-F1 (multiset overlap) for one prediction/reference pair."""
	pt = _squad_normalize(pred).split()
	rt = _squad_normalize(ref).split()
	if not pt or not rt:
		# Includes the both-empty case, which is not a perfect overlap of anything.
		return 0.0
	common = sum((Counter(pt) & Counter(rt)).values())
	if common == 0:
		return 0.0
	precision = common / len(pt)
	recall = common / len(rt)
	return 2.0 * precision * recall / (precision + recall)


def _strip_parentheticals(text: str) -> str:
	"""Drop parenthetical groups, e.g. '(Helio) Castroneves' -> 'Castroneves'.

	SearchQA golds often carry a parenthetical alias/first name the prediction
	omits; matching on the stripped core removes that systematic false negative.
	"""
	return " ".join(re.sub(r"\([^)]*\)", " ", str(text)).split())


def _canonicalize_title(text: str) -> str:
	"""Normalize a movie title for substring matching.

	Drops a trailing '(YYYY)' and moves a trailing article ('..., The') to the
	front, so the MovieLens gold 'Shawshank Redemption, The (1994)' matches a
	natural-order prediction 'The Shawshank Redemption'.
	"""
	s = re.sub(r"\(\d{4}\)", " ", str(text))
	s = " ".join(s.split())
	m = re.match(
		r"^(.*),\s*(the|a|an|la|le|les|los|las|il|el|die|der|das)$",
		s,
		flags=re.IGNORECASE,
	)
	if m:
		s = f"{m.group(2)} {m.group(1)}"
	return _normalize_text(s)


def match_accuracy(predictions: list[str], references: list[str]) -> float:
	"""Exact-match accuracy (SQuAD-normalized): fraction of predictions equal to
	the reference after lowercasing, stripping punctuation, and dropping articles.

	Each reference may be a single gold string or a list of acceptable answers
	(aliases); an item counts as correct if the prediction matches ANY of them.

	Items with neither a prediction nor a gold that survives normalization are not
	scored; :func:`scoring_coverage` reports how many those were.
	"""
	pairs = _scoreable_pairs(predictions, references)
	if not pairs:
		return 0.0
	total = 0.0
	for pred, ref in pairs:
		total += max(_em_pair(pred, r) for r in _as_ref_list(ref))
	return total / float(len(pairs))


def _contains_pair(pred_norm: str, ref: str) -> float:
	"""1.0 if the (already normalized) prediction contains this single gold."""
	ref_norm = _normalize_text(ref)
	ref_core = _normalize_text(_strip_parentheticals(ref))
	return 1.0 if (ref_norm and ref_norm in pred_norm) or (ref_core and ref_core in pred_norm) else 0.0


def contains_accuracy(predictions: list[str], references: list[str]) -> float:
	"""Containment accuracy: fraction of predictions that include the reference.

	Paper definition (arXiv:2510.06719v2): "whether the model output includes
	the gold answer / fictitious disease name / top-10 movie title". Each
	reference may be a single gold or a list of acceptable answers (aliases); the
	item counts if the prediction contains ANY of them.

	Scored over the same items as the other pair-comparison metrics, so the whole
	metric block for a cell shares one denominator.
	"""
	pairs = _scoreable_pairs(predictions, references)
	if not pairs:
		return 0.0
	correct = 0.0
	for pred, ref in pairs:
		p = _normalize_text(pred)
		correct += max(_contains_pair(p, r) for r in _as_ref_list(ref))
	return correct / float(len(pairs))


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
		pc = _canonicalize_title(pred)
		for ref in (refs or []):
			rv = _normalize_text(ref)
			rc = _canonicalize_title(ref)
			if (rv and rv in p) or (rc and rc in pc):
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
		Average ROUGE-L F1 score (0.0 to 1.0), over the scoreable items only.
	"""
	pairs = _scoreable_pairs(predictions, references)
	if not pairs:
		return 0.0

	scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
	total = 0.0
	for pred, ref in pairs:
		total += max(float(scorer.score(str(r), str(pred))["rougeL"].fmeasure)
		             for r in _as_ref_list(ref))
	return total / float(len(pairs))


def rouge_1(predictions: list[str], references: list[str]) -> float:
	"""Average ROUGE-1 F1 (unigram overlap) between predictions and references.

	Reported by DP-KSA (Tang et al., arXiv:2602.14374, §6.1) alongside ROUGE-L.
	Averaged over the scoreable items only.
	"""
	pairs = _scoreable_pairs(predictions, references)
	if not pairs:
		return 0.0

	scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
	total = 0.0
	for pred, ref in pairs:
		total += max(float(scorer.score(str(r), str(pred))["rouge1"].fmeasure)
		             for r in _as_ref_list(ref))
	return total / float(len(pairs))


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
	utility metrics. Each reference may be a single gold or a list of acceptable
	answers (aliases); the item scores the max similarity over that set. Averaged
	over the scoreable items only.
	"""
	pairs = _scoreable_pairs(predictions, references)
	if not pairs:
		return 0.0

	def sim(p: str, ref: str) -> float:
		r = _normalize_text(ref)
		# Two empty strings are zero edits apart, which read as a perfect similarity;
		# those pairs are excluded above rather than scored, so what is left with no
		# characters on either side has a gold that normalization emptied and a real
		# prediction (or the reverse), and it scores 0.
		m = max(len(p), len(r))
		return 1.0 - _levenshtein_distance(p, r) / m if m > 0 else 0.0

	total = 0.0
	for pred, ref in pairs:
		p = _normalize_text(pred)
		total += max(sim(p, r) for r in _as_ref_list(ref))
	return total / float(len(pairs))


def f1_score(predictions: list[str], references: list[str]) -> float:
	"""Calculate token-level F1 (SQuAD standard) between predictions and references.

	Tokens are SQuAD-normalized (lowercase, no punctuation, no articles) and
	overlap is counted as a MULTISET (Counter intersection), matching the SQuAD /
	QA-benchmark F1 the DP-KSA and Private-RAG papers report. Each reference may
	be a single gold string or a list of acceptable answers (aliases); the item
	scores the max F1 over that set.

	Args:
		predictions: List of predicted texts.
		references: List of ground truth references (str or list-of-str aliases).

	Returns:
		Average F1 score (0.0 to 1.0), over the scoreable items only.
	"""
	pairs = _scoreable_pairs(predictions, references)
	if not pairs:
		return 0.0

	total_f1 = 0.0
	for pred, ref in pairs:
		total_f1 += max(_token_f1_pair(pred, r) for r in _as_ref_list(ref))

	return total_f1 / float(len(pairs))


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
	# Averaged over the same items as the other pair-comparison metrics: a pair with
	# neither a prediction nor a gold left after normalization is dropped, not counted
	# as a zero, so one denominator describes the cell's whole metric block.
	scoreable = _scoreable_pairs(predictions, references)
	n = len(scoreable)
	if n == 0:
		return 0.0
	predictions = [p for p, _ in scoreable]
	references = [r for _, r in scoreable]

	# Empty/whitespace predictions (e.g. DPVote/SparseVote LD-Gumbel halts -> "")
	# (a) CRASH bert_score's roberta-large tokenizer (build_inputs_with_special_tokens
	# on an empty string), which previously degraded the WHOLE file to NaN, and (b) have
	# zero semantic overlap with any reference anyway. So we score only the non-empty
	# pairs and count each empty prediction as 0.0, averaging over all n. An all-empty
	# file returns 0.0 — never NaN (per request: report 0, not NaN).
	preds = predictions[:n]
	refs = references[:n]
	pairs = [(p, r) for p, r in zip(preds, refs) if p and p.strip()]
	if not pairs:
		return 0.0

	# We request the fast (Rust) tokenizer because with recent transformers the
	# *slow* RobertaTokenizer no longer exposes ``build_inputs_with_special_tokens``,
	# which bert_score calls internally. But if the environment cannot build a fast
	# tokenizer it silently falls back to the slow one and still raises that
	# AttributeError. The import itself can also fail (e.g. transformers refusing to
	# load against an out-of-range tokenizers pin). Rather than aborting the whole
	# offline scoring pass over every result file for one finicky metric, degrade to
	# 0.0 and warn once — all other metrics are still written. The import is inside the
	# try so import-time failures degrade too.
	try:
		from bert_score import BERTScorer

		scorer = _BERTSCORERS.get(lang)
		if scorer is None:
			scorer = BERTScorer(lang=lang, use_fast_tokenizer=True)
			_BERTSCORERS[lang] = scorer

		ne_preds = [p for p, _ in pairs]
		ne_refs = [r for _, r in pairs]
		_, _, F = scorer.score(ne_preds, ne_refs, verbose=False)
		# Sum over the scored (non-empty) preds; empties contribute 0. Mean over all n.
		return float(F.sum().item()) / n
	except Exception as exc:  # noqa: BLE001 — any tokenizer/version failure
		global _BERTSCORE_WARNED
		if not _BERTSCORE_WARNED:
			print(f"  [warn] bert_score_f1 unavailable, recording 0.0: {exc}")
			_BERTSCORE_WARNED = True
		return 0.0


_BERTSCORE_WARNED = False

# One resident scorer per language. ``lang`` alone fixes the underlying model and its
# layer count, and with the default ``idf=False``/``rescale_with_baseline=False`` a
# scorer keeps no state from the files it has already scored — each ``score()`` call
# embeds its own inputs and returns the same numbers whether it is the first or the
# hundredth. Keying on the language also keeps a scorer from being reused under a
# different layer count, which the model loader would apply by truncating in place.
_BERTSCORERS: dict[str, object] = {}


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


_PPL_MODELS: dict[tuple[str, str], tuple] = {}


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

	# The scoring model is read-only here: it stays in eval mode, is never given a
	# gradient, and the loop below feeds each text independently, so the numbers it
	# returns depend only on (model_name, device) and the text. That makes one
	# resident copy per such pair indistinguishable from a fresh load.
	entry = _PPL_MODELS.get((model_name, device))
	if entry is None:
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		if tokenizer.pad_token_id is None:
			tokenizer.pad_token = tokenizer.eos_token
		with cuda_oom_warning(model_name, device):
			model = AutoModelForCausalLM.from_pretrained(model_name)
			model = model.to(device)
		model.eval()
		entry = (tokenizer, model)
		_PPL_MODELS[(model_name, device)] = entry
	tokenizer, model = entry

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


def answer_rate(predictions: list[str]) -> float:
	"""Fraction of queries the method actually answered.

	Four of the benchmark's methods degenerate to an empty string in some regime —
	the sparse vote halts, the adaptive budget switches every document off, the
	propose-test-release step declines to release, the self-filter empties the
	synthetic corpus. Averaged into an overlap metric those empties score zero and
	become indistinguishable from a wrong answer, which reads as a method that is
	merely worse rather than one that has stopped responding. Reporting the rate
	separately is what tells the two apart, and it also makes the accuracy among
	answered queries recoverable as ``accuracy / answer_rate``.
	"""
	if not predictions:
		return float("nan")
	answered = sum(1 for p in predictions if str(p).strip())
	return answered / len(predictions)


def private_answer_rate(
	predictions: list[str], private_tokens_per_query: list[int]
) -> float:
	"""Fraction of answers carrying at least one privately released token.

	``answer_rate`` counts an answer as given whenever the string is non-empty, which
	says nothing about which mechanism produced it. DPSparseVoteRAG routes every token
	the sparse test leaves below threshold down a public, budget-free path, so a cell
	whose Limited-Domain vote can never fire still returns short answers -- and those
	answers score. Read next to a method whose answers are private token by token, that
	is free apparent utility measured exactly where the mechanism is off.

	This is the same count restricted to answers the private mechanism actually
	contributed to. Where it is 0.0 the cell's text metrics describe a no-context
	language model, not a differentially private one, and the two columns together are
	what say so.
	"""
	if not predictions:
		return float("nan")
	if len(predictions) != len(private_tokens_per_query):
		raise ValueError(
			f"predictions and private_tokens_per_query must be aligned per query, got "
			f"{len(predictions)} and {len(private_tokens_per_query)}."
		)
	private = sum(
		1
		for pred, n in zip(predictions, private_tokens_per_query)
		if int(n) > 0 and str(pred).strip()
	)
	return private / len(predictions)


def empty_rate(predictions: list[str]) -> float:
	"""Fraction of queries left unanswered; the complement of :func:`answer_rate`."""
	rate = answer_rate(predictions)
	return rate if np.isnan(rate) else 1.0 - rate


def retrieval_precision_at_k(
	retrieved_ids: list[list],
	relevant_ids: list[list],
	k: int = 50,
) -> float:
	"""Overlap between what a method retrieved and the non-private top-k.

	There is no ground truth here for which documents a question needed. What there is
	is the same index queried without any mechanism, so the quantity is agreement with
	the ranking the method would have produced had it not been private -- not relevance,
	and not correctness. Read that way it is the one number that makes budget exhaustion
	visible: once a per-document budget is spent the document stops being retrievable,
	and the overlap collapses while the generator keeps producing fluent answers.

	The overlap is over the retrieved SET, not a prefix of it, because a method that
	releases a set has no ranking to take a prefix of -- DP-RAG deliberately shuffles
	what it returns, since ordering it by score would leak the scores the mechanism
	hides. Taking a prefix of a shuffled set measured the shuffle: at ~62 released
	documents it scored 0.08 for a method whose real overlap with the non-private top-5
	was essentially total.

	A method whose retrieval carries no noise scores 1.0 by construction. That is
	correct and it is not an achievement, which is why the payload records whether the
	retrieval was private at all; the two must be read together.
	"""
	if not retrieved_ids or not relevant_ids:
		return float("nan")
	if len(retrieved_ids) != len(relevant_ids):
		raise ValueError(
			f"retrieved_ids and relevant_ids must be aligned per query, got "
			f"{len(retrieved_ids)} and {len(relevant_ids)}."
		)
	scores = []
	for retrieved, relevant in zip(retrieved_ids, relevant_ids):
		gold = {str(i) for i in (relevant or [])[:k]}
		if not gold or retrieved is None:
			# Nothing to score against, or the run never recorded what it retrieved.
			# Scoring that as 0.0 would publish "this method lost all of its
			# retrieval" for what is in fact a missing measurement.
			scores.append(float("nan"))
			continue
		# The whole released set, and the denominator is k. Dividing by what came
		# back would score a method that returns a single document — because every
		# other one has spent its budget — as perfect, turning the collapse this
		# metric exists to expose into a top mark.
		got = {str(i) for i in (retrieved or [])}
		scores.append(len(got & gold) / k)
	valid = [s for s in scores if not np.isnan(s)]
	if not valid:
		return float("nan")
	return float(np.mean(valid))


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
		Dictionary mapping metric names to their computed values, plus ``n_scored``
		and ``n_unscoreable`` — the denominator the pair-comparison metrics used and
		the number of items they had to drop.

	Raises:
		ValueError: If an unsupported metric name is provided.
	"""
	# ``alias_sets`` (list[list[str]]) gives per-item acceptable answers (aliases),
	# e.g. TriviaQA's answer.aliases or NQ-open's multiple golds. When present, the
	# text-overlap metrics score max-over-aliases; bert/relevancy stay on the single
	# ``references`` (they don't take a set). Distinct from ``reference_sets`` which
	# feeds hit_rate (MovieLens favorites) with a different semantic.
	alias_sets = kwargs.get("alias_sets")
	_ALIAS_AWARE = {"match_accuracy", "contains_accuracy", "rouge_l",
	                "rouge_1", "levenshtein", "f1_score"}

	def _refs(k: str):
		return alias_sets if (alias_sets is not None and k in _ALIAS_AWARE) else references

	keys = [str(name).strip().lower() for name in metric_names]

	metrics: dict[str, float] = {}
	# How many items the pair-comparison metrics actually averaged over, reported
	# alongside them whether or not it was asked for. A cell where a mechanism halted
	# on every question can hold items with nothing on either side; those are dropped
	# rather than scored, and a mean over a smaller denominator must not be readable as
	# a mean over the full question set. Computed against the same references those
	# metrics use, and only when at least one of them was requested — a call that asks
	# only for, say, hit_rate has no such denominator to report.
	if predictions and (_PAIR_COMPARISON_METRICS & set(keys)):
		metrics.update(scoring_coverage(predictions, _refs("match_accuracy")))

	for key in keys:
		if key in _COVERAGE_METRICS:
			metrics.update(scoring_coverage(predictions, _refs("match_accuracy")))
		elif key == "match_accuracy":
			metrics[key] = match_accuracy(predictions, _refs(key))
		elif key == "contains_accuracy":
			metrics[key] = contains_accuracy(predictions, _refs(key))
		elif key == "rouge_l":
			metrics[key] = rouge_l(predictions, _refs(key))
		elif key == "hit_rate":
			reference_sets = kwargs.get("reference_sets", [])
			metrics[key] = hit_rate(predictions, reference_sets)
		elif key == "rouge_1":
			metrics[key] = rouge_1(predictions, _refs(key))
		elif key == "levenshtein":
			metrics[key] = levenshtein(predictions, _refs(key))
		elif key == "f1_score":
			metrics[key] = f1_score(predictions, _refs(key))
		elif key == "bert_score_f1":
			lang = str(kwargs.get("lang", "en"))
			metrics[key] = bert_score_f1(predictions, references, lang=lang)
		elif key == "answer_rate":
			metrics[key] = answer_rate(predictions)
		elif key == "empty_rate":
			metrics[key] = empty_rate(predictions)
		elif key == "recall_at_k":
			retrieved_ids = kwargs.get("retrieved_ids", [])
			relevant_ids = kwargs.get("relevant_ids", [])
			metrics[key] = recall_at_k(retrieved_ids, relevant_ids)
		elif key == "retrieval_precision_at_k":
			retrieved_ids = kwargs.get("retrieved_ids", [])
			relevant_ids = kwargs.get("relevant_ids", [])
			metrics[key] = retrieval_precision_at_k(
				retrieved_ids, relevant_ids, k=int(kwargs.get("precision_k", 50))
			)
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
			raise ValueError(f"Utility metric '{key}' is not supported.")
	return metrics
