"""Membership inference attacks (MIA) against the retrieval corpus.

Two black-box attacks, both scored over a real member / non-member document
split (members are documents present in the retrieval index; non-members are
comparable documents deliberately held out):

- **S2MIA** (Li et al., *Generating is Believing*, arXiv:2406.19234), as adopted
  by DPVoteRAG (Koga et al., arXiv:2412.04697). The target document is split into
  a query part and a held-out part; the query is answered by the RAG system and
  the membership score is ``BLEU(answer, held-out part)``. Members reproduce their
  own held-out text (the index contains it) and score higher than non-members.
  We use the BLEU-only membership score of DPVoteRAG, not the BLEU+perplexity
  combination of the original S2MIA paper.

- **RAG-MIA** (Anderson et al., *Is My Data in Your Retrieval Database?*,
  arXiv:2405.20446), black-box variant. The system is asked whether the target
  document appears in the context; a "Yes" answer predicts membership.

Both are quantified by the ROC-AUC over the true membership labels: 0.5 means the
attack cannot distinguish members from non-members (empirical privacy), 1.0 means
perfect membership recovery (full leakage).

The attacks run **online** (they issue fresh probe queries through the live
pipeline, in ``experiments/mia_probe.py``) but are **scored offline** here from
the saved per-probe records, mirroring the deferred-metric design of the utility
metrics. Each probe record is a dict with keys:
``is_member`` (bool), ``gold`` (held-out reference text), ``doc`` (full target
document), ``s2mia_answer`` (RAG answer to the S2MIA query) and
``anderson_answer`` (RAG answer to the Yes/No membership question).
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from scipy.stats import rankdata


def bleu_similarity(prediction: str, reference: str) -> float:
    """BLEU (smoothed) of ``prediction`` against ``reference`` in ``[0, 1]``.

    Returns 0.0 when either side is empty so that degenerate/empty generations do
    not spuriously score as matches.
    """
    pred_tokens = str(prediction).strip().split()
    ref_tokens = str(reference).strip().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    smooth = SmoothingFunction().method1
    return float(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth))


def max_bleu_against_corpus(text: str, corpus_tokens: Sequence[Sequence[str]]) -> float:
    """Maximum BLEU of ``text`` over a pre-tokenised ``corpus_tokens``.

    Used for the DP-SynRAG *retention* score: how closely the closest generated
    (synthetic) document reproduces a target input document.
    """
    text_tokens = str(text).strip().split()
    if not text_tokens or not corpus_tokens:
        return 0.0
    smooth = SmoothingFunction().method1
    return max(
        float(sentence_bleu([ref], text_tokens, smoothing_function=smooth))
        for ref in corpus_tokens
    )


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-aware ROC-AUC (Mann-Whitney) of ``scores`` against binary ``labels``.

    Average ranks make each tie contribute exactly 0.5, so many equal scores
    (e.g. all-zero BLEU on short outputs) settle the AUC at 0.5 instead of an
    index-order artifact. Returns 0.5 when either class is empty.
    """
    labels_arr = np.asarray(list(labels), dtype=int)
    scores_arr = np.asarray(list(scores), dtype=float)
    pos = int((labels_arr == 1).sum())
    neg = int((labels_arr == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5
    ranks = rankdata(scores_arr, method="average")
    rank_sum_pos = float(ranks[labels_arr == 1].sum())
    auc = (rank_sum_pos - (pos * (pos + 1) / 2.0)) / (pos * neg)
    return float(max(0.0, min(1.0, auc)))


def tpr_at_fpr(labels: Sequence[int], scores: Sequence[float], target_fpr: float = 0.01) -> float:
    """Maximum true-positive rate achievable at false-positive rate <= ``target_fpr``.

    The low-FPR operating point reported alongside AUC in the MIA literature; a
    high value means a few members can be identified with near-certainty even when
    the overall AUC is modest. Returns 0.0 when either class is empty.
    """
    labels_arr = np.asarray(list(labels), dtype=int)
    scores_arr = np.asarray(list(scores), dtype=float)
    pos_mask = labels_arr == 1
    neg_mask = labels_arr == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    best = 0.0
    for threshold in np.unique(scores_arr):
        predicted = scores_arr >= threshold
        fpr = float((predicted & neg_mask).sum()) / n_neg
        if fpr <= target_fpr:
            tpr = float((predicted & pos_mask).sum()) / n_pos
            best = max(best, tpr)
    return best


def yes_no_score(text: str) -> float:
    """Anderson black-box membership signal: 1.0 if the answer says "Yes", else 0.0.

    The first standalone ``yes``/``no`` token decides. Answers containing neither
    are classified as non-member (0.0), following the paper's rule that outputs
    without a Yes/No token are treated as non-members.
    """
    for word in re.findall(r"[a-z]+", str(text).lower()):
        if word == "yes":
            return 1.0
        if word == "no":
            return 0.0
    return 0.0


def _labels_scores(
    records: Sequence[dict], score_fn: Callable[[dict], float]
) -> tuple[list[int], list[float]]:
    labels = [1 if r.get("is_member") else 0 for r in records]
    scores = [float(score_fn(r)) for r in records]
    return labels, scores


def score_mia_records(
    records: Sequence[dict], synthetic_texts: Sequence[str] | None = None
) -> dict[str, float]:
    """Compute all MIA AUCs from saved probe records.

    Args:
        records: per-probe dicts (see module docstring for keys).
        synthetic_texts: the generated corpus (DP-SynRAG only). When given, adds a
            ``synth_retention_auc`` that tests whether member *input* documents are
            reproduced in the generated corpus more than held-out ones.

    Returns:
        Dict of metric name -> value. ``s2mia_auc`` is the primary AUC-MIA.
        Empty dict when ``records`` is empty.
    """
    if not records:
        return {}

    metrics: dict[str, float] = {}

    s2_labels, s2_scores = _labels_scores(
        records, lambda r: bleu_similarity(r.get("s2mia_answer", ""), r.get("gold", ""))
    )
    metrics["s2mia_auc"] = roc_auc(s2_labels, s2_scores)
    metrics["s2mia_tpr_at_fpr01"] = tpr_at_fpr(s2_labels, s2_scores, 0.01)

    a_labels, a_scores = _labels_scores(
        records, lambda r: yes_no_score(r.get("anderson_answer", ""))
    )
    metrics["rag_mia_auc"] = roc_auc(a_labels, a_scores)

    if synthetic_texts:
        corpus_tokens = [str(t).split() for t in synthetic_texts if str(t).strip()]
        if corpus_tokens:
            r_labels, r_scores = _labels_scores(
                records, lambda r: max_bleu_against_corpus(r.get("doc", ""), corpus_tokens)
            )
            metrics["synth_retention_auc"] = roc_auc(r_labels, r_scores)

    return metrics
