"""DP keyword selection for DP-KSA (Tang et al., arXiv:2602.14374).

Given the per-document responses produced by the sample-and-aggregate stage, this
module builds the token-frequency histogram and runs the two DP mechanisms that
release the keyword set:

- :func:`build_token_histogram` — ``H[w] = #{i : w in y_i}`` (number of responses
  whose token set contains ``w``). The paper counts at the token level; we
  operationalise a "token" as a normalised whitespace word, which is the unit that
  yields keywords that can be re-injected into the final prompt (QA answers are
  1-4 such tokens).
- :func:`find_best_k` — FindBestK: ``k_hat = argmax_k {d_k + r(k) + Gumbel(b)}``
  over the count gaps ``d_k = H_(k) - H_(k+1)``. ``r(k)`` restricts the search to
  ``[k_min, k_max]`` (``-inf`` outside).
- :func:`top_k_with_ptr` — TopKWithPTR: a Gaussian Propose-Test-Release test
  ``d_hat = max(2, d_k) + N(0, (2*sigma)^2) - Phi(1 - delta_ptr; 0, 2*sigma)``.
  If ``d_hat > 2`` the exact top-``k_hat`` keywords are released, otherwise none
  (zero-shot fallback).
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# Minimal English stopword set, used only when exclude_stopwords=True. The paper
# does not remove stopwords; this is an optional utility knob.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "it", "this", "that", "with", "as", "at", "by",
    "from", "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
}


def normalize_tokens(
    text: str,
    *,
    lowercase: bool = True,
    exclude_stopwords: bool = False,
) -> list[str]:
    """Split text into normalised word tokens."""
    if lowercase:
        text = text.lower()
    tokens = _WORD_RE.findall(text)
    if exclude_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


def build_token_histogram(
    responses: list[str],
    *,
    lowercase: bool = True,
    exclude_stopwords: bool = False,
) -> dict[str, int]:
    """Histogram ``H[w]`` = number of responses whose token set contains ``w``.

    Counting unique tokens *per response* (set membership) is what bounds the
    sensitivity: changing one response moves each count by at most 1.
    """
    counter: Counter[str] = Counter()
    for response in responses:
        unique = set(
            normalize_tokens(
                response, lowercase=lowercase, exclude_stopwords=exclude_stopwords
            )
        )
        counter.update(unique)
    return dict(counter)


def sorted_counts_and_tokens(histogram: dict[str, int]) -> tuple[list[int], list[str]]:
    """Return counts (descending) and the matching tokens, with a deterministic
    tie-break by token text so the order is reproducible."""
    items = sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0]))
    counts = [int(c) for _, c in items]
    tokens = [t for t, _ in items]
    return counts, tokens


def gaps(counts: list[int]) -> list[int]:
    """Count gaps ``d_k = H_(k) - H_(k+1)`` for ``k = 1 .. len(counts) - 1``
    (returned 0-indexed: ``gaps()[k-1]`` is ``d_k``)."""
    return [counts[i] - counts[i + 1] for i in range(len(counts) - 1)]


def find_best_k(
    counts: list[int],
    *,
    gumbel_scale: float,
    k_min: int,
    k_max: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """FindBestK: pick ``k_hat`` via the Gumbel-max exponential mechanism.

    Returns ``(k_hat, d_khat)`` where ``d_khat`` is the (noiseless) gap at ``k_hat``.
    The regularizer ``r(k)`` is ``0`` for ``k in [k_min, k_max]`` and ``-inf``
    elsewhere, i.e. the argmax ranges over that window only.
    """
    g = gaps(counts)
    if not g:
        return 0, 0  # fewer than 2 distinct tokens: nothing to cut

    hi = min(k_max, len(g))           # k ranges over 1..len(g)
    lo = max(1, k_min)
    if lo > hi:
        lo = 1
        hi = len(g)

    best_k = lo
    best_score = -math.inf
    for k in range(lo, hi + 1):
        noisy = g[k - 1] + float(rng.gumbel(0.0, gumbel_scale))
        if noisy > best_score:
            best_score = noisy
            best_k = k
    return best_k, int(g[best_k - 1])


def ptr_threshold(sigma: float, delta_ptr: float) -> float:
    """``Phi(1 - delta_ptr; 0, 2*sigma)`` — the (1 - delta_ptr) quantile of a
    centered Gaussian with standard deviation ``2*sigma``."""
    # Inverse standard normal CDF via the error function (no SciPy dependency).
    z = math.sqrt(2.0) * _erfinv(2.0 * (1.0 - delta_ptr) - 1.0)
    return z * (2.0 * sigma)


def top_k_with_ptr(
    d_khat: int,
    *,
    sigma: float,
    delta_ptr: float,
    rng: np.random.Generator,
) -> bool:
    """TopKWithPTR test. Returns True iff the exact top-k may be released."""
    noise = float(rng.normal(0.0, 2.0 * sigma))
    d_hat = max(2.0, float(d_khat)) + noise - ptr_threshold(sigma, delta_ptr)
    return d_hat > 2.0


def _erfinv(x: float) -> float:
    """Inverse error function (Winitzki approximation + one Newton step)."""
    if x <= -1.0:
        return -math.inf
    if x >= 1.0:
        return math.inf
    if x == 0.0:
        return 0.0
    a = 0.147
    ln = math.log(1.0 - x * x)
    term = 2.0 / (math.pi * a) + ln / 2.0
    y = math.copysign(math.sqrt(math.sqrt(term * term - ln / a) - term), x)
    # One Newton refinement step against erf.
    y -= (math.erf(y) - x) / (2.0 / math.sqrt(math.pi) * math.exp(-y * y))
    return y
