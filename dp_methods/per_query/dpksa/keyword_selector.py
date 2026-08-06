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

import re
from collections import Counter
from collections.abc import Callable
from statistics import NormalDist

import numpy as np

# DPKSA-12: the paper counts at the "token" level; we operationalise a token as a
# normalised whitespace word (runs of alphanumerics). Defensible — QA answers are
# 1-4 such words and they re-inject cleanly into the final keyword prompt.
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


def padded_gap(counts: list[int], k: int) -> int:
    """Count gap ``d_k = H_(k) - H_(k+1)`` with the sorted histogram implicitly
    zero-padded (paper §4.2: ``H_(k) = 0`` beyond the observed tokens), so ``d_k``
    is defined for every ``k >= 1`` regardless of how many distinct tokens appear."""
    hk = counts[k - 1] if 1 <= k <= len(counts) else 0
    hk1 = counts[k] if 1 <= k + 1 <= len(counts) else 0
    return int(hk - hk1)


def find_best_k(
    counts: list[int],
    *,
    gumbel_scale: float,
    k_min: int,
    k_max: int,
    rng: np.random.Generator,
    gumbel: Callable[[float], float] | None = None,
) -> tuple[int, int]:
    """FindBestK: pick ``k_hat`` via the Gumbel-max exponential mechanism.

    ``k_hat = argmax_k {d_k + r(k) + Gumbel(b)}`` over the **fixed, public** range
    ``k in [k_min, k_max]`` (paper Alg. 3: ``k = 1..N-1`` with the regularizer
    ``r(k)`` zero inside the window and ``-inf`` outside). The candidate set does
    NOT depend on the private data: gaps are read off the zero-padded sorted
    histogram, so every ``k`` in the window has a defined (possibly zero) gap even
    with few distinct tokens. Returns ``(k_hat, d_khat)`` with ``d_khat`` the
    noiseless gap at ``k_hat``.

    ``gumbel`` overrides how a draw is taken, for a caller that wants to observe
    the stream (``core.noise_trace``). It must be one Gumbel(0, scale) sample per
    call on ``rng``, so the sequence of draws — and hence ``k_hat`` — is the same
    either way; ``rng`` is then unused directly.
    """
    lo = max(1, int(k_min))
    hi = max(lo, int(k_max))
    draw = gumbel if gumbel is not None else (lambda scale: rng.gumbel(0.0, scale))

    best_k = lo
    best_score = float("-inf")
    for k in range(lo, hi + 1):
        score = padded_gap(counts, k) + float(draw(gumbel_scale))
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, padded_gap(counts, best_k)


def ptr_threshold(sigma: float, delta_ptr: float) -> float:
    """``Phi(1 - delta_ptr; 0, 2*sigma)`` — the (1 - delta_ptr) quantile of a
    centered Gaussian with standard deviation ``2*sigma`` (exact, via NormalDist)."""
    return NormalDist(0.0, 2.0 * sigma).inv_cdf(1.0 - delta_ptr)


def top_k_with_ptr(
    d_khat: int,
    *,
    sigma: float,
    delta_ptr: float,
    rng: np.random.Generator,
) -> bool:
    """TopKWithPTR test. Returns True iff the exact top-k may be released.

    `max(2, d_khat)` is 1-Lipschitz in d_khat, so it does not increase the gap
    sensitivity (Δd_k = 2): the Gaussian noise scale 2*sigma already absorbs that
    sensitivity, keeping the test Gaussian-PTR-valid.
    """
    noise = float(rng.normal(0.0, 2.0 * sigma))
    d_hat = max(2.0, float(d_khat)) + noise - ptr_threshold(sigma, delta_ptr)
    return d_hat > 2.0
