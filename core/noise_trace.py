"""A per-query signature of the noise a mechanism drew.

The benchmark reports error bars over repetitions of the same workload, and those
bars only mean anything if ``run_seed`` moves the mechanism's randomness while
``data_seed`` holds the corpus and the questions fixed. Nothing in a finished run
shows that. Two seeds that produce the same answer are ambiguous: either the noise
never varied, or it varied and the model rounded it away — a small model that loops
on one phrase makes every voter propose the same token, so the vote histogram is a
single spike and no noise draw can move the argmax.

The fingerprint settles it. Each query records the *first* noise sample its RNG
produces, hashed to a short digest. It differs across seeds exactly when the seed
reaches the sampler, whether or not the noise changed a single decision.

Two properties make it sound as an instrument:

* **Data-independent.** The first sample of a query is drawn before the mechanism
  has looked at any count: the vote histograms are built from a forward pass that
  happens after the RNG is seeded, and the document shuffle uses a separate Python
  RNG; DP-KSA's per-document generation is a greedy decode that draws nothing here,
  so its first draw is the opening Gumbel of FindBestK, taken over a fixed public
  window of ``k``. Its distribution and scale come only from public parameters (epsilon, delta,
  the threshold), so the value is a function of the seed and the public query text
  alone. No count, no partition and no document enters it — it is not a release
  about the private data at all, so it costs nothing in the privacy accounting.
* **Free of extra randomness.** Nothing here samples. The fingerprint is read off
  the draw the mechanism was already going to make, so the noise stream keeps its
  exact position and a fixed seed still emits exactly the same tokens.

**It is nevertheless off by default, and must stay that way.** Data-independence is
not the whole privacy question. The fingerprint is derived from the mechanism's own
noise, and the DP guarantee rests on that noise staying secret. Hashing does not make
it secret: a float64 draw at a *publicly known* scale carries only tens of bits of
usable entropy, so a 64-bit digest is wide enough to identify a unique preimage, and
an adversary can recover the sample by searching plausible values at that scale. For
DPVoteRAG the first draw is the Gumbel in the limited-domain bottom-out threshold
``v_perp``; an adversary who recovers it and also sees whether the query released a
token or halted learns whether ``v_max >= v_perp`` against a now-known threshold,
which sharpens their knowledge of the true maximum vote count. For DP-KSA it is the
first FindBestK Gumbel, the perturbation added to the count gap at the smallest ``k``
in the window; an adversary who recovers it and sees the released ``k_hat`` learns
that this gap plus a now-known offset lost the argmax, which bounds a gap of the
private histogram. Small, but a real erosion of the guarantee — and the grid runs behind the published numbers must carry
nothing derived from the noise stream.

So the fingerprint is emitted only under ``BENCHDP_NOISE_FINGERPRINT=1``, which the
smoke path sets for itself (``experiments/runner.py --smoke``). Do not make it the
default, and do not persist it from a run whose artifacts are published alongside a
DP claim.
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import Any

import numpy as np

#: Diagnostic switch. Off unless explicitly set: see the module docstring for why a
#: hashed noise sample is not safe to ship with a run that makes a DP claim.
FINGERPRINT_ENV = "BENCHDP_NOISE_FINGERPRINT"


def fingerprint_enabled() -> bool:
    """True when the diagnostic mode is explicitly on."""
    return os.environ.get(FINGERPRINT_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _digest(value: float) -> str:
    """A short, stable digest of one float.

    The draw is hashed rather than published: a raw sample would hand a reader a
    piece of the noise stream, and the fingerprint only ever has to be compared for
    equality across seeds.
    """
    return hashlib.sha256(struct.pack("<d", float(value))).hexdigest()[:16]


class NoiseFingerprintMixin:
    """Records the first noise sample of a query as a short hash.

    Expects the host generator to expose ``self._np_rng``. Call
    :meth:`_reset_noise_fingerprint` right after the per-query reseed, then draw
    noise through :meth:`_gumbel` / :meth:`_laplace` instead of touching the
    generator directly.
    """

    _noise_fingerprint: str | None = None
    #: Read once per query, so a run cannot half-record: either every query of the
    #: query loop carries a fingerprint or none of them does.
    _noise_fingerprint_on: bool = False

    @property
    def noise_fingerprint(self) -> str | None:
        """The digest for the query just run.

        None when the diagnostic mode is off — the default — and also when the query
        drew no noise at all.
        """
        return self._noise_fingerprint

    def _reset_noise_fingerprint(self) -> None:
        self._noise_fingerprint = None
        self._noise_fingerprint_on = fingerprint_enabled()

    def _record_first_noise(self, draw: Any) -> Any:
        """Fingerprint ``draw`` if it is the query's first, and return it unchanged."""
        if self._noise_fingerprint_on and self._noise_fingerprint is None:
            flat = np.ravel(np.asarray(draw, dtype=np.float64))
            if flat.size:
                self._noise_fingerprint = _digest(float(flat[0]))
        return draw

    def _gumbel(self, scale: float, size: int | None = None):
        """Gumbel(0, scale), one call on the same RNG the mechanism already used."""
        if size is None:
            draw = self._np_rng.gumbel(0.0, float(scale))
        else:
            draw = self._np_rng.gumbel(0.0, float(scale), size=size)
        return self._record_first_noise(draw)

    def _laplace(self, scale: float) -> float:
        """Laplace(0, scale), one call on the same RNG the mechanism already used."""
        return float(self._record_first_noise(float(self._np_rng.laplace(0.0, float(scale)))))
