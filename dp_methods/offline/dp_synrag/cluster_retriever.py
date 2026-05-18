"""DP-SynRAG Stage 1c: per-cluster noisy mean embedding + DP threshold.

Reference: arXiv:2510.06719v2, Section "Stage 1: Soft Clustering",
sub-step "Embedding-based Retrieval".

For each soft cluster C_r returned by DPHistogramClusterer, we:

    1. Compute embeddings E(d_i) for every document d_i in C_r and L2-normalize
       them, so that ||E(d_i)|| = 1 (matching the sensitivity assumption of
       the paper's Gaussian mechanism).
    2. Form a noisy sum:
           mu(C_r) = sum_{d_i in C_r} E(d_i) + N(0, sigma_mu^2 I_d).
    3. Score each document by its cosine-style similarity to mu(C_r):
           s_i = E(d_i) . mu(C_r).
    4. Sample a threshold theta_s via the continuous exponential mechanism on
       the top-k utility:
           u(theta_s) = -| #{i : s_i >= theta_s} - k |.
       Sensitivity of u is 1, so the mechanism uses exp(eps * u / 2).
    5. Return S_r = { i in C_r : s_i > theta_s }.

Sensitivity notes:
    - mu uses the Gaussian mechanism with L2 sensitivity 1 (paper assumes
      ||E(d)|| <= 1 and add/remove neighboring datasets), giving
      rho_mu = 1 / (2 sigma_mu^2).
    - The threshold mechanism is pure ε-DP, contributing
      rho_theta_s = 1 / (8 epsilon_theta_s^2) in the zCDP composition.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


class DPClusterRetriever:
    """Stage 1c: noisy centroid + DP threshold to retrieve S_r within each C_r."""

    def __init__(
        self,
        embedder: Any,
        k: int,
        sigma_mu: float,
        epsilon_theta_s: float,
        min_sim: float = -1.0,
        max_sim: float = 1.0,
        seed: int = 42,
        verbose: bool = False,
    ):
        if k <= 0:
            raise ValueError("k must be >= 1.")
        if sigma_mu < 0:
            raise ValueError("sigma_mu must be >= 0.")
        if epsilon_theta_s <= 0:
            raise ValueError("epsilon_theta_s must be > 0.")
        if min_sim >= max_sim:
            raise ValueError("min_sim must be < max_sim.")

        self.embedder = embedder
        self.k = int(k)
        self.sigma_mu = float(sigma_mu)
        self.epsilon_theta_s = float(epsilon_theta_s)
        self.min_sim = float(min_sim)
        self.max_sim = float(max_sim)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.last_stats: dict = {}

    def retrieve_clusters(
        self,
        documents: list[str],
        clusters: list[list[int]],
    ) -> list[list[int]]:
        """Return S_r for each non-trivial cluster (empty list for trivial ones)."""
        subsets: list[list[int]] = [[] for _ in clusters]
        kept_sizes: list[int] = []

        for r, members in enumerate(clusters):
            if len(members) < 2:
                # Cannot apply an exp-mech threshold with <2 candidates;
                # forward the singleton (if any) so the cluster still has content.
                subsets[r] = list(members)
                kept_sizes.append(len(members))
                continue

            E_r = self._embed_normalized([documents[i] for i in members])
            d = E_r.shape[1]

            rng = self._rng_for_cluster(r)
            noise = rng.normal(loc=0.0, scale=self.sigma_mu, size=d).astype(np.float32)
            mu_r = E_r.sum(axis=0) + noise

            mu_norm = np.linalg.norm(mu_r)
            if mu_norm > 1e-10:
                mu_r = mu_r / mu_norm

            sims = (E_r @ mu_r).astype(np.float64)
            theta_s = self._sample_threshold(sims, rng)

            chosen_local = np.flatnonzero(sims > theta_s).tolist()
            if not chosen_local:
                chosen_local = [int(np.argmax(sims))]

            subsets[r] = [members[i] for i in chosen_local]
            kept_sizes.append(len(subsets[r]))

        self.last_stats = {
            "k": int(self.k),
            "sigma_mu": float(self.sigma_mu),
            "epsilon_theta_s": float(self.epsilon_theta_s),
            "n_clusters_input": len(clusters),
            "n_clusters_with_subset": sum(1 for s in subsets if s),
            "mean_subset_size": (
                float(np.mean(kept_sizes)) if kept_sizes else 0.0
            ),
        }
        return subsets

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_normalized(self, texts: list[str]) -> np.ndarray:
        vecs = np.asarray(self.embedder.embed(texts), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype(np.float32)

    def _rng_for_cluster(self, r: int) -> np.random.Generator:
        # Per-cluster RNG so independent runs are reproducible.
        key = f"{self.seed}|{r}".encode("utf-8")
        offset = int(hashlib.sha256(key).hexdigest()[:8], 16)
        return np.random.default_rng(self.seed + offset)

    def _sample_threshold(
        self, sims: np.ndarray, rng: np.random.Generator
    ) -> float:
        """Continuous exponential mechanism for the top-k threshold."""
        sorted_sims = np.sort(sims.astype(np.float64))
        n_real_sims = len(sorted_sims)
        sorted_sims = np.insert(sorted_sims, 0, self.min_sim)
        sorted_sims = np.append(sorted_sims, self.max_sim)
        sorted_sims = np.clip(sorted_sims, self.min_sim, self.max_sim)

        # Utility per left endpoint: #{sims >= theta_s} - k
        # counts_above[j] = number of REAL sims >= sorted_sims[j]
        # Real sims are at array indices 1..n_real_sims (after min prepend).
        # For θ_s > sorted_sims[j], sims >= θ_s are those at indices > j.
        # Among real sims (1..n_real_sims), those with index > j gives n_real_sims - j count.
        counts_above = n_real_sims - np.arange(len(sorted_sims))
        utilities = -np.abs(counts_above - self.k)
        delta = np.diff(sorted_sims)

        # Numerical stability: shift utilities so the max exponent is 0.
        max_u = float(utilities[:-1].max())
        log_weights = self.epsilon_theta_s * (utilities[:-1] - max_u) / 2.0
        pdf = np.exp(log_weights) * delta
        s = float(pdf.sum())
        if not np.isfinite(s) or s <= 0:
            if self.verbose:
                print(
                    "[DP-SynRAG] threshold pdf degenerate; falling back to median."
                )
            return float(sorted_sims[len(sorted_sims) // 2])
        pdf /= s
        return float(rng.choice(sorted_sims[:-1], p=pdf))
