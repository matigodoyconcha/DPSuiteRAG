"""DP-SynRAG Stage 1a (histogram) + 1b (soft clustering).

Reference: arXiv:2510.06719v2, Section "Stage 1: Soft Clustering".

Given the binary extraction matrix Ext in {0, 1}^(N x |V|) produced by
KeywordExtractor, this module:

    1. Computes h(D) = sum_i Ext_i in R^|V|
    2. Adds isotropic Gaussian noise:
           h'(D) = h(D) + N(0, sigma_h^2 I_|V|)
    3. Selects the top-R keywords by noisy frequency.
    4. Assigns documents to up to L clusters anchored on those keywords,
       processing keywords in reverse frequency order (paper Stage 1b):

           for r = R, ..., 1:
               C_r = { d_i : w_r in d_i,
                       sum_{r' > r} 1[d_i in C_{r'}] < L }

       The reverse-order rule guarantees that each document lands in at most L
       clusters and that low-frequency anchor keywords absorb their matches
       first, before high-frequency keywords take over the remaining capacity.

Privacy sensitivity:
    Adding/removing one document changes Ext by one row, which has L2 norm
    sqrt(K). The Gaussian mechanism with variance sigma_h^2 per coordinate
    therefore gives zCDP value rho_h = K / (2 sigma_h^2) (paper formula).
"""

from __future__ import annotations

import numpy as np


class DPHistogramClusterer:
    """Stage 1a + 1b: noisy keyword histogram + reverse-order soft clustering."""

    def __init__(
        self,
        R: int,
        L: int,
        sigma_h: float,
        seed: int = 42,
    ):
        if R <= 0:
            raise ValueError("R must be >= 1.")
        if L <= 0:
            raise ValueError("L must be >= 1.")
        if sigma_h < 0:
            raise ValueError("sigma_h must be >= 0.")

        self.R = int(R)
        self.L = int(L)
        self.sigma_h = float(sigma_h)
        self.seed = int(seed)

        self.last_stats: dict = {}

    def cluster(
        self, extractions: np.ndarray
    ) -> tuple[np.ndarray, list[list[int]]]:
        """Compute noisy top-R keywords and the L-bounded soft cluster assignment.

        Args:
            extractions: binary matrix of shape (N, |V|).

        Returns:
            top_R_idx: int array of length R with the indices of the chosen
                keywords in descending order of noisy frequency.
            clusters: list of length R; clusters[r] is the list of document
                indices assigned to cluster r (anchored on top_R_idx[r]).
        """
        if extractions.ndim != 2:
            raise ValueError(
                f"extractions must be 2D (N, |V|), got shape {extractions.shape}."
            )
        N, V = extractions.shape
        if V == 0:
            raise ValueError("Empty vocabulary.")
        if self.R > V:
            raise ValueError(
                f"R={self.R} exceeds vocabulary size |V|={V}."
            )

        rng = np.random.default_rng(self.seed)

        # Stage 1a: noisy histogram.
        h = extractions.sum(axis=0).astype(np.float64)
        if self.sigma_h > 0:
            h_noisy = h + rng.normal(loc=0.0, scale=self.sigma_h, size=V)
        else:
            h_noisy = h

        # Top-R by noisy count (descending).
        top_R_idx = np.argsort(h_noisy)[-self.R :][::-1].astype(np.int64)

        # Stage 1b: reverse-order soft assignment.
        cluster_count_per_doc = np.zeros(N, dtype=np.int32)
        clusters: list[list[int]] = [[] for _ in range(self.R)]

        for r in reversed(range(self.R)):
            w_idx = int(top_R_idx[r])
            # Candidate docs containing this keyword and still under the cap.
            has_word = extractions[:, w_idx] > 0
            under_cap = cluster_count_per_doc < self.L
            members = np.flatnonzero(has_word & under_cap).tolist()
            clusters[r] = members
            for i in members:
                cluster_count_per_doc[i] += 1

        non_empty = sum(1 for c in clusters if c)
        self.last_stats = {
            "N": int(N),
            "vocab_size": int(V),
            "R": int(self.R),
            "L": int(self.L),
            "sigma_h": float(self.sigma_h),
            "non_empty_clusters": int(non_empty),
            "mean_cluster_size": (
                float(np.mean([len(c) for c in clusters])) if clusters else 0.0
            ),
            "mean_clusters_per_doc": float(np.mean(cluster_count_per_doc)),
            "max_clusters_per_doc": int(cluster_count_per_doc.max()) if N else 0,
        }
        return top_R_idx, clusters
