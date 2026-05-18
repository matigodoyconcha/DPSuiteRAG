# DP-SynRAG

This module implements DP-SynRAG (Differentially Private RAG via Synthetic Document Generation) for the BenchDP-RAG framework.

Reference paper:
- Anonymous et al. (2025), "DP-SynRAG: Differentially Private RAG via Synthetic Document Generation", arXiv:2510.06719v2

## What Changes vs Vanilla RAG

DP-SynRAG is an **indexing-time** privacy method. Unlike DP-RAG (which applies DP at every query), DP-SynRAG:

1. Builds a private synthetic corpus **once** from the original documents.
2. Serves all queries using a standard **vanilla** RAG pipeline over the synthetic corpus.

This means the DP budget is spent once per dataset, not once per query. Inference is fully non-private (CosineSimilarityRetriever + HFGenerator).

## Pipeline (Algorithm 1 from the paper)

### Stage 1a — Keyword extraction + noisy histogram

For each document `d_i`, an LLM extracts K keywords restricted to a fixed vocabulary V:

```
Ext_i ∈ {0,1}^|V|,   ||Ext_i||_1 = K
h(D) = Σ_i Ext_i
h'(D) = h(D) + N(0, σ_h² I_|V|)
```

**Component:** `KeywordExtractor`

**Sensitivity:** Each document contributes exactly K ones → L2 sensitivity = √K → zCDP term `ρ_h = K/(2σ_h²)`.

### Stage 1b — Soft clustering (top-R keywords)

Select the R highest-frequency keywords from `h'(D)`. Assign each document to clusters anchored on those keywords, processed in **reverse frequency order**, with each document entering at most L clusters:

```
for r = R, ..., 1:
    C_r = { d_i : w_r ∈ d_i, |{r' > r : d_i ∈ C_r'}| < L }
```

**Component:** `DPHistogramClusterer`

### Stage 1c — Per-cluster retrieval via noisy centroid + DP threshold

For each cluster C_r:

```
μ_r = Σ_{d_i ∈ C_r} E(d_i) + N(0, σ_μ² I_d)
θ_s ~ ExponentialMechanism(u(θ_s) = -|#{sim ≥ θ_s} - k|, ε_θs)
S_r = { d_i ∈ C_r : sim(E(d_i), μ_r) > θ_s }
```

**Component:** `DPClusterRetriever`

**Note:** Embeddings are L2-normalized before summation to bound sensitivity. The zCDP terms are `ρ_μ = 1/(2σ_μ²)` and `ρ_θs = 1/(8ε_θs²)` (exponential mechanism, sensitivity 1).

### Stage 2d — Private token prediction (clipped-logit aggregation)

For each cluster S_r, generate a synthetic document token-by-token:

```
z_n(S_r) = Σ_{d_i ∈ S_r} clip_c(L_model(d_i, y_{<n}))
y_{r,n} ~ softmax(z_n / τ)          # τ=0 → argmax (paper default)
```

Where `clip_c` applies L∞-bounded normalization:
1. Exponentiate & normalize: `l_exp(t) = exp(l(t) - max l)`
2. Center: `l_cent = l_exp - (max(l_exp) + min(l_exp)) / 2`
3. L∞ clip: `clip_c(l)(t) = l_cent(t) · min(1, c / ||l_cent||_∞)`

**Component:** `DPSyntheticGenerator`

**Privacy:** Each document contributes ≤ c to z_n in L∞; zCDP term `ρ_token = T/2 · (c/τ)²` per cluster application.

### Stage 2e — Self-filter (optional, no privacy cost)

A separate LLM call on each synthetic document asks YES/NO whether it is task-relevant. Only YES documents are kept. Since synthetic texts are already DP-guaranteed outputs, this is pure post-processing.

**Component:** `SelfFilter`

## Privacy Budget (zCDP)

The full privacy composition follows the paper formula (Section "Privacy Analysis"):

```
ρ = K/(2σ_h²)  +  L · [ 1/(8ε_θs²)  +  1/(2σ_μ²)  +  T/2 · (c/τ)² ]

(ε, δ)-DP conversion:
ε = ρ + √(4ρ · ln(1/δ))
```

**Greedy mode (τ=0):** The token term `T/2·(c/τ)²` diverges. In this case the per-token mechanism degenerates to argmax (Report Noisy Max equivalent), which the implementation handles via the fallback `ρ_token = T·c²/2`.

**Component:** `metrics.privacy_accounting_zcdp.zcdp_dp_synrag`

## Parameters

| Parameter | Description | Paper default |
|-----------|-------------|---------------|
| `K` | Keywords extracted per document | 10 |
| `R` | Number of clusters (top-R keywords) | 50 |
| `L` | Max clusters a document can join | 2 |
| `σ_h` | Gaussian noise on keyword histogram | calibrated to ε |
| `k` | Target documents per cluster (exp-mech) | 10–15 |
| `σ_μ` | Gaussian noise on per-cluster centroid | calibrated to ε |
| `ε_θs` | DP budget for exp-mech threshold per cluster | calibrated to ε |
| `T` | Tokens per synthetic document | 30–70 |
| `c` | L∞ clipping bound | 1.0 |
| `τ` | Softmax temperature (0 = greedy) | 0.0 |
| `δ` | Failure probability for (ε,δ)-DP | 1e-3 |

## Embedding Model

The paper uses `sentence-transformers/multi-qa-mpnet-base-dot-v1`. For fast iteration the default config uses `sentence-transformers/all-MiniLM-L6-v2`.

## LLMs Evaluated in the Paper

- Phi-4-mini (3.8B) — default in paper configs
- Gemma-2 (2B)
- Llama-3.1 (8B)

## Privacy Notes

### Singleton cluster bypass
Clusters with fewer than 2 members (singletons) skip Stage 1c entirely in
`DPClusterRetriever`: no Gaussian noise is added to µ_r and no exp-mech
threshold θ_s is sampled.  The singleton document is forwarded directly into
S_r.

**Implication:** the zCDP costs ρ_µ and ρ_θs assumed in Theorem 1 are NOT
charged for singleton clusters.  The actual privacy spend is therefore **lower**
than the bound computed by `zcdp_dp_synrag` (if singletons are frequent) — the
accounting is conservative for the normal case, but the per-cluster leakage for
individual singleton documents is bounded only by Stage 1a+1b (the histogram
mechanism), not by Stage 1c.  In practice, singleton clusters are rare when R and
L are calibrated to the corpus size.

### Centroid normalization
After adding Gaussian noise, µ_r is L2-normalized before computing similarities
(`cluster_retriever.py`, lines 95–97).  This is post-processing of a DP output
and therefore does **not** affect the privacy guarantee (post-processing theorem).
Removing the normalization changes utility but not privacy.

## Neighboring Dataset Convention

The accounting assumes an **add/remove** neighboring relation (one document is added or removed). This bounds the histogram sensitivity to √K and the centroid sensitivity to 1 (given L2-normalized embeddings).

## Performance Note

Keyword extraction requires one LLM forward pass per document (N passes total). Synthetic generation requires `O(R · max|S_r| · T)` forward passes. On CPU with Phi-3.5-mini this can take several hours for N=1000. Use the `max_samples` config key and the FAISS-backed cache (keyed on all DP hyperparams) to avoid redundant builds.
