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

**Note:** Embeddings are L2-normalized before summation to bound sensitivity. The zCDP terms are `ρ_μ = 1/(2σ_μ²)` and `ρ_θs = ε_θs²/8` (exponential mechanism, sensitivity 1). **Paper Theorem 1 writes `1/(8ε_θs²)`**; the code uses the standard EM→zCDP bound `ε_θs²/8` — see the deviation note under *Privacy Budget*.

### Stage 2d — Private token prediction (clipped-logit aggregation)

For each cluster S_r, generate a synthetic document token-by-token:

```
z_n(S_r) = Σ_{d_i ∈ S_r} clip_c(L_model(d_i, y_{<n}))
y_{r,n} ~ softmax(z_n / τ)          # paper uses τ=0; code requires τ>1e-6 (see Privacy Budget)
```

Where `clip_c` applies L∞-bounded normalization:
1. Exponentiate (max-shifted): `l_exp(t) = exp(l(t) - max l)` — equivalently `e^{l(t)}/max_s e^{l(s)}`; no sum-normalization is applied.
2. Center: `l_cent = l_exp - (max(l_exp) + min(l_exp)) / 2`
3. L∞ clip: `clip_c(l)(t) = l_cent(t) · min(1, c / ||l_cent||_∞)`

**Component:** `DPSyntheticGenerator`

**Privacy:** Each document contributes ≤ c to z_n in L∞; zCDP term `ρ_token = T/2 · (c/τ)²` per cluster application.

### Stage 2e — Self-filter (optional, no privacy cost)

A separate LLM call on each synthetic document asks YES/NO whether it is task-relevant. Only YES documents are kept. Since synthetic texts are already DP-guaranteed outputs, this is pure post-processing.

**Component:** `SelfFilter`

## Privacy Budget (zCDP)

The full privacy composition (Section "Privacy Analysis"), **as implemented**:

```
ρ = K/(2σ_h²)  +  L · [ ε_θs²/8  +  1/(2σ_μ²)  +  T/2 · (c/τ)² ]

(ε, δ)-DP conversion:
ε = ρ + √(4ρ · ln(1/δ))
```

> **Deliberate deviation — `ρ_θs`.** Paper Theorem 1 writes the threshold term as
> `1/(8ε_θs²)`; this code uses `ε_θs²/8`, the standard DP→zCDP bound for an
> exponential mechanism of pure-DP parameter `ε_θs` (sensitivity 1). The two are
> reciprocals and agree at the calibration default `ε_θs=1`; they diverge when
> `ε_θs` is swept. The paper's form is dimensionally inverted (ρ should grow with
> `ε_θs`, not shrink), so `ε_θs²/8` is the correct bound. See the
> `metrics/privacy_accounting_zcdp` module docstring for the full rationale.

**Greedy mode (τ=0) is not supported.** The token term `T/2·(c/τ)²` diverges as
τ→0, so no valid privacy bound exists. `zcdp_dp_synrag` raises `ValueError` when
`τ ≤ 1e-6`. The paper runs experiments at τ=0, but its Theorem 1 proof assumes τ>0;
this implementation therefore requires `τ > 1e-6`. (There is no `ρ_token = T·c²/2`
fallback — set τ > 0.)

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

The paper uses `sentence-transformers/multi-qa-mpnet-base-dot-v1`. This benchmark standardizes on `Snowflake/snowflake-arctic-embed-m-v1.5` (shared across all methods so the retriever is not a cross-method confound); see the "Standardized retriever" entry under [Deviations from the paper](#deviations-from-the-paper).

## LLMs Evaluated in the Paper

- Phi-4-mini (3.8B) — default in paper configs
- Gemma-2 (2B)
- Llama-3.1 (8B)

## Privacy Notes

### Small clusters (no bypass — paper-faithful)
Stage 1c runs on **every non-empty cluster, including singletons** — the paper
applies no special-case to small clusters (the threshold EM targets ~`k` documents
and Stage 2e self-filter handles the low quality of small-subset synthetic text).
So `DPClusterRetriever` adds the Gaussian µ_r noise and samples the exp-mech
threshold θ_s for singletons too, spending the ρ_µ and ρ_θs that Theorem 1 charges.
This honours the per-stage accounting and removes the per-individual weakening that
a singleton bypass would create. Empty clusters (0 members) yield an empty `S_r`.

If the privately-sampled θ_s excludes every document, `S_r` is empty and the cluster
produces no synthetic document — there is **no** non-private argmax fallback.

### Deterministic seeding (reproducibility caveat)
The per-cluster Gaussian noise (µ_r) and exp-mechanism threshold (θ_s) are drawn
from an RNG seeded deterministically from `random_seed | cluster_index`
(`cluster_retriever.py`). This makes synthetic-corpus builds **reproducible** across
runs, which is intended for benchmarking. The formal (ε,δ) guarantee is a property
of the noise *distribution* and is unaffected, but a fixed seed means the released
output is a deterministic function of the inputs rather than a fresh draw — not a
deployment-grade randomness assumption. Same caveat as DP-KSA / DP-RAG.

### Centroid normalization
After adding Gaussian noise, µ_r is L2-normalized before computing similarities
(`cluster_retriever.py`, lines 95–97).  This is post-processing of a DP output
and therefore does **not** affect the privacy guarantee (post-processing theorem).
Removing the normalization changes utility but not privacy.

## Deviations from the paper

- **`ρ_θs` formula corrected.** Paper Theorem 1 writes the threshold
  exponential-mechanism term as `1/(8ε_θs²)`; this is dimensionally inverted
  (ρ should grow with `ε_θs`, not shrink), so the code uses the standard
  DP→zCDP bound `ε_θs²/8` instead, matching only at the calibration default
  `ε_θs=1`. (`cluster_retriever.py`, `metrics/privacy_accounting_zcdp.py`; see
  "Privacy Budget" above)

- **Greedy sampling (τ→0) is not supported.** The paper's reported
  experiments run at τ≈0, but Theorem 1's proof — and the
  `ρ_token=(T/2)(c/τ)²` term — require τ>0, which diverges as τ→0.
  `DPSyntheticGenerator` and `zcdp_dp_synrag` both reject `τ≤1e-6` rather than
  silently falling back to an unaccounted argmax/greedy token. (`synthetic_generator.py`,
  `metrics/privacy_accounting_zcdp.py`)

- **Centroid normalization not in the paper.** `μ_r` is L2-normalized after
  adding Gaussian noise before computing similarities; the paper does not do
  this explicitly. It is privacy-neutral (post-processing of an already-DP
  value) and only affects utility. (`cluster_retriever.py`)

- **Deterministic per-cluster RNG seeding.** The centroid noise and
  exponential-mechanism threshold are drawn from an RNG seeded from
  `(random_seed, cluster_index)` rather than a fresh draw each run, so
  synthetic-corpus builds are reproducible for benchmarking. This does not
  change the noise distribution (still (ε,δ)-DP), but the released output is
  a deterministic function of the inputs, not a deployment-grade randomness
  assumption. (`cluster_retriever.py`, `synthetic_generator.py`)

- **Most Stage 1/2 hyperparameters are reconstructed, not paper-specified.**
  The paper's Appendix C.3 says only that `R, L, K, k, c, τ, σ_h, σ_μ, ε_θs`
  were "tuned using the validation queries," without publishing values; every
  such number in the configs is a benchmark guess, not a paper citation.
  (`experiments/configs/paper_faithful/dp_synrag.yaml`)

- **ε is reported honestly rather than back-solved to the paper's
  ε_total=10.** The paper operates at a fixed total budget of 10 (Table 1);
  this benchmark instead fixes `T`, `c`, `τ`, `σ_h`, `σ_μ`, `ε_θs` at coherent
  values and lets the zCDP accountant report whatever ε results — usually far
  above 10, since `ρ_token=(T/2)(c/τ)²·L` dominates. `c` was deliberately
  raised from a back-solved `0.16` to `1.0` (full config) / swept over
  `[0.15, 0.25, 0.625]` (paper-faithful config) for sharper, more readable
  synthetic text, accepting the resulting high ε instead of hiding it.
  (`experiments/configs/full/dp_synrag.yaml`,
  `experiments/configs/paper_faithful/dp_synrag.yaml`)

- **`L` lowered from the reconstructed paper value in the paper-faithful
  sweep.** `L` (max clusters per document) is lowered from 5 to 2 specifically
  to cut the group-privacy multiplier on `ρ_token`, keeping the ε-utility
  sweep in a usable range at the cost of each document contributing to fewer
  clusters. (`experiments/configs/paper_faithful/dp_synrag.yaml`)

- **Standardized retriever instead of the paper's embedder.** All
  benchmarked DP methods share `Snowflake/snowflake-arctic-embed-m-v1.5`
  rather than the paper's `multi-qa-mpnet-base-dot-v1`, so the retriever is
  not a confound in the cross-method comparison ("fairness > per-paper
  faithfulness"). The Stage 1c threshold window (`min_sim=-0.5, max_sim=0.8`)
  is recalibrated to this embedder's cosine range instead of the paper's.
  (`experiments/configs/full/dp_synrag.yaml`,
  `experiments/configs/paper_faithful/dp_synrag.yaml`)

- **Evaluated on datasets beyond the paper's three.** The harmonized `full`
  config additionally runs DP-SynRAG on ChatDoctor, Natural Questions and
  TriviaQA (beyond the paper's Medical Synth / Movielens / SearchQA), so
  every method in the benchmark shares the same dataset suite; the
  self-filter is kept only for ChatDoctor (fixed diagnosis-naming task) and
  disabled for the open-domain QA additions. (`experiments/configs/full/dp_synrag.yaml`)

- **Self-filter reject-all degrades to no-RAG instead of erroring.** If the
  self-filter rejects every synthetic document — expected at a tight budget,
  where generated text is too generic to name a diagnosis/title — Algorithm
  1's discard rule leaves the corpus empty. Rather than aborting the run, the
  benchmark treats this as a genuine operating point of the mechanism:
  retrieval is disabled and answers fall back to the model's parametric
  knowledge. (`indexer.py`, `experiments/run_dp_synrag.py`,
  `experiments/configs/paper_faithful/dp_synrag.yaml`)

## Neighboring Dataset Convention

The accounting assumes an **add/remove** neighboring relation (one document is added or removed). This bounds the histogram sensitivity to √K and the centroid sensitivity to 1 (given L2-normalized embeddings).

## Performance Note

Keyword extraction requires one LLM forward pass per document (N passes total). Synthetic generation requires `O(R · max|S_r| · T)` forward passes. On CPU with Phi-3.5-mini this can take several hours for N=1000. Use the `max_samples` config key and the FAISS-backed cache (keyed on all DP hyperparams) to avoid redundant builds.
