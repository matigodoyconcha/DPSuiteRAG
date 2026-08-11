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

`V` is the NLTK (v3.9.1) English `words` corpus with the English stopwords removed (paper §6.1), applied with no length or character filtering. For each document `d_i`, an LLM extracts K keywords restricted to that vocabulary V:

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

**Note:** Embeddings are L2-normalized before summation to bound sensitivity. The zCDP terms are `ρ_μ = 1/(2σ_μ²)` and `ρ_θs = ε_θs²/8` (exponential mechanism, sensitivity 1) — **not** the paper's printed `1/(8ε_θs²)`, which is provably infeasible at the paper's own parameters; see the note under *Privacy Budget*.

### Stage 2d — Private token prediction (clipped-logit aggregation)

For each cluster S_r, generate a synthetic document token-by-token:

```
z_n(S_r) = Σ_{d_i ∈ S_r} clip_c(L_model(d_i, y_{<n}))
y_{r,n} ~ softmax(z_n / τ)          # paper's own generation step runs at τ=1.0 (Table 7); code requires τ>1e-6 (see Privacy Budget)
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

> **`ρ_θs` formula corrected.** Paper Theorem 1 writes the threshold term as
> `1/(8ε_θs²)`; this code uses `ε_θs²/8`, the standard DP→zCDP bound for an
> exponential mechanism of pure-DP parameter `ε_θs` (sensitivity 1). The two
> are reciprocals and agree only at `ε_θs=1`; they diverge everywhere else.
> The paper's printed form is not merely dimensionally suspect (ρ should grow
> with `ε_θs`, not shrink) — it is provably not what the paper computed: at
> the paper's own `ε_θs=0.4` (Table 7) and `L=5`, the reciprocal term alone,
> `L · 1/(8·0.4²) = 3.9`, already exceeds the total `ρ≈2.2` that the paper's
> stated `ε_total=10` budget allows, making the paper's own headline result
> infeasible under its printed formula. The value consistent with the paper's
> own token-term derivation (Lemma 3) — and with the standard bounded-range
> exponential-mechanism bound — is `ε_θs²/8`, which is what this code
> implements. See the `metrics/privacy_accounting_zcdp` module docstring for
> the full derivation.

**Greedy mode (τ=0) is not supported.** The token term `T/2·(c/τ)²` diverges as
τ→0, so no valid privacy bound exists at τ=0. This matches the paper: its own
synthetic-generation exponential mechanism (Algorithm 1, Stage 2d) runs at
`τ=1.0` (Table 7), not τ=0 — the τ=0 mentioned elsewhere in the paper (§6.1) is
the unrelated downstream RAG-reader decoding temperature, free post-processing
over the already-DP synthetic corpus. `zcdp_dp_synrag` raises `ValueError` when
`τ ≤ 1e-6`, matching Theorem 1's requirement that τ>0. (There is no
`ρ_token = T·c²/2` fallback — set τ > 0.)

**Component:** `metrics.privacy_accounting_zcdp.zcdp_dp_synrag`

### Offline regime: ε is flat in the number of queries

DP-SynRAG is the only offline method in the benchmark, and this is its defining
property rather than an implementation detail. The budget above buys the corpus;
answering a question against that corpus touches no private data, so it is
post-processing and costs nothing. On the shared workload axis the whole ε–Q curve
is therefore a horizontal line — `epsilon_workload` reads `1 / 5 / 10 / 50` at
`Q = 1` and exactly the same values at `Q = 1000`, whereas every query-time method
pays again per question and its ε grows with Q.

The calibration solves against the ρ the shared axis prices, so it is
`epsilon_workload` that lands on the axis point exactly. The payload's
`epsilon_total` is the *build-time* accountant's number, the classic
`ρ + √(4ρ ln(1/δ))` conversion, and it reads slightly higher — `1.48 / 6.01 /
11.29 / 51.37` for the same four cells — because the workload accountant converts
zCDP to (ε,δ) through the optimal Rényi order, a tighter bound. The two describe
the same mechanism at the same ρ; the cross-method axis is the workload one.

The comparison this method is in is consequently asymmetric by
construction: at a large workload DP-SynRAG is cheap for the same corpus quality,
and at `Q=1` it is the most expensive method in the benchmark.

**Component:** `metrics.workload_accounting.dp_synrag_mechanisms` (`regime="offline"`)

## Parameters

| Parameter | Description | Paper default |
|-----------|-------------|---------------|
| `K` | Keywords extracted per document | 10 |
| `R` | Number of clusters (top-R keywords) | 50 |
| `L` | Max clusters a document can join | 5 |
| `σ_h` | Gaussian noise on keyword histogram | calibrated to ε |
| `k` | Target documents per cluster (exp-mech threshold; distinct from `inference.k_retrieved`, 10–15) | 80–100 |
| `σ_μ` | Gaussian noise on per-cluster centroid | calibrated to ε |
| `ε_θs` | DP budget for exp-mech threshold per cluster | calibrated to ε |
| `T` | Tokens per synthetic document | 30–70 |
| `c` | L∞ clipping bound | 1.0 |
| `τ` | Softmax temperature for token sampling; must be `> 1e-6` (the paper's own generation step runs at `τ=1.0`, Table 7) — see Privacy Budget | 1.0 |
| `δ` | Failure probability for (ε,δ)-DP | 1e-3 (the benchmark's common workload target is 1e-2) |

In the `full` config, `c`, `σ_h`, `σ_μ` and `ε_θs` are not set by hand: the runner
derives all four from `synthetic_generation.epsilon_targets`, the points of the shared
benchmark axis. See *Deviations from the paper* for the derived values and for what they
imply about the readability of the synthetic text.

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
(`cluster_retriever.py`, `DPClusterRetriever.retrieve_clusters`).  This is post-processing of a DP output
and therefore does **not** affect the privacy guarantee (post-processing theorem).
Removing the normalization changes utility but not privacy.

## Deviations from the paper

- **`ρ_θs` formula corrected.** Paper Theorem 1 writes the threshold
  exponential-mechanism term as `1/(8ε_θs²)`; this is not merely
  dimensionally suspect (ρ should grow with `ε_θs`, not shrink) — it is
  provably not what the paper computed, since at the paper's own `ε_θs=0.4`
  (Table 7) and `L=5` the reciprocal term alone (`L·1/(8·0.4²) = 3.9`)
  exceeds the total `ρ≈2.2` that the paper's own `ε=10` budget allows. The
  code uses the standard DP→zCDP bound `ε_θs²/8` instead, which is
  consistent with the paper's own token-term derivation (Lemma 3) and
  matches the reciprocal form only at `ε_θs=1`. (`cluster_retriever.py`,
  `metrics/privacy_accounting_zcdp.py`; see "Privacy Budget" above)

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

- **Stage 1/2 hyperparameters follow the paper's Table 7 (Appendix C.3).**
  The paper publishes per-dataset values for `K, R, L, k, T, ε_θs, τ, ρ_hist,
  ρ_retr` in Table 7, with Tables 8–11 giving sensitivity sweeps over them —
  these are paper-specified, not reconstructed. `σ_h` and `σ_μ` are derived
  directly from the published `ρ_hist` and `ρ_retr` (e.g. `σ_h≈7.071` from
  `ρ_hist=0.1`, `σ_μ≈7.454` from `ρ_retr=0.009`); `ε_θs=0.4`, `k=80–100`, and
  `τ=1.0` are taken directly from Table 7. The one hyperparameter genuinely
  absent from Table 7 is `c` (the L∞ clipping bound), which this benchmark
  derives from the accountant — per dataset to an honest `ε≈10` in the
  paper-faithful config, or swept in the full config — see the `ε` deviation
  bullet below. (`experiments/configs/paper_faithful/dp_synrag.yaml`)

- **ε is derived, not read off: the whole build is scaled to a point of the
  shared axis.** The paper fixes the stage parameters and reports the budget
  they imply; the benchmark needs the reverse, because every method has to sit
  at the same `ε ∈ {1, 3, 5, 10, 50}` at `δ=1e-2`. The `full` config therefore
  lists `synthetic_generation.epsilon_targets` and the runner
  (`experiments/run_dp_synrag.py::_sweep_points` →
  `metrics.privacy_accounting_zcdp.calibrate_dp_synrag`) derives all four stage
  parameters together: every stage's ρ is quadratic in its own parameter, so one
  scalar `s = √(ρ_target/ρ_ref)` divides the noise scales and multiplies the
  budgets and the clip, holding the four stages' proportions at the paper's
  Table 7 ratios. At `T=70` (Medical Synth, Movielens, ChatDoctor, NQ, TriviaQA):

  | target ε | `c` | `σ_h` | `σ_μ` | `ε_θs` | ρ |
  |---|---|---|---|---|---|
  | 1   | 0.02428 | 291.28 | 307.06 | 0.00971 | 0.103 |
  | 5   | 0.08398 | 84.20  | 88.76  | 0.03359 | 1.236 |
  | 10  | 0.13910 | 50.84  | 53.59  | 0.05564 | 3.391 |
  | 50  | 0.40303 | 17.54  | 18.49  | 0.16121 | 28.466 |

  Scaling only `c` — the natural knob, since the token term dominates ρ (99.9%
  of it here) — cannot reach the tight end of that axis: the histogram,
  threshold and centroid stages alone spend `ρ = 0.245`, i.e. `ε ≈ 1.75` on the
  shared axis at `δ=1e-2` (`2.37` under the build-time conversion), so `ε=1` is
  unreachable at any clip whatsoever and `ε=5` is the first point with any room
  under it. Moving all four stages removes the floor, which is why
  the calibration is over the four and not over `c`. The paper-faithful config
  keeps the opposite convention: the scales are the paper's, `c` is the knob,
  and the reported ε is whatever composes out. (`experiments/configs/full/dp_synrag.yaml`,
  `metrics/privacy_accounting_zcdp.py`)

- **The synthetic text is only legible over a narrow part of that grid.**
  This is the method's sharpest limitation and it is structural, not a tuning
  accident. `clip_c` exponentiates the max-shifted logits, so `l_exp ∈ (0,1]`
  with `max = 1` and `min ≈ 0`; centring by the midrange puts every row in
  roughly `[-0.5, +0.5]` before the L∞ clip. Two consequences follow.

  First, **`c ≥ 0.5` is a no-op clip**: `min(1, c/‖l_cent‖_∞) = 1` already at
  `c = 0.5`, so SearchQA's `ε=50` point (`c = 0.615`, `T=30`) produces a
  byte-identical corpus to `c = 0.5`. The top of the ε axis saturates by
  construction: the accountant charges more privacy, the text does not change.
  On the `T=70` datasets the axis stops just short of that (`c = 0.403` at
  `ε=50`), so there the saturation is a limit the grid approaches rather than
  reaches.

  Second, a document's own top token receives `≈ +c` and essentially every
  other token `≈ -c`, so aggregating `n` documents that agree gives the top
  token a lead of **at most** `2·c·n` over the tail. Sampled at `τ=1` over a
  vocabulary of `V` tokens, that sampler is near-uniform while `2·c·n ≪ ln V`
  and a deterministic argmax once `2·c·n ≫ ln V`. With the config's cluster
  target `cluster_retrieval.k = 80` and `ln V ≈ 10.4` (32k vocab) / `12.2`
  (200k), at the unanimous-cluster bound:

  | target ε | `c` | `2·c·n` | p(top token) | entropy (nats) |
  |---|---|---|---|---|
  | 1   | 0.0243 | 3.9  | 0.0015 | 10.37 (uniform) |
  | 5   | 0.0840 | 13.4 | 0.955  | 0.65 |
  | 10  | 0.1391 | 22.3 | 1.000  | 0.00 |
  | 50  | 0.4030 | 64.5 | 1.000  | 0.00 |

  Taking the window as `p(top) ∈ [0.01, 0.99]`, it is `c ≈ 0.036–0.094` at
  `n=80` (`0.029–0.075` at `n=100`), which maps to `ε ≈ 1.7–5.8` at `T=70` and
  `ε ≈ 1.0–3.3` at `T=30`. Exactly one shipped point, `ε=5`, falls inside it,
  and it sits near the deterministic edge. Below the window the corpus is
  uniform noise over the vocabulary; above it the corpus is the noiseless argmax
  continuation, which degenerates into repetition loops ("…is very friendly and
  friendly. He is very friendly and friendly…"). Real clusters do not agree
  unanimously, so the effective lead is below `2·c·n` and the true window sits
  somewhat higher in ε than the bound above; the bound is what can be derived
  without running the generator. Either way DP-SynRAG's ε–utility curve should
  be read as a curve with three regimes rather than four comparable points, and
  the window moves with the cluster size and the generator's vocabulary rather
  than being a property of the budget alone.
  (`synthetic_generator.py::_clip_c_row`, `_sample_token`)

- **Answer length unified at 30 tokens.** `inference.max_new_tokens = 30` for
  every dataset and every generator, matching the rest of the benchmark, so
  containment- and overlap-based metrics see the same amount of text from every
  method. (`experiments/configs/full/dp_synrag.yaml`)

- **Reduced corpora and a proportionally smaller `R`.** The `full` config
  indexes a 20% subsample of each paper corpus — 1600 of 8000 (Medical Synth),
  816 of 4083 (Movielens), 1410 of 7054 (SearchQA), 1600 for the harmonized
  additions — because the build is `O(N)` LLM passes for keyword extraction plus
  `O(R · |S_r| · T)` for generation, and the full corpora put a single sweep cell
  in the tens of hours. `R` (the number of synthetic documents) is scaled with
  it: 50 everywhere, 100 for SearchQA. The resulting coverage is **not** constant
  across datasets:

  | dataset | indexed docs | `R` | R/corpus | real docs per synthetic doc |
  |---|---|---|---|---|
  | Medical Synth | 1600 | 50  | 3.1% | 32 |
  | Movielens     | 816  | 50  | 6.1% | 16 |
  | SearchQA      | 1410 | 100 | 7.1% | 14 |
  | ChatDoctor / NQ / TriviaQA | 1600 | 50 | 3.1% | 32 |

  The paper's own ratios are 6.3% / 12.2% / 14.2%, so every dataset here
  compresses about twice as hard as the paper's, and Medical Synth twice as hard
  again as SearchQA. A cross-dataset comparison of DP-SynRAG's utility is
  therefore partly a comparison of compression ratios.
  (`experiments/configs/full/dp_synrag.yaml`)

- **Per-dataset `overrides:`, the only config in the benchmark that has them.**
  The paper varies its stage parameters per dataset, and two of its stages are
  task-specific, so a single global block cannot express it. Each override and
  why it exists:

  | dataset | override | status |
  |---|---|---|
  | Movielens | `cluster_retrieval.k: 100` | paper value for this dataset |
  | Movielens | `inference.k_retrieved: 15` | paper value for this dataset |
  | Movielens | `self_filter.prompt_template` (movie title) | required — the global prompt asks for a *diagnosis name*, which no movie document contains, so the filter would reject the entire corpus |
  | Movielens | `metrics.utility: [hit_rate, perplexity]` | required — the gold is a set of favourite titles, so a single-reference match is undefined |
  | SearchQA | `clustering.R: 100` | paper uses `R=1000` here versus 500 elsewhere; scaled with the corpus |
  | SearchQA | `cluster_retrieval.k: 100` | paper value for this dataset |
  | SearchQA | `self_filter.enabled: false` | paper applies no self-filter to SearchQA (no fixed task to filter for) |
  | SearchQA | `synthetic_generation.T: 30` | paper value (70 elsewhere) |
  | NQ, TriviaQA | `self_filter.enabled: false` | required — same reason as Movielens' prompt, for open-domain QA there is no diagnosis to name |

  ChatDoctor keeps the global medical self-filter and needs no override. The
  overrides are deep-merged **before** the ε grid is derived, so SearchQA's
  `T=30` is what the calibration solves against: its points are `c = 0.0370 /
  0.1282 / 0.2123 / 0.6151`, larger than the `T=70` datasets' at every target
  because a shorter synthetic document spends the token term fewer times, and
  each reaches the same ε as every other dataset. The sweep directory is named
  by the axis coordinate rather than by `c` — `target_1/`, `target_5/`,
  `target_10/`, `target_50/` — so the same cell of the shared axis has the same
  path on every dataset even where the `c` implementing it differs.
  (`experiments/run_dp_synrag.py`, `experiments/configs/full/dp_synrag.yaml`)

- **`inference.k_retrieved` is 10 (15 for Movielens), not the benchmark's 50.**
  The synthetic corpus holds `R` documents in total — 50, or 100 for SearchQA —
  so retrieving 50 would hand the reader the entire corpus and turn retrieval
  into a no-op. The values are the paper's, and even at 10 a query already sees
  20% of the synthetic corpus (30% on Movielens), against 0.25% for a method
  retrieving 50 of 20000 real documents. (`experiments/configs/full/dp_synrag.yaml`)

- **`retrieval_precision_at_k` is undefined for this method and is reported
  blank.** Every other method retrieves from the real corpus, so its ranking can
  be scored against the non-private ranking over the same index. DP-SynRAG
  retrieves from a corpus it generated: a synthetic document descends from a
  cluster, not from any one real document, and its id does not name a corpus
  document at all. The runner therefore records no `retrieved_ids`, and the
  scorer leaves the column empty rather than writing a 0 that would read as
  "retrieved nothing relevant". What is recorded instead is `cluster_sources`,
  the real documents that fed each cluster, which makes the related question —
  did the synthetic corpus preserve the documents a non-private retrieval would
  have used — measurable through the cluster. (`indexer.py`,
  `experiments/score_results.py`)

- **Standardized retriever instead of the paper's embedder.** All
  benchmarked DP methods share `Snowflake/snowflake-arctic-embed-m-v1.5`
  rather than the paper's `multi-qa-mpnet-base-dot-v1`, so the retriever is
  not a confound in the cross-method comparison ("fairness > per-paper
  faithfulness"). The Stage 1c threshold window is `min_sim=0.0, max_sim=1.0`,
  the range the paper fixes for `θ_s`; a wider lower bound would add a flat
  max-utility plateau below the real cosine range and pull the sampled threshold
  to the floor, at which point `S_r` stops discriminating.
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

Keyword extraction requires one LLM forward pass per document (N passes total). Synthetic generation requires `O(R · max|S_r| · T)` forward passes. On CPU with Phi-3.5-mini this can take several hours for N=1000.

Two caches keep a sweep affordable, and they sit on opposite sides of the privacy boundary:

- **Keyword extraction** (`experiments/extraction_cache.py`) is the last stage before any DP mechanism touches the data — `indexer.build_from_documents` runs it first, and only then the noisy histogram. Its output depends on the generator, the documents and the extractor's own settings (`K`, `max_new_tokens`, the prompt, the hashed vocabulary), and on nothing else: not the seed, not the budget, not the clip. The cache key covers exactly those, so one extraction serves every cell of the ε sweep for a (dataset, generator) pair and nothing DP-dependent can be reused across configurations. Decoding is greedy, so the omission of a sampling seed from the key is not a source of drift.
- **The synthetic corpus** (`run_dp_synrag.py::_synth_cache_dir`) is the mechanism's output, so its key carries every parameter that shapes it — the stage parameters, the self-filter, the embedder, the generator, the document selection, and the run seed. The run seed is in the key on purpose: without it a sweep over seeds would hand every repetition the first one's corpus and report a mechanism variance of about zero. The run id is deliberately *not* in the key, so the same configuration reuses a corpus across invocations; a `.synth_complete` marker written after `indexer.save` is what separates a finished build from an abandoned one, and a directory without it is discarded rather than loaded. MIA runs never read the cache at all, since the key describes the configuration and not the member subset the probe indexes.
