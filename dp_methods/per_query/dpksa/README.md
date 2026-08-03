# DP-KSA

Implementation of the DP-KSA method (Differentially Private Keyword Selection and
Aggregation) described in Tang, Flemings, Wang & Annavaram,
*"Differentially Private Retrieval-Augmented Generation"*
([arXiv:2602.14374](https://arxiv.org/abs/2602.14374)).

No official code was released by the authors; this is a from-paper reimplementation
following the formalisation in the thesis (mechanism steps and Lemmas KSA-1/2/3).

## Components

- [`generator.py`](generator.py) — `DPKSAGenerator`, the full `M_KSA(x, D)`
  mechanism (sample-and-aggregate → FindBestK → TopKWithPTR → keyword-conditioned
  generation).
- [`keyword_selector.py`](keyword_selector.py) — token histogram, `find_best_k`
  (exponential mechanism / Gumbel-max over count gaps) and `top_k_with_ptr`
  (Gaussian Propose-Test-Release).
- [`pipeline.py`](pipeline.py) — `DPKSAPipeline`, retrieves the top-`N` and
  delegates to the generator.

The `(ε,δ)` accounting is delegated to
[`metrics/privacy_accounting_dpksa.py`](../../../metrics/privacy_accounting_dpksa.py),
which composes FindBestK + TopKWithPTR under Rényi DP and converts to `(ε,δ)`-DP
(reusing the exponential-mechanism RDP curve and the BBGHS'19 conversion from
`privacy_accounting_autodp.py`).

## Mechanism

For each query `x`:

1. **Retrieval & sample-and-aggregate.** The retriever returns the top-`N`
   documents `D_1,…,D_N`. Each `D_i` is paired with `x` and generates its own short
   response `y_i = F(D_i, x)` (disjoint: 1 document → 1 response). The
   token-frequency histogram is `H[w] = #{i : w ∈ y_i}` (number of responses whose
   normalised-word set contains `w`).
2. **FindBestK.** `k̂ = argmax_k {d_k + r(k) + Gumbel(b)}` over the count gaps
   `d_k = H_(k) − H_(k+1)`, with `k` ranging over a **fixed, public** window
   `[k_min, k_max] ⊆ {1,…,N−1}` (paper Alg. 3: `r(k) = 0` inside, `−∞` outside).
   The candidate set is data-independent: gaps are read off the **zero-padded**
   sorted histogram, so every `k` in the window has a defined gap even with few
   distinct tokens. Gap sensitivity `Δd_k = 2` (Lemma KSA-1). The window
   `[k_min, k_max]` is a **utility** knob: it is public and data-independent, so it
   is absent from the accountant and cannot change the guarantee. Both configs
   place it at low rank, where the count cliff of short-answer QA sits (see
   Deviations below).
3. **TopKWithPTR.** Always runs (paper Alg. 1); the only branch is the noisy test
   `d̂ = max(2, d_k̂) + N(0, (2σ)²) − Φ(1−δ_ptr; 0, 2σ)`. If `d̂ > 2` the exact
   top-`k̂` keywords are released; otherwise none are released (zero-shot
   fallback). Degenerate histograms (few/zero distinct tokens) flow through this
   test with gap 0 and fail it with probability `≥ 1 − δ_ptr` — there is no
   data-dependent early exit.
4. **Final generation.** `y = F(x, {w_i})` from the query and the privately
   released keywords only. The database `D` is never re-accessed (genuine
   post-processing).

## Privacy unit & gap sensitivity (Lemma KSA-1)

Neighbouring datastores `D ~ D'` differ in exactly one document `d*` (the
document-level neighbourhood used throughout the thesis). Under the disjoint
sample-and-aggregate partition, changing `d*` alters at most one of the `N`
retrieved documents; since top-`N` retrieval **refills** the freed slot, one of the
`N` responses is **replaced**, so every token count moves by at most 1 *in either
direction* (`‖ΔH‖_∞ ≤ 1`) and each gap by at most 2 (`Δd_k ≤ 2`). The
both-directions move is what makes the sensitivity 2 (not 1, as in the add-only
source algorithm) — see Lemma KSA-1 and the observation after Lemma KSA-2 (the
factor-of-2). The accountant uses this `Δd_k = 2` via `eps_em = 4/b`.

Both configs index whole corpus documents (`chunking.enabled: false`), so one
indexed entry is one source document and the reported `ε` is a document-level
guarantee. Splitting a document into `c` chunks would protect a chunk, and reaching
the document from there costs group privacy `(c·ε, c·e^{(c−1)ε}·δ)`, whose `δ` grows
exponentially in `c`; `max_document_words: 400` caps prompt length without splitting
the privacy unit.

## Privacy (ε as input)

`ε` is an input budget (the paper sweeps `ε ∈ {1,2,3,5,8}`, `δ = 1e-4`). The
FindBestK Gumbel scale `b` and the PTR noise `σ` are **calibrated** to the target
`(ε, δ)`:

- FindBestK is a pure-DP exponential mechanism (`eps_em = 2·Δd_k / b = 4/b`).
- TopKWithPTR has RDP curve `α / (2σ²)` and is `δ_ptr`-approximate.
- They are composed by RDP addition and converted to `(ε, δ)`-DP. Total
  `δ = δ_ptr + δ_conv`.

The values swept in `full/dpksa.yaml` are per-query budgets chosen so that the same
two mechanisms, composed over a 100-query workload, land on the benchmark's shared
privacy axis; the mechanism `δ = 1e-4` here and the workload `δ = 1e-2` of that axis
are separate quantities (see Deviations).

> **Factor-of-2 note.** The paper's main text (Alg. 3 Gumbel scale `2/eps_k`, and
> the informal `eps_k`-DP statement) drops the `Δd_k = 2` that the paper's **own
> Appendix A.11** uses when deriving the exponential-mechanism RDP — i.e. the paper
> is internally consistent at `Δ = 2`; only the main-text scale omits it. The
> accountant uses `eps_em = 2·Δd_k / b = 4/b`, consistent with Appendix A.11. This
> reconciles the main-text scale with the appendix proof; it is **not** a correction
> of a `Δ = 1` claim (the paper never claims `Δ = 1`).

## Deviations from the paper

The benchmark runs every method through one **common evaluation protocol**: the
same six datasets, 20,000 indexed documents, 40 questions, the same ≤4B LLMs, the
same embedder, no chunking, a 50-token answer, and one shared privacy axis. Where
that protocol and the paper disagree, the protocol wins, and the difference is
recorded here.

### Corpus, queries and datasets

The paper retrieves from the full ~21M-passage Wikipedia corpus and scores a
100-question subset of NQ and TriviaQA. `full/dpksa.yaml` indexes 20,000 documents
per dataset and scores 40 questions, across the benchmark's six datasets (NQ,
TriviaQA, SearchQA, ChatDoctor, a Dirichlet medical set and MovieLens);
`paper_faithful/dpksa.yaml` keeps NQ and TriviaQA on the full DPR `psgs_w100`
index. A 20,000-document corpus is far denser in near-duplicates per query than
21M passages, which raises the agreement among the `N` per-document responses and
therefore the count gaps the mechanism reads; it is not a neutral downscaling.

### Answer length

`max_final_tokens: 50` is the benchmark-wide answer length, identical to every
other method's generation cap, so that answer-length effects do not enter the
cross-method comparison. `max_answer_tokens: 16` is a different quantity: it bounds
the per-document response `y_i` inside the sample-and-aggregate stage, which is
never shown to the scorer and only feeds the token histogram. Sixteen tokens is
long enough for the 1–4-token QA answers the histogram is meant to concentrate on
and short enough to keep 80 generations per query affordable, so it stays at the
mechanism's own value rather than following the shared answer length.

### Retriever

DP-KSA uses the benchmark's standardized single-tower cosine embedder
(`Snowflake/snowflake-arctic-embed-m-v1.5`) rather than the paper's retriever, so
that the retriever is not a confound across methods.

### `N = 80` is a mechanism parameter, not a retrieval depth

`n_docs: 80` (with `k_retrieved: 80`) is the size of the sample-and-aggregate
partition: the mechanism runs exactly one generation per retrieved document and
histograms the `N` responses, so `N` fixes the range of the counts `H_(k)` and
hence of the gaps `d_k ∈ [0, N]` that FindBestK ranks and TopKWithPTR tests. The
paper's §6.3.2 ablation selects 80. Other methods retrieve 50 documents, but their
50 is a retrieval depth handed to a prompt or a vote; setting `N = 50` here would
change the mechanism's aggregation width and shrink the largest gap the histogram
can produce by 37%, so the two numbers are not interchangeable and `N` is left at
the paper's value. One consequence is that `retrieval_precision_at_k` is scored at
`k = 80` for DP-KSA and at `k = 50` for the methods that retrieve 50
(`experiments/score_results.py` reads `k_retrieved` from the payload), so the
precision column is not comparable across methods at face value.

### The shared privacy axis

`epsilon_values: [0.1084, 0.2835, 0.7702, 1.7758, 4.2262]` replaces the paper's
per-query sweep `{1, 2, 3, 5, 8}`. These are per-query budgets whose **workload**
epsilon — FindBestK plus the PTR test composed over `Q = 100` queries at
`δ = 1e-2`, `metrics/workload_accounting.py` — lands on the benchmark's common
axis `{1, 3, 10, 30, 100}`, so a DP-KSA point and another method's point at the
same axis value are the same privacy level. DP-KSA is a per-query method whose
cost grows with the query stream, and paying 100 queries out of a per-query budget
is what pushes the per-query numbers an order of magnitude below the paper's.

Two deltas coexist and are not interchangeable. `delta: 1.0e-4` is the paper's
**mechanism** delta and is what `calibrate_dpksa` splits, half to the PTR
false-release (`δ_ptr = 5e-5`, `ptr_delta_fraction: 0.5`) and half to the RDP→DP
conversion slack; it is the delta that sets `σ` and the release threshold
`Φ^{-1}(1−δ_ptr)·2σ`. `DEFAULT_DELTA = 1e-2` is the **axis** delta covering the
whole workload; `δ_ptr` is approximate-DP mass that no Rényi curve absorbs, so it
is charged once per query and contributes `100 × 5e-5 = 5e-3` additively, half of
the 1e-2 target, leaving 5e-3 of conversion slack. The mechanism delta is
deliberately not loosened to match the axis: at `δ = 1e-5` the release threshold
rises about 21% and the method stops releasing anything at all.

### Keyword selection at these budgets

This is the deviation that matters most for reading DP-KSA's utility numbers.
The calibrated noise at each grid point, and the count gap `d` a histogram would
need for the PTR test to release with probability 1/2:

| per-query `ε` | axis `ε` | Gumbel `b` | `σ` | PTR threshold | `d` for 50% release | expected releases / 40 queries |
|---|---|---|---|---|---|---|
| 0.1084 | 1 | 136.9 | 34.24 | 266.4 | 268.4 | 0.002 |
| 0.2835 | 3 | 56.66 | 14.17 | 110.2 | 112.2 | 0.005 |
| 0.7702 | 10 | 22.59 | 5.65 | 43.9 | 45.9 | 0.35 |
| 1.7758 | 30 | 10.53 | 2.63 | 20.5 | 22.5 | 3.9 |
| 4.2262 | 100 | 4.86 | 1.21 | 9.4 | 11.4 | 14.2 |

A gap is a difference of two counts over `N = 80` responses, so `d ≤ 80` is a hard
physical ceiling; measured histograms reach a median maximum gap of 15 and a 90th
percentile of 42. At the two lowest points the required gap (268 and 112) exceeds
what a histogram of 80 responses can produce **even in the best case**, so the test
releases with probability `δ_ptr = 5e-5` and nothing else: 0.002 and 0.005 expected
releases per 40-question cell. At the third point the required gap of 46 is
attainable in principle but is beyond the 90th percentile of observed histograms,
giving 0.35 expected releases. Only the top two points release routinely (3.9 and
14.2 of 40).

The consequence must be stated plainly: at the lower half of the shared axis every
DP-KSA cell is a **zero-shot LLM** — the final prompt carries the question and no
keywords — while the payload still records a valid DP epsilon. Those cells measure
the generator's parametric knowledge, not a privacy–utility tradeoff, and a flat
utility curve there is the mechanism refusing to release, not a mechanism
degrading gracefully. `ptr_pass_rate` in each result payload is the number to read
first; the numbers above are the expectations it should match.

### The FindBestK window

`k_min: 2, k_max: 20`. The paper's Alg. 3 gives `[15, 30]` only as an illustrative
`r(k)` and never reports the window it used. On short-answer QA the count cliff
sits at low rank — the reference answer is 1–4 tokens — so `[15, 30]` samples the
flat tail where `d_k` is 0–1, `max(2, d_k)` is pinned at 2, and the release
probability collapses to `δ_ptr`. Measured on the observed histograms, over the
200 queries of a full five-point sweep, `[15, 30]` yields 8.0 expected releases
against 18.5 for `[2, 20]`, 20.3 for `[2, 10]` and 26.0 for `[1, 4]`.

At this grid the window choice is second-order compared with the budget. The
Gumbel scale is 137, 57 and 23 at the three lowest points, against gaps bounded by
80, so FindBestK is effectively a **uniform draw over the window**: at
`ε = 0.1084` it picks the noiseless argmax `k` with probability 0.059 against the
uniform 0.053 over `[2, 20]`. Only at the top of the grid does the selection carry
signal (0.418 at `ε = 4.2262`, `b = 4.86`). A uniform draw makes the window a
choice of *which gaps get sampled*, and narrower windows concentrated on low rank
sample larger gaps: `[1, 4]` releases most, `[2, 20]` least of the three. `[2, 20]`
is kept anyway, for two reasons the release count does not capture — `k_min = 1`
would release a single keyword, which starves the final prompt, and the 10%
release-rate gap to `[2, 10]` buys the headroom to release more keywords at the
two budgets where the test actually fires. The window is public and privacy-
neutral, so this is a utility choice and nothing else.

### Stopwords

`exclude_stopwords: false` follows the paper, which does not filter the histogram.
The tokens that genuinely pollute the histogram are echoed prompt scaffolding
(`Instruction:`, `Context:`, `Answer:`), and those are removed at the source by the
chat template and `_strip_scaffold`, not by a stopword list. The knob is kept for
ablations.

### The final prompt sees keywords only

`_keyword_prompt` builds the final generation from the question and the released
keywords; the retrieved documents never reach it, and the datastore is not
re-accessed after the histogram (that is what makes stage 4 post-processing). So
DP-KSA answers from at most `k̂` bare words while every other method in the
benchmark conditions on retrieved passages, private or synthetic. Its utility is
therefore not comparable like-for-like: a gap against, say, DP-RAG or MURAG is
mostly the gap between a keyword-conditioned task and a document-conditioned one,
and reading it as a privacy–utility deficit of the mechanism attributes to DP the
cost of a strictly harder task.

### Accounting and implementation choices

- **`eps_em = 4/b`, not the paper's literal `2/b`.** The paper's main text states
  `ε_k`-DP with Gumbel scale `2/ε_k`, but its own Appendix (Def. A.7 / Thm. A.11)
  derives the exponential-mechanism RDP curve at gap sensitivity `Δd_k = 2`, which
  implies `eps_em = 2·Δd_k/b = 4/b`; the main text under-reports by a factor of 2.
  The accountant always uses the appendix-consistent `4/b`, about 1.3x more noise
  than the main-text formula (`metrics/privacy_accounting_dpksa.py`).
- **`ptr_noise_coupling: 0.25`.** `σ = coupling · b` ties the PTR noise to the
  Gumbel scale so that calibration is a bisection on one scalar; the paper does not
  fix the ratio. At `coupling = 1.0` the release threshold is roughly twice as
  large and the test almost never passes. Calibration still hits the target `ε`
  (enforced by the upper-bound guard in `calibrate_dpksa`), so the ratio is
  privacy-neutral.
- **"Token" is a normalized whitespace word.** The paper counts at the "token"
  level without fixing the unit; `keyword_selector.py` uses a normalized run of
  alphanumerics, the unit that re-injects cleanly into the final keyword prompt.
- **Deterministic, seed-derived noise (benchmark-only).** The per-query RNG for the
  Gumbel draws and the PTR draw is seeded from `random_seed + hash(query)` rather
  than fresh secret randomness, so runs reproduce exactly. The formal `(ε,δ)`-DP
  guarantee assumes fresh *secret* randomness: with a fixed public seed an
  adversary who knows the query could predict and invert the noise. Benchmark-only,
  not deployment-grade (`dp_methods/per_query/dpksa/generator.py`).

## Relevant parameters

| Parameter | Description | Class default | `full/dpksa.yaml` |
|---|---|---|---|
| `target_epsilon` / `target_delta` | Privacy budget per query | 3.0 / 1e-4 | swept / 1e-4 |
| `n_docs` | Sample-and-aggregate partition size (`N`) | 80 | 80 |
| `max_answer_tokens` | Tokens per per-document response `y_i` | 16 | 16 |
| `max_final_tokens` | Tokens for the final keyword-conditioned answer | 32 | 50 |
| `k_min` / `k_max` | FindBestK search window (fixed, public) | 1 / `n_docs`−1 | 2 / 20 |
| `ptr_delta_fraction` | Share of `δ` assigned to the PTR false-release | 0.5 | 0.5 |
| `ptr_noise_coupling` | Couples `σ = coupling · b` for 1-D calibration | 1.0 | 0.25 |
| `lowercase` / `exclude_stopwords` | Histogram token normalisation | True / False | True / False |
| `random_seed` | Seed (per-query RNG derived from `hash(query)`) | 42 | 42 |

> The paper counts at the token level; a "token" is operationalised here as a
> normalised whitespace word (the unit that re-injects cleanly into the final
> prompt; QA answers are 1–4 such tokens).

> ⚠️ **Deterministic noise (benchmark-only).** The per-query RNG is seeded from
> `random_seed + hash(query)` for reproducible runs, so the Gumbel/PTR noise is a
> fixed function of the (public) query and seed. The formal `(ε,δ)`-DP guarantee
> assumes fresh *secret* randomness — with a fixed public seed an adversary who
> knows the query could predict and invert the noise. This is intentional for the
> benchmark but **not deployment-grade**: for any real release, seed from a
> cryptographically secure RNG.

## Output (`last_run_stats`)

`n_docs`, `k_hat`, `gap`, `ptr_passed`, `n_keywords`, `keywords`,
`target_epsilon`, `achieved_epsilon`, `delta`, `gumbel_scale`, `sigma`.

YAMLs live in [`experiments/configs/`](../../../experiments/configs/):
`full/dpksa.yaml` (full sweep) and `paper_faithful/dpksa.yaml` (paper LLMs + datasets).
