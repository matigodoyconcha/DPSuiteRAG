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
   `[k_min, k_max]` is a **utility** knob (privacy-neutral, since it is public); the
   paper used `r(k)` ≈ `[15, 30]`, while the configs here use a smaller window for
   short-answer QA.
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

In this benchmark the indexed documents are the **chunks** produced by the chunker
(`FixedSizeChunker`, `overlap=20`), and retrieval returns distinct index entries —
so the privacy unit is one **chunk**. A source document spanning several
overlapping chunks therefore contributes several units; if a source-document
neighbourhood were intended instead, chunks would have to be deduplicated to one
per document.

## Privacy (ε as input)

`ε` is an input budget (the paper sweeps `ε ∈ {1,2,3,5,8}`, `δ = 1e-4`). The
FindBestK Gumbel scale `b` and the PTR noise `σ` are **calibrated** to the target
`(ε, δ)`:

- FindBestK is a pure-DP exponential mechanism (`eps_em = 2·Δd_k / b = 4/b`).
- TopKWithPTR has RDP curve `α / (2σ²)` and is `δ_ptr`-approximate.
- They are composed by RDP addition and converted to `(ε, δ)`-DP. Total
  `δ = δ_ptr + δ_conv`.

> **Factor-of-2 note.** The paper's main text (Alg. 3 Gumbel scale `2/eps_k`, and
> the informal `eps_k`-DP statement) drops the `Δd_k = 2` that the paper's **own
> Appendix A.11** uses when deriving the exponential-mechanism RDP — i.e. the paper
> is internally consistent at `Δ = 2`; only the main-text scale omits it. The
> accountant uses `eps_em = 2·Δd_k / b = 4/b`, consistent with Appendix A.11. This
> reconciles the main-text scale with the appendix proof; it is **not** a correction
> of a `Δ = 1` claim (the paper never claims `Δ = 1`).

## Deviations from the paper

- **FindBestK search window narrowed for short-answer QA.** The paper's Alg. 3
  gives `[15, 30]` only as an illustrative `r(k)` example and never reports the
  window actually used. On NQ/TQA the reference answer is 1–4 tokens (the
  paper's own Figs. 3–4 are answer-length histograms), so the count "cliff"
  sits at low rank; a literal `[15, 30]` window pushes FindBestK into the flat
  tail (`d_k = 1`), TopKWithPTR then never passes, and the ε-sweep curve is
  flat and ε-invariant. The paper-faithful config uses `[1, 8]` to capture the
  cliff; the full sweep config keeps `[15, 30]` as the paper's literal example.
  The window is a public, privacy-neutral utility knob (absent from the
  accountant), so neither choice affects the DP guarantee
  (`experiments/configs/full/dpksa.yaml`, `experiments/configs/paper_faithful/dpksa.yaml`).
- **PTR noise coupling lowered from 1:1 to 0.25:1.** `σ = ptr_noise_coupling · b`
  couples the PTR Gaussian noise to the FindBestK Gumbel scale for 1-D
  calibration; the paper does not fix this ratio. At `coupling = 1.0` the PTR
  threshold was so large the test almost never passed, degenerating to the
  zero-shot fallback; `coupling = 0.25` roughly halves the threshold and gives
  a better pass rate at `ε ≥ 3`. Calibration still enforces the target `ε`
  (checked by the upper-bound guard), so this is privacy-neutral
  (`experiments/configs/full/dpksa.yaml`, `experiments/configs/paper_faithful/dpksa.yaml`).
- **Exponential-mechanism sensitivity accounted as `eps_em = 4/b`, not the
  paper's literal `2/b`.** The paper's main text states `ε_k`-DP with Gumbel
  scale `2/ε_k` (i.e. `eps_em = 2/b`), but its own Appendix (Def. A.7 / Thm.
  A.11) derives the exponential-mechanism RDP curve at gap sensitivity
  `Δd_k = 2`, which implies `eps_em = 2·Δd_k/b = 4/b` — the main text
  under-reports by a factor of 2. The accountant always uses the
  appendix-consistent `4/b` (about 1.3x more noise than the paper's literal
  main-text formula) (`metrics/privacy_accounting_dpksa.py`).
- **"Token" operationalized as a normalized whitespace word.** The paper counts
  histogram entries at the "token" level without fixing the unit; this
  implementation defines a token as a normalized run of alphanumerics, since
  that is the unit that re-injects cleanly into the final keyword prompt (QA
  answers are typically 1–4 such words)
  (`dp_methods/per_query/dpksa/keyword_selector.py`).
- **Privacy unit is the chunk, not the source document.** The indexed
  "documents" are the overlapping chunks produced by `FixedSizeChunker`, and
  retrieval returns distinct chunk entries, so the neighboring-datastore
  relation `D ~ D'` is realized at the chunk level rather than the
  source-document level assumed by the paper's formalization; a source
  document spanning several overlapping chunks contributes several privacy
  units. A document-level neighborhood would require deduplicating chunks to
  one per source document (`dp_methods/per_query/dpksa/generator.py`).
- **Deterministic, seed-derived noise (benchmark-only).** The per-query RNG for
  the FindBestK Gumbel draws and the PTR Gaussian draw is seeded from
  `random_seed + hash(query)` rather than fresh secret randomness, so noise is
  reproducible across benchmark re-runs. The formal `(ε,δ)`-DP guarantee
  assumes fresh secret randomness; with a fixed public seed, an adversary who
  knows the query could in principle predict and invert the noise. This is
  explicitly flagged as benchmark-only and not deployment-grade — a real
  release would need a cryptographically secure RNG
  (`dp_methods/per_query/dpksa/generator.py`).
- **Corpus size and query count reduced from the paper's scale.** The paper
  retrieves from the full ~21M-passage Wikipedia corpus and evaluates a
  100-question subset per dataset. Indexing the full corpus is infeasible
  locally, so `max_index_docs` is a tractable stand-in (20,000 passages in the
  full sweep config; ~12M — about 57% of DPR psgs_w100 — in the paper-faithful
  config), and `max_test_queries` is capped at 40 per dataset, below the
  paper's 100, for tractable runtime
  (`experiments/configs/full/dpksa.yaml`, `experiments/configs/paper_faithful/dpksa.yaml`).
- **Standardized retriever/embedder shared across all benchmarked methods.**
  DP-KSA uses the project's standardized single-tower cosine embedder
  (`Snowflake/snowflake-arctic-embed-m-v1.5`) rather than whatever retriever
  the paper used, so that the retriever is not a confound in the cross-method
  comparison — fairness across methods is prioritized over per-paper retriever
  faithfulness
  (`experiments/configs/full/dpksa.yaml`, `experiments/configs/paper_faithful/dpksa.yaml`).

## Relevant parameters

| Parameter | Description | Default |
|---|---|---|
| `target_epsilon` / `target_delta` | Privacy budget per query | 3.0 / 1e-4 |
| `n_docs` | Documents retrieved for sample-and-aggregate (`N`) | 80 |
| `max_answer_tokens` | Tokens per per-document response | 16 |
| `max_final_tokens` | Tokens for the final keyword-conditioned answer | 32 |
| `k_min` / `k_max` | FindBestK search window (fixed, public) | 1 / `n_docs`−1 |
| `ptr_delta_fraction` | Share of `δ` assigned to the PTR false-release | 0.5 |
| `ptr_noise_coupling` | Couples `σ = coupling · b` for 1-D calibration | 1.0 |
| `lowercase` / `exclude_stopwords` | Histogram token normalisation | True / False |
| `random_seed` | Seed (per-query RNG derived from `hash(query)`) | 42 |

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
`dpksa.yaml` (full sweep) and `dpksa_paper_faithful.yaml` (paper LLMs + datasets).
