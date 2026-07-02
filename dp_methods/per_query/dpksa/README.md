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
