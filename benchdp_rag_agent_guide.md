# BenchDP-RAG — Agent Implementation Guide

> This document describes **how to structure and build** the evaluation framework for Differential Privacy methods in RAG systems. It is the **official implementation base**: components not yet built are implemented **exactly as described here**; components already built are documented here as they stand in the code (the code is the source of truth for implemented methods).

> **Status convention.** Each block/method carries a marker:
> - `✅ DONE` — implemented and the guide describes the actual code.
> - `🚧 PENDING` — still specification only; implement it **as described in this guide**.
> - `➕ ADDED` — exists in the code and was not in the original version of the guide; documented here for completeness.

---

## 1. Project Purpose

Build a modular, reproducible, and extensible Python framework that allows:

1. Running a complete RAG pipeline with interchangeable components.
2. Substituting any pipeline component with a Differential Privacy (DP) version.
3. Evaluating and comparing 6 state-of-the-art DP-RAG algorithms under homogeneous conditions.
4. Producing tables and privacy–utility trade-off curves for different ε values.

The methods: **DP-RAG, DPVoteRAG/DPSparseVoteRAG, DP-KSA, PRIVATE-RAG, DP-SynRAG**
(DPVoteRAG and DPSparseVoteRAG count as two algorithms, for six in total).

| Method | Status |
|---|---|
| DP-RAG | ✅ DONE |
| DPVoteRAG / DPSparseVoteRAG | ✅ DONE |
| DP-KSA | ✅ DONE |
| DP-SynRAG | ✅ DONE |
| PRIVATE-RAG | 🚧 PENDING |

---

## 2. Directory Tree

Actual repository state. `(stub)` = empty file that must still be implemented following this guide.

```
benchdp_rag/
│
├── core/                          # ✅ RAG pipeline abstract classes
│   ├── __init__.py
│   ├── chunker.py                 # BaseChunker
│   ├── embedder.py                # BaseEmbedder
│   ├── indexer.py                 # BaseIndexer
│   ├── retriever.py               # BaseRetriever
│   ├── reranker.py                # BaseReranker
│   ├── prompt_builder.py          # BasePromptBuilder
│   ├── generator.py               # BaseGenerator
│   ├── postprocessor.py           # BasePostprocessor
│   └── pipeline.py                # RAGPipeline (can be subclassed — see §11.2)
│
├── vanilla/                       # ✅ Standard RAG implementation (no DP)
│   ├── chunker.py                 # SentenceChunker, FixedSizeChunker
│   ├── embedder.py                # HFEmbedder (sentence-transformers)
│   ├── indexer.py                 # FAISSIndexer
│   ├── retriever.py               # CosineSimilarityRetriever
│   ├── reranker.py                # IdentityReranker
│   ├── prompt_builder.py          # SimplePromptBuilder
│   ├── generator.py               # HFGenerator
│   └── postprocessor.py           # IdentityPostprocessor
│
├── dp_methods/
│   │
│   ├── offline/                   # Budget spent ONCE at indexing time; queries are non-private
│   │   ├── __init__.py
│   │   └── dp_synrag/             # ✅ DONE — decomposed into 6 components
│   │       ├── __init__.py
│   │       ├── keyword_extractor.py   # KeywordExtractor (Stage 1a)
│   │       ├── clusterer.py           # DPHistogramClusterer (Stage 1a+1b)
│   │       ├── cluster_retriever.py   # DPClusterRetriever (Stage 1c)
│   │       ├── synthetic_generator.py # DPSyntheticGenerator (Stage 2d)
│   │       ├── self_filter.py         # SelfFilter (Stage 2e)
│   │       ├── indexer.py             # DPSyntheticIndexer (orchestrates 1a→2e)
│   │       ├── generator.py           # (vestigial stub — inference uses HFGenerator)
│   │       └── README.md
│   │
│   └── per_query/                 # ε consumed on EVERY query at inference time
│       ├── __init__.py
│       ├── dp_rag/                # ✅ DONE
│       │   ├── __init__.py
│       │   ├── retriever.py       # DPRetriever, DPTopPRetriever
│       │   ├── prompt_builder.py  # ➕ PerDocPromptBuilder
│       │   ├── generator.py       # DPAggregatedGenerator (DP-ICL)
│       │   └── README.md
│       │
│       ├── dpvoterag/             # ✅ DONE
│       │   ├── __init__.py
│       │   ├── generator.py       # VoteRAGGenerator (Limited Domain Gumbel)
│       │   ├── sparse_vote_generator.py  # SparseVoteRAGGenerator (LD + SVT)
│       │   ├── pipeline.py        # ➕ DPVoteRAGPipeline (subclass — see §11.2)
│       │   └── README.md
│       │
│       ├── dpksa/                 # ✅ DONE
│       │   ├── __init__.py
│       │   ├── generator.py       # DPKSAGenerator (sample-and-aggregate + PTR)
│       │   ├── keyword_selector.py # FindBestK + TopKWithPTR
│       │   ├── pipeline.py        # DPKSAPipeline (subclass — see §11.2)
│       │   └── README.md
│       │
│       └── private_rag/           # 🚧 PENDING — all files are stubs
│           ├── __init__.py        #    (stub)
│           ├── retriever.py       #    (stub) BudgetAwareRetriever
│           ├── budget_manager.py  #    (stub) BudgetManager
│           └── README.md          #    (stub)
│
├── metrics/
│   ├── __init__.py                # Re-exports utility + privacy + zcdp
│   ├── utility.py                 # ✅ see §5.1 (expanded list)
│   ├── privacy.py                 # ✅ mia_auc_bleu, private_entity_leakage, …
│   ├── privacy_accounting.py            # ➕ RDP / advanced composition
│   ├── privacy_accounting_autodp.py     # ➕ AutodpLDAccountant (DPVoteRAG)
│   ├── privacy_accounting_zcdp.py       # ➕ zCDP composition (DP-SynRAG)
│   └── privacy_accounting_dpksa.py      # ➕ RDP composition (DP-KSA)
│
├── bench_datasets/                # ✅ + ➕ additional datasets
│   ├── __init__.py
│   ├── base_dataset.py            # BaseDataset
│   ├── natural_questions.py
│   ├── triviaqa.py
│   ├── hotpotqa.py
│   ├── squad.py
│   ├── chatdoctor.py              # medical domain (priority)
│   ├── searchqa.py                # ➕ (DP-SynRAG)
│   ├── movielens.py               # ➕ (DP-SynRAG)
│   └── medical_dirichlet_phi3.py  # ➕ DP-RAG dataset; DP-SynRAG "Medical Synth"
│
├── bench_llms/                    # ✅
│   ├── __init__.py
│   ├── base_llm.py                # BaseLLM
│   └── huggingface.py             # HuggingFaceLLM
│
├── experiments/
│   ├── configs/                   # YAML per experiment (+ _quick variants)
│   │   ├── vanilla.yaml           # ✅
│   │   ├── dp_rag.yaml / dp_rag_quick.yaml          # ✅
│   │   ├── dpvoterag.yaml / dpvoterag_quick.yaml    # ✅
│   │   ├── dpsparsevoterag.yaml / …_quick.yaml      # ✅
│   │   ├── dp_synrag.yaml         # ✅
│   │   ├── dpksa.yaml / dpksa_quick.yaml              # ✅
│   │   └── private_rag.yaml       # 🚧 (stub)
│   │
│   ├── index_cache.py             # ➕ FAISS cache keyed on DP hyperparams
│   ├── run_vanilla.py             # ✅
│   ├── run_dp_rag.py              # ✅
│   ├── run_dpvoterag.py           # ✅
│   ├── run_dpsparsevoterag.py     # ✅ (standalone script, not shared with vote)
│   ├── run_dp_synrag.py           # ✅
│   ├── run_dpksa.py               # ✅
│   ├── run_private_rag.py         # 🚧 (stub)
│   └── runner.py                  # central CLI — see §7.3
│
├── results/                       # NOT committed
├── analysis/
│   ├── plot_tradeoff.py
│   └── summary_table.py
│
├── tests/                         # see §9 (expanded list)
├── requirements.txt
├── README.md
└── setup.py
```

---

## 3. Abstract Classes (`core/`) — ✅ DONE

Each abstract class defines the **contract** fulfilled by both vanilla and every DP method. The current code respects exactly what is described here; maintain these signatures when implementing new methods.

### 3.1 `BaseChunker` (`core/chunker.py`)
- `chunk(document: str) -> list[str]`
- `get_config() -> dict`

Implementations: **Vanilla** `FixedSizeChunker`, `SentenceChunker`.

### 3.2 `BaseEmbedder` (`core/embedder.py`)
- `embed(texts: list[str]) -> np.ndarray` — shape `(len(texts), embedding_dim)`.
- `embed_query(query: str) -> np.ndarray` — shape `(embedding_dim,)`.

Implementations: **Vanilla** `HFEmbedder` (used by all current methods).

### 3.3 `BaseIndexer` (`core/indexer.py`)
- `build(vectors, metadata: list[dict])`
- `save(path: str)` / `load(path: str)`
- `get_index()`

Implementations: **Vanilla** `FAISSIndexer`. **DP-SynRAG (✅)** `DPSyntheticIndexer`.

### 3.4 `BaseRetriever` (`core/retriever.py`)
- `retrieve(query_vector, k: int) -> list[dict]`

Implementations: **Vanilla** `CosineSimilarityRetriever`. **DP-RAG (✅)** `DPRetriever` / `DPTopPRetriever`. **PRIVATE-RAG (🚧)** `BudgetAwareRetriever`.

### 3.5 `BaseReranker` (`core/reranker.py`)
- `rerank(query: str, chunks: list[dict]) -> list[dict]`

Implementations: **Vanilla** `IdentityReranker`. All current DP methods use `IdentityReranker`.

### 3.6 `BasePromptBuilder` (`core/prompt_builder.py`)
- `build(query: str, chunks: list[dict]) -> str`

Implementations: **Vanilla** `SimplePromptBuilder`. **DP-RAG (✅)** `PerDocPromptBuilder` (one prompt per doc + one public prompt; serialized as JSON string to preserve the `-> str` signature).

### 3.7 `BaseGenerator` (`core/generator.py`)
- `generate(prompt: str) -> str`
- `generate_with_dp(prompts: list[str], epsilon: float, delta: float) -> str` (may raise `NotImplementedError` in vanilla).

Implementations: **Vanilla** `HFGenerator`. **DP-RAG (✅)** `DPAggregatedGenerator`. **DPVoteRAG (✅)** `VoteRAGGenerator`. **DPSparseVoteRAG (✅)** `SparseVoteRAGGenerator`. **DP-SynRAG (✅)** uses vanilla `HFGenerator` for inference (DP happens at indexing time).

### 3.8 `BasePostprocessor` (`core/postprocessor.py`)
- `process(response: str) -> str`

Implementations: **Vanilla** `IdentityPostprocessor` (used by all current DP methods).

### 3.9 `RAGPipeline` (`core/pipeline.py`)
Orchestrates the injected components:
- `index(documents: list[str])` — Chunker → Embedder → Indexer.
- `query(query: str, k: int) -> str` — Embedder → Retriever → Reranker → PromptBuilder → Generator → Postprocessor.

`RAGPipeline` knows nothing about DP; it receives instances from outside. **Can be subclassed** when a method changes the `query()` flow (see §11.2).

---

## 4. DP Methods (`dp_methods/`)

Each module only overrides the components it modifies; the rest are taken from `vanilla/`.

### 4.1 DP-RAG (`dp_methods/per_query/dp_rag/`) — ✅ DONE

Paper: Grislain et al., Sarus Technologies (2024), arXiv:2412.19291 — repo `sarus-tech/dp-rag`.

Modified components: **Retriever + PromptBuilder + Generator** (three, not two).

**`DPRetriever` / `DPTopPRetriever`** (`retriever.py`): retrieves candidates by similarity and applies a DP threshold. `DPRetriever` = exponential mechanism top-k (selects k docs by count). `DPTopPRetriever` = exponential mechanism top-p (selects by weighted fraction p). Retrieval sensitivity = 1.

**`PerDocPromptBuilder`** (`prompt_builder.py`): builds **k+1 prompts** — one per retrieved private document, plus one **public** prompt (non-private context, configurable, empty by default) that the generator always treats as the last in the list. Serialized as JSON string to respect the `build(...) -> str` signature.

**`DPAggregatedGenerator`** (`generator.py`): DP-ICL (In-Context Learning) mechanism, token by token:
1. Gets logits per private prompt.
2. Per-token transformations on private logits: `l_norm → center → clip` (sensitivity = clipping parameter C; each doc contributes at most C to utility).
3. Public prompt logits enter as prior, weighted by θ, **without** transformation: `U_ICL(r) = θ·ln(L_pub(r)) + Σⱼ l_clipped_j(r)`.
4. Stochastic sampling via exponential mechanism: `r ∝ exp(ε·U_ICL/(2C))`.
5. Repeats until EOS or `max_tokens`.

Actual module parameters: `epsilon_retrieval`, `epsilon_generation`, `K` (docs selected for prompts), `max_docs` (cap after threshold filtering), `theta` (weight of the public prior), `delta` (passed to the external accountant only; individual mechanisms are pure ε-DP). Output per experiment: `results/dp_rag/eps_r{eps_r}_eps_g{eps_g}.json`.

Privacy: each mechanism is pure (ε,0)-DP; δ only appears when composing T tokens via RDP/advanced composition (`metrics/privacy_accounting.py`). δ = 1e-3 by default (paper). Cost: `O(K·T)` forward passes per query.

### 4.2 DPVoteRAG / DPSparseVoteRAG (`dp_methods/per_query/dpvoterag/`) — ✅ DONE

Paper: Koga et al., arXiv:2412.04697 — replica of the official repository `tacchan7412/DPRAG`.

Modified component: **Generator**. Also includes `pipeline.py` with `DPVoteRAGPipeline(RAGPipeline)` (permitted subclass — see §11.2) that recomputes the retrieval `k` as `m·k` and calls `generator.generate_vote(query, docs)`.

The `(ε,δ)` accounting is delegated to `metrics/privacy_accounting_autodp.py::AutodpLDAccountant` (EM-peeling + RDP composition + BBGHS conversion), a direct port of `utils/dp_accountant.py` from the official repository. The accountant is **online**: it stops generation when `epsilon_total`/`delta_total` is exhausted.

**`VoteRAGGenerator`** (`generator.py`) — **Limited Domain Gumbel** mechanism (Durfee & Rogers, NeurIPS 2019), per token:
1. Randomly partitions the `m·k` retrieved docs into `m` voters (k docs each).
2. Builds `m` prompts; each voter produces the next token by greedy decoding.
3. Vote histogram; counts are sorted and the top `k_bar` are taken.
4. Threshold `h_⊥ = h_{k_bar} + 1 + log(min(sens, k_bar, dim − k_bar)/δ)/ε`; Gumbel(0,1/ε) noise on counts and threshold.
5. If `max(v_counts) ≥ v_⊥` → winning token (success); otherwise → `NotFoundLDTop1` and `fail_mode='stop'` (paper default, arXiv:2412.04697v3 §2.1) halts generation (`exit_status=1`).
6. The online accountant records ε per token; if `epsilon_total`/`delta_total` is exceeded, stops (`exit_status=2`).

**`SparseVoteRAGGenerator`** (`sparse_vote_generator.py`) — extends the above with Sparse Vector Technique:
1. Generates a public token `y_nonRAG` without context (does not consume LD budget).
2. Measures agreement `a_t` with the private voters.
3. Compares `a_t + Lap(0,4/sparse_eps)` against `τ̂ = τ + Lap(0,2/sparse_eps)`.
4. If `a_t_noisy ≤ τ̂` → applies LD-Gumbel (private token, spends `epsilon_token`); if `>` → accepts the public token (no LD cost).
5. Optional exclusion of stopwords / question-words (not private).

Parameters: `m` (voters, def. 30), `k` (docs/voter, def. 1), `epsilon_token`/`delta_token` (2.0/1e-5), `epsilon_total`/`delta_total` (10.0/1e-4), `k_bar`, `sensitivity` (add/remove ⇒ 2.0), `tmax_absolute`, `n_peels`, `fail_mode` (def. `'stop'`), `random_seed`; Sparse adds `sparse_fraction` (def. 0.5, splits `epsilon_token` equally between SVT and LD per paper §3.3), `tau` (def. m/2), `exclude_stopwords`, `exclude_question_words`, `stopwords_path`. Output `last_run_stats`: `exit_status`, `tokens_generated`, `n_private_tokens`, `dp_spent_epsilon`, `dp_flags_per_token`, …

### 4.3 DP-KSA (`dp_methods/per_query/dpksa/`) — ✅ DONE

Paper: Tang, Flemings, Wang & Annavaram, arXiv:2602.14374 (2026). No official code
released; reimplemented from the paper with the thesis formalisation (Lemmas
KSA-1/2/3). **Per-query** method. Privacy unit & sensitivity (Lemma KSA-1): neighbors
`D ~ D'` differ in one document; the disjoint sample-and-aggregate partition + the
top-`N` refill means one of the `N` responses is **replaced**, so counts move by
≤1 *in either direction* (`‖ΔH‖∞ ≤ 1`) and gaps by ≤2 (`Δd_k = 2`, hence
`eps_em = 4/b`). Indexed documents are chunks, so the unit is one chunk.

Modified component: **Generator**, with a `DPKSAPipeline(RAGPipeline)` subclass
(permitted exception — see §11.2) that retrieves the top-`N` and delegates to
`generate_ksa(query, docs)`.

**`DPKSAGenerator`** (`generator.py`) runs `M_KSA(x, D)`:
1. **Sample-and-aggregate**: the top-`N` retrieved docs each generate a short
   response `y_i = F(D_i, x)` (disjoint partition: 1 doc → 1 response).
2. **Histogram** `H[w] = #{i : w ∈ y_i}` over normalised-word tokens
   (`keyword_selector.build_token_histogram`).
3. **FindBestK** (`keyword_selector.find_best_k`): `k̂ = argmax_k {d_k + r(k) +
   Gumbel(b)}` over the count gaps `d_k = H_(k) − H_(k+1)` for `k` in a **fixed,
   public** window `[k_min, k_max] ⊆ {1..N−1}` (data-independent candidate set;
   gaps read off the zero-padded sorted histogram); gap sensitivity `Δd_k = 2`
   (Lemma KSA-1).
4. **TopKWithPTR** (`keyword_selector.top_k_with_ptr`): **always runs** (paper
   Alg. 1) — Gaussian Propose-Test-Release `d̂ = max(2, d_k̂) + N(0,(2σ)²) −
   Φ(1−δ_ptr; 0, 2σ)`; releases the exact top-`k̂` keywords iff `d̂ > 2`, otherwise
   none (zero-shot fallback). No data-dependent early exit for degenerate
   histograms.
5. **Final generation**: `y = F(x, {w_i})` from the query + privately released
   keywords only; `D` is never re-accessed (genuine post-processing).

**Accounting** (`metrics/privacy_accounting_dpksa.py`): ε is an **input** budget.
The Gumbel scale `b` and PTR noise `σ` are calibrated to the target (ε, δ) by RDP
composition of the exponential mechanism (FindBestK) and the Gaussian PTR test,
converted to (ε,δ)-DP (reusing `_rdp_em` and the BBGHS conversion from
`privacy_accounting_autodp.py`). Total `δ = δ_ptr + δ_conv`.

> **Factor of 2**: the paper's main text (Alg. 3 Gumbel scale `2/eps_k`) omits the
> `Δd_k = 2` that its **own Appendix A.11** uses for the exponential-mechanism RDP —
> the paper is internally consistent at `Δ = 2`, not claiming `Δ = 1`. The accountant
> uses `eps_em = 2·Δd_k / b = 4/b`, consistent with Appendix A.11.

Module parameters: `target_epsilon`, `target_delta`, `n_docs` (N, paper 80),
`max_answer_tokens`, `max_final_tokens`, `k_min`/`k_max`, `ptr_delta_fraction`,
`ptr_noise_coupling`, `lowercase`, `exclude_stopwords`, `random_seed`. Output:
`results/dpksa/{dataset}/epsilon_{ε}.json`. Paper: ε ∈ {1,2,3,5,8}, δ = 1e-4,
generators Qwen2.5-3B / Llama-3.2-3B / Llama-3.1-8B.

### 4.4 PRIVATE-RAG (`dp_methods/per_query/private_rag/`) — 🚧 PENDING

> Implement **as described here**.

Modified components: **Retriever** + new component **BudgetManager**.

`BudgetManager` (`budget_manager.py`): auxiliary component (does not inherit from abstract classes). Maintains `{doc_id: remaining_budget}`. Exposes `can_query(doc_id) -> bool`, `consume(doc_id, cost: float)`, `reset()` / `save_state()` / `load_state()`. Supports compositional Rényi DP.
`BudgetAwareRetriever` (MURAG): top-K by similarity → filters docs with exhausted budget → returns only available ones.
`AdaptiveBudgetAwareRetriever` (MURAG-ADA): adaptive similarity threshold per query; retrieves until it obtains K docs that exceed the dynamic threshold and have remaining budget.

Module parameters: `epsilon_per_query`, `delta`, `initial_budget_per_doc`, `renyi_order`.

### 4.5 DP-SynRAG (`dp_methods/offline/dp_synrag/`) — ✅ DONE

Paper: Anonymous et al. (2025), arXiv:2510.06719v2. **Indexing-time** method: the DP budget is spent **once per dataset**; inference is non-private (`CosineSimilarityRetriever` + vanilla `HFGenerator` over the synthetic corpus). Decomposed into 6 components orchestrated by `DPSyntheticIndexer`:

**Stage 1a — `KeywordExtractor`**: per document, an LLM extracts K keywords restricted to a fixed vocabulary V. `Ext_i ∈ {0,1}^|V|`, `‖Ext_i‖₁ = K`. Noisy histogram `h'(D) = Σ Ext_i + N(0, σ_h² I)`. L2 sensitivity = √K.

**Stage 1b — `DPHistogramClusterer`**: selects the R most frequent keywords from `h'(D)`; assigns each doc to clusters in reverse frequency order, capped at L clusters per doc.

**Stage 1c — `DPClusterRetriever`**: per cluster, noisy centroid `μ_r = Σ E(d_i) + N(0, σ_μ² I)` (L2-normalised embeddings before summation); threshold via exponential mechanism `θ_s ~ ExpMech(u = −|#{sim≥θ_s} − k|, ε_θs)`; `S_r = {d_i : sim(E(d_i), μ_r) > θ_s}`.

**Stage 2d — `DPSyntheticGenerator`**: per cluster generates a synthetic doc token by token. `z_n = Σ clip_c(L_model(d_i, y_{<n}))`, `y ~ softmax(z_n/τ)` (τ=0 ⇒ argmax, paper default). `clip_c`: exponentiate (max-shifted, no sum-normalisation) → centre → L∞ clip to c.

**Stage 2e — `SelfFilter`** (optional, no DP cost): an evaluator LLM (without access to originals) decides YES/NO if the synthetic doc is task-relevant; pure post-processing on already-DP outputs.

The approved synthetic docs are embedded and indexed in FAISS as in vanilla.

**Privacy (zCDP, not Laplacian)**:
```
ρ = K/(2σ_h²) + L·[ ε_θs²/8 + 1/(2σ_μ²) + T/2·(c/τ)² ]
ε = ρ + √(4ρ·ln(1/δ))      # (ε,δ)-DP conversion
```
**Deliberate deviation — ρ_θs formula**: paper Theorem 1 writes `1/(8ε_θs²)`;
code uses `ε_θs²/8` (standard DP→zCDP bound for the exponential mechanism via the
bounded-range refinement, Cesar & Rogers 2021 on Bun & Steinke 2016). The paper's
form is dimensionally inverted (ρ must grow with ε_θs) — a Theorem 1 typo.
They agree at ε_θs=1 (calibration default) and diverge for sweeps. See
`metrics/privacy_accounting_zcdp.py` module docstring for full rationale.

**τ=0 removed**: greedy sampling (`τ≤1e-6`) now raises `ValueError` — the formula
`T/2·(c/τ)²` diverges and no valid privacy bound exists. Accountant: `metrics/privacy_accounting_zcdp.zcdp_dp_synrag`.

> **ε is an OUTPUT, not an input.** Unlike the other methods, no `epsilon_values` list is swept here: ε is *derived* from the zCDP accountant given `σ_h, σ_μ, ε_θs, c, τ, K, L, T, δ`. The runner prints the derived ε alongside the metrics.

Parameters: `K` (10), `R` (50), `L` (2), `σ_h`, `k` (10–15), `σ_μ`, `ε_θs`, `T` (30–70), `c` (1.0), `τ` (0.0), `δ` (1e-3). Paper embedder: `multi-qa-mpnet-base-dot-v1`. Paper LLMs: Phi-4-mini / Gemma-2 / Llama-3.1.

---

## 5. Metrics Module (`metrics/`)

### 5.1 Utility (`metrics/utility.py`) — ✅ + ➕
Each function takes `(predictions, references)` and returns a scalar.
- `match_accuracy` — normalised exact match (lowercase + whitespace).
- `contains_accuracy` ➕ — fraction where the reference is contained in the prediction (DP-SynRAG paper definition: gold answer / disease / movie title).
- `rouge_l` — ROUGE-L F-measure.
- `f1_score` ➕ — token-level F1 (replaces the original `bleu`).
- `bert_score_f1` — BERTScore F1 (formerly `bertscore_f1`).
- `recall_at_k` ➕, `answer_faithfulness` ➕, `answer_relevancy` ➕, `perplexity` ➕.
- `compute_utility_metrics` ➕ — aggregator that computes the subset requested by config.

> `bleu` **no longer exists** as a utility metric; BLEU survives internally only inside `metrics/privacy.py` for the MIA attack.

### 5.2 Privacy (`metrics/privacy.py`) — ✅
- `mia_auc_bleu(...)` — BLEU-based MIA attack (DPVoteRAG); AUC ∈ [0,1].
- `private_entity_leakage(responses, private_entities)` — counts leaked private entities.
- `bleu_membership_score` ➕, `privacy_budget_consumption` ➕, `compute_privacy_metrics` ➕.

### 5.3 Privacy Accounting (`metrics/privacy_accounting*.py`) — ➕ ADDED

Concept **absent in the original guide version** and central to the framework: ε is not just a config knob; for several methods it is a quantity **derived** from the noise/clipping parameters via a formal accountant.

- `privacy_accounting.py` — RDP / advanced / basic composition for DP-RAG (`ε_total ≤ ε_r + T·ε_g` as upper bound; RDP gives a tighter bound).
- `privacy_accounting_autodp.py` — `AutodpLDAccountant`: EM-peeling + RDP + BBGHS conversion for DPVoteRAG/Sparse (direct port from the official repository). **Online** accountant: cuts generation when `epsilon_total`/`delta_total` is exhausted.
- `privacy_accounting_zcdp.py` — `zcdp_dp_synrag` + `print_zcdp_analysis`: zCDP composition → (ε,δ) conversion for DP-SynRAG.

Practical rule: **DP-RAG and DPVote** consume ε as budget (input) and the accountant verifies/composes; **DP-SynRAG** produces ε as output of the zCDP accountant.

---

## 6. Datasets Module (`bench_datasets/`)

### 6.1 `BaseDataset` (`base_dataset.py`) — ✅
Abstract methods (actual signature, unchanged from the original design):
- `load() -> None`
- `get_documents() -> list[str]`
- `get_queries() -> list[dict]` — each item with keys `query` and `answer`.
- `get_private_entities() -> list[str]` — empty for non-private datasets.

### 6.2 Concrete datasets — ✅ + ➕
- `natural_questions.py`, `triviaqa.py`, `hotpotqa.py`, `squad.py`
- `chatdoctor.py` — medical domain (**priority**)
- ➕ `searchqa.py`, `movielens.py`, `medical_dirichlet_phi3.py` — extra datasets
  (`medical_dirichlet_phi3` is DP-RAG's sarus-tech dataset, reused by DP-SynRAG as *Medical Synth*).

### 6.3 LLMs Module (`bench_llms/`) — ✅
- `BaseLLM` exposes the interface to build the pipeline generator.
- `HuggingFaceLLM` takes `model_name`. Selection via `llm.name` in YAML.

---

## 7. Experiments (`experiments/`)

### 7.1 YAML Configuration — actual schema

> The flat schema from the original design (`epsilon_values`, `metrics:{utility,privacy}`, `output_dir`) **was replaced** by nested blocks. Use this schema when creating new configs (including the private_rag stub YAML).

```yaml
# Real example: experiments/configs/vanilla.yaml
method: vanilla
dataset:
  name: squad                 # squad|chatdoctor|natural_questions|triviaqa|hotpotqa|…
  index_split: train
  test_split: validation
  max_index_docs: 200
  max_test_queries: 25
  docs_path:                  # optional: local jsonl
  queries_path:
llm:
  name: huggingface
  model_name: sshleifer/tiny-gpt2
models:
  embedder: sentence-transformers/all-MiniLM-L6-v2
pipeline:
  chunk_size: 120
  chunk_overlap: 20
  k_retrieved: 2
generation:
  max_new_tokens: 48
random_seed: 42
output_path: results/vanilla/vanilla_squad.json
```

- Methods with ε sweeps (DP-RAG, DPVote) include their own parameter blocks (e.g. `epsilon_retrieval`, `epsilon_generation`, `m`, `k`, `epsilon_token`, …).
- **DP-SynRAG** does not include `epsilon_values`: it includes `keyword_extractor / clustering / cluster_retrieval / synthetic_generation / self_filter / delta` and ε is printed as derived (see §4.6).
- Each config has a `_quick` variant for smoke runs.

### 7.2 Per-Method Scripts (`experiments/run_*.py`) — ✅
Each script: loads its YAML, instantiates components (vanilla + DP where applicable), builds the pipeline, iterates the DP grid from the config, indexes, runs queries, computes metrics, and saves JSON to `results/{method}/...`. Direct invocation:

```
python experiments/run_dp_rag.py --config experiments/configs/dp_rag.yaml [--verbose] [--verbose-max-examples N]
```

`run_dpsparsevoterag.py` is a **standalone** script (does not share with `run_dpvoterag.py`).

### 7.3 Central Runner (`experiments/runner.py`) — 🚧 PENDING

> Implement **as described here**. Today experiments are launched by invoking each `run_*.py` directly; the central CLI is missing.

CLI with: `--method` (or `all`), `--dataset` (or `all`), `--llm-name`, `--llm-model`, `--config` (optional, overrides defaults), `--dry-run`. Internally imports each `run_*.py` and calls its `main()` (or launches it as a subprocess).

---

## 8. Analysis and Visualisation (`analysis/`)

- `plot_tradeoff.py` — reads `results/`, generates ε vs utility and ε vs privacy (AUC-MIA) curves per dataset; figures in `results/figures/`.
- `summary_table.py` — CSV/LaTeX table: Method, Dataset, ε, MatchAccuracy, BERTScore F1, ROUGE-L, MIA-AUC, EntityLeakage. For DP-SynRAG use the **derived** ε from the accountant.

---

## 9. Tests (`tests/`)

pytest, no exhaustive coverage. Actual state (✅ exist; original design tests that were missing have been expanded):
- `test_core.py`, `test_vanilla.py`, `test_metrics.py`, `test_datasets.py` — ✅
- ➕ `test_dp_methods.py`, `test_dpvoterag.py`, `test_limited_domain_gumbel.py`, `test_llms.py`, `test_runner_smoke.py`, `test_seed_reproducibility.py`

When implementing PRIVATE-RAG, add its tests following the pattern in `test_dp_methods.py`. (DP-KSA tests live in `test_dpksa.py`.)

---

## 10. Implementation TO-DO List

Actual state per block:

- **BLOCK 0** Setup — ✅ DONE
- **BLOCK 1** Abstract classes `core/` — ✅ DONE (criterion: `@abstractmethod` + pipeline with mocks)
- **BLOCK 2** Vanilla — ✅ DONE
- **BLOCK 3** Datasets — ✅ DONE (+ 4 extra DP-SynRAG datasets)
- **BLOCK 4** Metrics — ✅ DONE (+ privacy accounting, §5.3)
- **BLOCK 5** DP Methods:
  - 5.1 **DP-RAG** — ✅ DONE
  - 5.2 **DPVoteRAG / DPSparseVoteRAG** — ✅ DONE
  - 5.3 **DP-KSA** — ✅ DONE
  - 5.4 **PRIVATE-RAG** — 🚧 PENDING (`BudgetManager` Rényi → `BudgetAwareRetriever` (MURAG) → `AdaptiveBudgetAwareRetriever` (MURAG-ADA) → run + yaml)
  - 5.5 **DP-SynRAG** — ✅ DONE
- **BLOCK 6** Runner + tests:
  - 6.1 `runner.py` — 🚧 PENDING
  - 6.3–6.5 core/vanilla tests — ✅ DONE
- **BLOCK 7** Full experiments — pending (depends on completing Block 5)
- **BLOCK 8** Analysis and figures — `plot_tradeoff.py` / `summary_table.py` present; manual validation pending

Suggested next steps: 5.4 PRIVATE-RAG → Block 7 (full experiments) → Block 8 (analysis and figures).

---

## 11. Important Notes for the Agent

1. **Strict block order**: do not start Block 5 (pending methods) without a working Block 2. Block 4 can run in parallel with Block 3. (Both ✅.)

2. **Dependency injection — with explicit exception**: `RAGPipeline` never imports directly from `vanilla/` or `dp_methods/`; it always receives instances from outside. **Permitted exception**: when a method changes the *flow* of `query()` (not just a component), **subclassing** `RAGPipeline` inside the method's module is allowed. Reference case: `dp_methods/per_query/dpvoterag/pipeline.py::DPVoteRAGPipeline`, which recomputes the retrieval `k` as `m·k` and delegates to `generator.generate_vote(...)` because voting does not fit the standard `retrieve→generate` signature. The subclass still receives its components injected from outside; only the orchestration changes. Maintain this pattern (subclass of `RAGPipeline`, not a parallel ad-hoc pipeline) for future methods that require it.

3. **DPSparseVoteRAG — two deliberate Sparse-path choices (paper, not reference code)**:
   - *Budget split*: `epsilon_token` is split via `sparse_fraction` (default 0.5 = equal split per paper §3.3). Derived `sparse_eps = epsilon_token × sparse_fraction`; LD epsilon = `epsilon_token × (1 − sparse_fraction)`. The reference repo instead exposes `dp_eps` and `sparse_eps` as independent knobs; deriving both from one `epsilon_token` keeps the total per-step budget comparable across DPVoteRAG/DPSparseVoteRAG for fair curves. Design decision, not a bug.
   - *SVT accounting*: the SVT is charged **per Above-Threshold epoch** (Lyu-Su-Li 2017) — `sparse_eps` once per threshold instantiation (initial + each resample after a private token), via `accountant.account_svt_step`. Below-threshold comparisons and the agreement-count Laplace `Lap(4/sparse_eps)` are covered by the same epoch budget and not charged separately. The reference draws this Laplace but never composes its cost online; we do, which is strictly more conservative (reported ε stays a valid upper bound).

4. **Reproducibility**: all experiments fix `random_seed`, propagated to PyTorch/NumPy/Python. Additionally a per-query RNG derived from `hash(query)` is used for independence between queries (see tests `test_seed_reproducibility.py`). Maintain this convention in new methods.

5. **Scalability**: DPVoteRAG requires K×M inferences per query (pilot: K=5, M=2 max; report time/query). DP-KSA runs `N` per-document generations + one final generation per query (paper N=80; reduce `n_docs` for smoke runs). DP-SynRAG has the highest offline cost (one forward per doc in extraction + `O(R·max|S_r|·T)` in generation) — use `max_samples` and the FAISS cache (`experiments/index_cache.py`, keyed on DP hyperparams) to avoid rebuilds.

6. **ε as input vs output**: DP-RAG, DPVote and DP-KSA consume ε as budget (the accountant composes/verifies); **DP-SynRAG derives ε** from the zCDP accountant given the noise parameters. The analysis (§8) must read the correct ε for each method.
