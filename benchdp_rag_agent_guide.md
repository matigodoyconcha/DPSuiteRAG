# BenchDP-RAG — Guía de Implementación para Agente

> Este documento describe **cómo estructurar y construir** el framework de evaluación de métodos de Privacidad Diferencial en sistemas RAG. Es la **base oficial** de implementación: lo que aún no está construido se implementa **como sale aquí**; lo que ya está construido se documenta aquí tal como quedó en el código (el código es la fuente de verdad para los métodos ya implementados).

> **Convención de estado.** Cada bloque/método lleva un marcador:
> - `✅ HECHO` — implementado y la guía describe el código real.
> - `🚧 PENDIENTE` — todavía es solo especificación; impleméntalo **como sale en esta guía**.
> - `➕ AÑADIDO` — existe en el código y no estaba en la versión original de la guía; aquí queda documentado.

---

## 1. Propósito del Proyecto

Construir un framework Python modular, reproducible y extensible que permita:

1. Ejecutar un pipeline RAG completo con componentes intercambiables.
2. Substituir cualquier componente del pipeline por una versión que aplique Privacidad Diferencial (DP).
3. Evaluar y comparar 6 métodos del estado del arte bajo condiciones homogéneas.
4. Producir tablas y curvas del trade-off privacidad-utilidad para distintos valores de ε.

Los 6 métodos: **LPRAG, DP-RAG, DPVoteRAG/DPSparseVoteRAG, RP-RAG, PRIVATE-RAG, DP-SynRAG**.

| Método | Estado |
|---|---|
| DP-RAG | ✅ HECHO |
| DPVoteRAG / DPSparseVoteRAG | ✅ HECHO |
| DP-SynRAG | ✅ HECHO |
| LPRAG | 🚧 PENDIENTE |
| RP-RAG | 🚧 PENDIENTE |
| PRIVATE-RAG | 🚧 PENDIENTE |

---

## 2. Árbol de Directorios

Estado real del repositorio. `(stub)` = archivo vacío que aún debe implementarse según esta guía.

```
benchdp_rag/
│
├── core/                          # ✅ Clases abstractas del pipeline RAG
│   ├── __init__.py
│   ├── chunker.py                 # BaseChunker
│   ├── embedder.py                # BaseEmbedder
│   ├── indexer.py                 # BaseIndexer
│   ├── retriever.py               # BaseRetriever
│   ├── reranker.py                # BaseReranker
│   ├── prompt_builder.py          # BasePromptBuilder
│   ├── generator.py               # BaseGenerator
│   ├── postprocessor.py           # BasePostprocessor
│   └── pipeline.py                # RAGPipeline (puede subclasearse — ver §11.2)
│
├── vanilla/                       # ✅ Implementación RAG estándar (sin DP)
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
│   ├── lprag/                     # 🚧 PENDIENTE — todos los archivos son stubs
│   │   ├── __init__.py            #    (stub)
│   │   ├── chunker.py             #    (stub) EntityPerturbedChunker
│   │   └── README.md              #    (stub)
│   │
│   ├── dp_rag/                    # ✅ HECHO
│   │   ├── __init__.py
│   │   ├── retriever.py           # DPRetriever, DPTopPRetriever
│   │   ├── prompt_builder.py      # ➕ PerDocPromptBuilder
│   │   ├── generator.py           # DPAggregatedGenerator (DP-ICL)
│   │   └── README.md
│   │
│   ├── dpvoterag/                 # ✅ HECHO
│   │   ├── __init__.py
│   │   ├── generator.py           # VoteRAGGenerator (Limited Domain Gumbel)
│   │   ├── sparse_vote_generator.py  # SparseVoteRAGGenerator (LD + SVT)
│   │   ├── pipeline.py            # ➕ DPVoteRAGPipeline (subclase — ver §11.2)
│   │   └── README.md
│   │
│   ├── rp_rag/                    # 🚧 PENDIENTE — todos los archivos son stubs
│   │   ├── __init__.py            #    (stub)
│   │   ├── indexer.py             #    (stub) RandomProjectionIndexer
│   │   ├── retriever.py           #    (stub) RPRetriever
│   │   └── README.md              #    (stub)
│   │
│   ├── private_rag/               # 🚧 PENDIENTE — todos los archivos son stubs
│   │   ├── __init__.py            #    (stub)
│   │   ├── retriever.py           #    (stub) BudgetAwareRetriever
│   │   ├── budget_manager.py      #    (stub) BudgetManager
│   │   └── README.md              #    (stub)
│   │
│   └── dp_synrag/                 # ✅ HECHO — descompuesto en 6 componentes
│       ├── __init__.py
│       ├── keyword_extractor.py   # KeywordExtractor (Stage 1a)
│       ├── clusterer.py           # DPHistogramClusterer (Stage 1a+1b)
│       ├── cluster_retriever.py   # DPClusterRetriever (Stage 1c)
│       ├── synthetic_generator.py # DPSyntheticGenerator (Stage 2d)
│       ├── self_filter.py         # SelfFilter (Stage 2e)
│       ├── indexer.py             # DPSyntheticIndexer (orquesta 1a→2e)
│       ├── generator.py           # (stub vestigial — inferencia usa HFGenerator)
│       └── README.md
│
├── metrics/
│   ├── __init__.py                # Re-exporta utility + privacy + zcdp
│   ├── utility.py                 # ✅ ver §5.1 (lista ampliada)
│   ├── privacy.py                 # ✅ mia_auc_bleu, private_entity_leakage, …
│   ├── privacy_accounting.py            # ➕ RDP / advanced composition
│   ├── privacy_accounting_autodp.py     # ➕ AutodpLDAccountant (DPVoteRAG)
│   └── privacy_accounting_zcdp.py       # ➕ zCDP composition (DP-SynRAG)
│
├── bench_datasets/                # ✅ + ➕ datasets extra
│   ├── __init__.py
│   ├── base_dataset.py            # BaseDataset
│   ├── natural_questions.py
│   ├── triviaqa.py
│   ├── hotpotqa.py
│   ├── squad.py
│   ├── chatdoctor.py              # dominio médico (prioritario)
│   ├── searchqa.py                # ➕ (DP-SynRAG)
│   ├── movielens.py               # ➕ (DP-SynRAG)
│   ├── medical_synth.py           # ➕ (DP-SynRAG)
│   └── medical_dirichlet_phi3.py  # ➕ (DP-SynRAG)
│
├── bench_llms/                    # ✅
│   ├── __init__.py
│   ├── base_llm.py                # BaseLLM
│   └── huggingface.py             # HuggingFaceLLM
│
├── experiments/
│   ├── configs/                   # YAML por experimento (+ variantes _quick)
│   │   ├── vanilla.yaml           # ✅
│   │   ├── dp_rag.yaml / dp_rag_quick.yaml          # ✅
│   │   ├── dpvoterag.yaml / dpvoterag_quick.yaml    # ✅
│   │   ├── dpsparsevoterag.yaml / …_quick.yaml      # ✅
│   │   ├── dp_synrag.yaml         # ✅
│   │   ├── lprag.yaml             # 🚧 (stub)
│   │   ├── rp_rag.yaml            # 🚧 (stub)
│   │   └── private_rag.yaml       # 🚧 (stub)
│   │
│   ├── index_cache.py             # ➕ caché FAISS keyed en hiperparams DP
│   ├── run_vanilla.py             # ✅
│   ├── run_dp_rag.py              # ✅
│   ├── run_dpvoterag.py           # ✅
│   ├── run_dpsparsevoterag.py     # ✅ (script propio, no compartido con vote)
│   ├── run_dp_synrag.py           # ✅
│   ├── run_lprag.py               # 🚧 (stub)
│   ├── run_rp_rag.py              # 🚧 (stub)
│   ├── run_private_rag.py         # 🚧 (stub)
│   └── runner.py                  # 🚧 (stub) CLI central — ver §7.3
│
├── results/                       # NO se commitea
├── analysis/
│   ├── plot_tradeoff.py
│   └── summary_table.py
│
├── tests/                         # ver §9 (lista ampliada)
├── requirements.txt
├── README.md
└── setup.py
```

---

## 3. Clases Abstractas (`core/`) — ✅ HECHO

Cada clase abstracta define el **contrato** que cumplen tanto vanilla como cada método DP. El código actual respeta exactamente lo descrito aquí; mantener estas firmas al implementar nuevos métodos.

### 3.1 `BaseChunker` (`core/chunker.py`)
- `chunk(document: str) -> list[str]`
- `get_config() -> dict`

Implementaciones: **Vanilla** `FixedSizeChunker`, `SentenceChunker`. **LPRAG (🚧)** `EntityPerturbedChunker`.

### 3.2 `BaseEmbedder` (`core/embedder.py`)
- `embed(texts: list[str]) -> np.ndarray` — shape `(len(texts), embedding_dim)`.
- `embed_query(query: str) -> np.ndarray` — shape `(embedding_dim,)`.

Implementaciones: **Vanilla** `HFEmbedder`. RP-RAG usa el embedder vanilla (la proyección va en el Indexer).

### 3.3 `BaseIndexer` (`core/indexer.py`)
- `build(vectors, metadata: list[dict])`
- `save(path: str)` / `load(path: str)`
- `get_index()`

Implementaciones: **Vanilla** `FAISSIndexer`. **RP-RAG (🚧)** `RandomProjectionIndexer`. **DP-SynRAG (✅)** `DPSyntheticIndexer`.

### 3.4 `BaseRetriever` (`core/retriever.py`)
- `retrieve(query_vector, k: int) -> list[dict]`

Implementaciones: **Vanilla** `CosineSimilarityRetriever`. **DP-RAG (✅)** `DPRetriever` / `DPTopPRetriever`. **RP-RAG (🚧)** `RPRetriever`. **PRIVATE-RAG (🚧)** `BudgetAwareRetriever`.

### 3.5 `BaseReranker` (`core/reranker.py`)
- `rerank(query: str, chunks: list[dict]) -> list[dict]`

Implementaciones: **Vanilla** `IdentityReranker`. Todos los métodos DP actuales usan `IdentityReranker`.

### 3.6 `BasePromptBuilder` (`core/prompt_builder.py`)
- `build(query: str, chunks: list[dict]) -> str`

Implementaciones: **Vanilla** `SimplePromptBuilder`. **DP-RAG (✅)** `PerDocPromptBuilder` (un prompt por doc + un prompt público; se serializan como JSON string para mantener la firma `-> str`).

### 3.7 `BaseGenerator` (`core/generator.py`)
- `generate(prompt: str) -> str`
- `generate_with_dp(prompts: list[str], epsilon: float, delta: float) -> str` (puede levantar `NotImplementedError` en vanilla).

Implementaciones: **Vanilla** `HFGenerator`. **DP-RAG (✅)** `DPAggregatedGenerator`. **DPVoteRAG (✅)** `VoteRAGGenerator`. **DPSparseVoteRAG (✅)** `SparseVoteRAGGenerator`. **DP-SynRAG (✅)** usa `HFGenerator` vanilla en inferencia (la DP ocurre en indexación).

### 3.8 `BasePostprocessor` (`core/postprocessor.py`)
- `process(response: str) -> str`

Implementaciones: **Vanilla** `IdentityPostprocessor` (usado por todos los métodos DP actuales).

### 3.9 `RAGPipeline` (`core/pipeline.py`)
Orquesta los componentes inyectados:
- `index(documents: list[str])` — Chunker → Embedder → Indexer.
- `query(query: str, k: int) -> str` — Embedder → Retriever → Reranker → PromptBuilder → Generator → Postprocessor.

`RAGPipeline` no sabe nada de DP; recibe instancias desde afuera. **Puede subclasearse** cuando un método cambia el flujo de `query()` (ver §11.2).

---

## 4. Métodos DP (`dp_methods/`)

Cada módulo solo sobreescribe los componentes que modifica; el resto se toma de `vanilla/`.

### 4.1 LPRAG (`dp_methods/lprag/`) — 🚧 PENDIENTE

> Implementar **como sale aquí**.

Componente modificado: **Chunker**. `EntityPerturbedChunker` debe:
1. Aplicar chunking estándar (puede delegar en `FixedSizeChunker`).
2. Detectar entidades sensibles por chunk (personas, lugares, diagnósticos) con un NER.
3. Para cada entidad, aplicar el mecanismo exponencial con parámetro ε y sustituirla por una alternativa semánticamente similar.
4. Devolver los chunks perturbados.

Parámetros del módulo: `epsilon`, `entity_types`, `candidate_model`.

### 4.2 DP-RAG (`dp_methods/dp_rag/`) — ✅ HECHO

Paper: Grislain et al., Sarus Technologies (2024), arXiv:2412.19291 — repo `sarus-tech/dp-rag`.

Componentes modificados: **Retriever + PromptBuilder + Generator** (tres, no dos).

**`DPRetriever` / `DPTopPRetriever`** (`retriever.py`): recupera candidatos por similitud y aplica un umbral DP. `DPRetriever` = mecanismo exponencial top-k (selecciona k docs por conteo). `DPTopPRetriever` = mecanismo exponencial top-p (selecciona por fracción ponderada p). Sensibilidad de recuperación = 1.

**`PerDocPromptBuilder`** (`prompt_builder.py`): construye **k+1 prompts** — uno por documento privado recuperado, más un prompt **público** (contexto no privado, configurable, por defecto vacío) que el generador trata siempre como el último de la lista. Se serializan como JSON string para respetar la firma `build(...) -> str`.

**`DPAggregatedGenerator`** (`generator.py`): mecanismo DP-ICL (In-Context Learning), token a token:
1. Obtiene logits por prompt privado.
2. Transformaciones por token sobre logits privados: `l_norm → center → clip` (sensibilidad = parámetro de clipping C; cada doc aporta ≤ C a la utilidad).
3. Los logits del prompt público entran como prior, ponderados por θ, **sin** transformación: `U_ICL(r) = θ·ln(L_pub(r)) + Σⱼ l_clipped_j(r)`.
4. Muestreo estocástico vía mecanismo exponencial: `r ∝ exp(ε·U_ICL/(2C))`.
5. Repite hasta EOS o `max_tokens`.

Parámetros reales del módulo: `epsilon_retrieval`, `epsilon_generation`, `K` (docs seleccionados para los prompts), `max_docs` (tope tras filtrado por umbral), `theta` (peso del prior público), `delta` (solo se pasa al accountant externo; los mecanismos individuales son ε-DP puros). Salida por experimento: `results/dp_rag/eps_r{eps_r}_eps_g{eps_g}.json`.

Privacidad: cada mecanismo es (ε,0)-DP puro; δ aparece solo al componer T tokens vía RDP/composición avanzada (`metrics/privacy_accounting.py`). δ = 1e-3 por defecto (paper). Coste: `O(K·T)` forward passes por query.

### 4.3 DPVoteRAG / DPSparseVoteRAG (`dp_methods/dpvoterag/`) — ✅ HECHO

Paper: Koga et al., arXiv:2412.04697 — réplica del repo oficial `tacchan7412/DPRAG`.

Componente modificado: **Generator**. Además trae `pipeline.py` con `DPVoteRAGPipeline(RAGPipeline)` (subclase permitida — ver §11.2) que recalcula el `k` de recuperación como `m·k` y llama `generator.generate_vote(query, docs)`.

El accounting `(ε,δ)` se delega en `metrics/privacy_accounting_autodp.py::AutodpLDAccountant` (EM-peeling + composición RDP + conversión BBGHS), port directo de `utils/dp_accountant.py` del repo oficial.

**`VoteRAGGenerator`** (`generator.py`) — mecanismo **Limited Domain Gumbel** (Durfee & Rogers, NeurIPS 2019), por token:
1. Particiona aleatoriamente los `m·k` docs recuperados en `m` voters (k docs c/u).
2. Construye `m` prompts; cada voter produce el siguiente token por greedy decoding.
3. Histograma de votos; se ordenan los conteos y se toman los top `k_bar`.
4. Umbral `h_⊥ = h_{k_bar} + 1 + log(min(sens, k_bar, dim − k_bar)/δ)/ε`; ruido Gumbel(0,1/ε) sobre conteos y umbral.
5. Si `max(v_counts) ≥ v_⊥` → token ganador (success); si no → `NotFoundLDTop1` y el comportamiento depende de `fail_mode` (`'ld_pate'` default: reintenta con `k_bar = dim−1`; `'stop'`: detiene).
6. El accountant online registra ε por token; si excede `epsilon_total`/`delta_total`, detiene (`exit_status=2`).

**`SparseVoteRAGGenerator`** (`sparse_vote_generator.py`) — extiende lo anterior con Sparse Vector Technique:
1. Genera token público `y_nonRAG` sin contexto (no consume budget LD).
2. Mide acuerdo `a_t` con los voters privados.
3. Compara `a_t + Lap(0,4/sparse_eps)` contra `τ̂ = τ + Lap(0,2/sparse_eps)`.
4. Si `a_t_noisy ≤ τ̂` → aplica LD-Gumbel (token privado, gasta `epsilon_token`); si `>` → acepta el token público (sin coste LD).
5. Exclusión opcional de stopwords / question-words (no privadas).

Parámetros: `m` (voters, def. 30), `k` (docs/voter, def. 1), `epsilon_token`/`delta_token` (2.0/1e-5), `epsilon_total`/`delta_total` (10.0/1e-4), `k_bar`, `sensitivity` (add/remove ⇒ 2.0), `tmax_absolute`, `n_peels`, `fail_mode`, `random_seed`; Sparse añade `sparse_eps` (def. 1.0), `tau` (def. m/2), `exclude_stopwords`, `exclude_question_words`, `stopwords_path`. Salida `last_run_stats`: `exit_status`, `tokens_generated`, `n_private_tokens`, `dp_spent_epsilon`, `dp_flags_per_token`, …

> Limitación formal conocida (DPSparseVoteRAG): el ahorro de budget cuando el token coincide con el LLM público sigue la convención del repo oficial (`--dp_eps`, `--sparse_eps` independientes). Para reproducir el paper estrictamente: `sparse_eps = epsilon_token/2`. Documentado en el README del módulo.

### 4.4 RP-RAG (`dp_methods/rp_rag/`) — 🚧 PENDIENTE

> Implementar **como sale aquí**.

Componentes modificados: **Indexer + Retriever**.

`RandomProjectionIndexer`: genera una matriz gaussiana aleatoria R `(d_original, d_projected)` **una sola vez**, la persiste junto al índice, y aplica `v' = v @ R` a cada embedding antes de añadirlo a FAISS.
`RPRetriever`: aplica la misma R al vector de query y busca por similitud en el espacio proyectado.

**No tiene garantías DP formales** (no hay ε ni δ). Documentarlo explícitamente en el README y tratarlo como baseline heurístico; en la tabla comparativa sus columnas de ε van "N/A".

Parámetros del módulo: `d_projected`, `random_seed`.

### 4.5 PRIVATE-RAG (`dp_methods/private_rag/`) — 🚧 PENDIENTE

> Implementar **como sale aquí**.

Componentes modificados: **Retriever** + componente nuevo **BudgetManager**.

`BudgetManager` (`budget_manager.py`): auxiliar (no hereda de clases abstractas). Mantiene `{doc_id: remaining_budget}`. Expone `can_query(doc_id) -> bool`, `consume(doc_id, cost: float)`, `reset()` / `save_state()` / `load_state()`. Soporte para Rényi DP composicional.
`BudgetAwareRetriever` (MURAG): top-K por similitud → filtra docs con budget agotado → devuelve solo los disponibles.
`AdaptiveBudgetAwareRetriever` (MURAG-ADA): umbral de similitud adaptativo por consulta; recupera hasta obtener K que superen el umbral dinámico y tengan budget.

Parámetros del módulo: `epsilon_per_query`, `delta`, `initial_budget_per_doc`, `renyi_order`.

### 4.6 DP-SynRAG (`dp_methods/dp_synrag/`) — ✅ HECHO

Paper: Anonymous et al. (2025), arXiv:2510.06719v2. Método **indexing-time**: el budget DP se gasta **una vez por dataset**; la inferencia es no privada (`CosineSimilarityRetriever` + `HFGenerator` vanilla sobre el corpus sintético). Descompuesto en 6 componentes orquestados por `DPSyntheticIndexer`:

**Stage 1a — `KeywordExtractor`**: por documento, un LLM extrae K keywords restringidas a un vocabulario fijo V. `Ext_i ∈ {0,1}^|V|`, `‖Ext_i‖₁ = K`. Histograma ruidoso `h'(D) = Σ Ext_i + N(0, σ_h² I)`. Sensibilidad L2 = √K.

**Stage 1b — `DPHistogramClusterer`**: selecciona los R keywords más frecuentes de `h'(D)`; asigna cada doc a clusters en orden inverso de frecuencia, con tope L clusters por doc.

**Stage 1c — `DPClusterRetriever`**: por cluster, centroide ruidoso `μ_r = Σ E(d_i) + N(0, σ_μ² I)` (embeddings L2-normalizados antes de sumar); umbral por mecanismo exponencial `θ_s ~ ExpMech(u = −|#{sim≥θ_s} − k|, ε_θs)`; `S_r = {d_i : sim(E(d_i), μ_r) > θ_s}`.

**Stage 2d — `DPSyntheticGenerator`**: por cluster genera un doc sintético token a token. `z_n = Σ clip_c(L_model(d_i, y_{<n}))`, `y ~ softmax(z_n/τ)` (τ=0 ⇒ argmax, default del paper). `clip_c`: exponenciar+normalizar → centrar → clip L∞ a c.

**Stage 2e — `SelfFilter`** (opcional, sin coste DP): un LLM evaluador (sin acceso a originales) decide YES/NO si el doc sintético es task-relevant; post-procesamiento puro sobre salidas ya DP.

Los docs sintéticos aprobados se embeben e indexan en FAISS como vanilla.

**Privacidad (zCDP, no Laplaciano)**:
```
ρ = K/(2σ_h²) + L·[ 1/(8ε_θs²) + 1/(2σ_μ²) + T/2·(c/τ)² ]
ε = ρ + √(4ρ·ln(1/δ))      # conversión (ε,δ)-DP
```
Modo greedy (τ=0): el término de token diverge → degenera a Report-Noisy-Max, manejado con fallback `ρ_token = T·c²/2`. Accountant: `metrics/privacy_accounting_zcdp.zcdp_dp_synrag`.

> **ε es SALIDA, no entrada.** A diferencia de los demás métodos, aquí no se barre una lista `epsilon_values`: ε se *deriva* del accountant zCDP a partir de `σ_h, σ_μ, ε_θs, c, τ, K, L, T, δ`. El runner imprime el ε derivado junto a las métricas.

Parámetros: `K` (10), `R` (50), `L` (2), `σ_h`, `k` (10–15), `σ_μ`, `ε_θs`, `T` (30–70), `c` (1.0), `τ` (0.0), `δ` (1e-3). Embedder paper: `multi-qa-mpnet-base-dot-v1`. LLMs paper: Phi-4-mini / Gemma-2 / Llama-3.1.

---

## 5. Módulo de Métricas (`metrics/`)

### 5.1 Utilidad (`metrics/utility.py`) — ✅ + ➕
Cada función recibe `(predictions, references)` y devuelve un escalar.
- `match_accuracy` — exact match normalizado (lower + whitespace).
- `contains_accuracy` ➕ — fracción donde la referencia está contenida en la predicción (definición del paper DP-SynRAG: gold answer / disease / movie title).
- `rouge_l` — F-measure ROUGE-L.
- `f1_score` ➕ — F1 token-level (reemplaza al `bleu` original).
- `bert_score_f1` — F1 de BERTScore (antes `bertscore_f1`).
- `recall_at_k` ➕, `answer_faithfulness` ➕, `answer_relevancy` ➕, `perplexity` ➕.
- `compute_utility_metrics` ➕ — agregador que computa el subconjunto pedido por config.

> `bleu` ya **no existe** como métrica de utilidad; el BLEU sobrevive internamente solo dentro de `metrics/privacy.py` para el ataque MIA.

### 5.2 Privacidad (`metrics/privacy.py`) — ✅
- `mia_auc_bleu(...)` — ataque MIA basado en BLEU (DPVoteRAG); AUC ∈ [0,1].
- `private_entity_leakage(responses, private_entities)` — cuenta entidades privadas filtradas.
- `bleu_membership_score` ➕, `privacy_budget_consumption` ➕, `compute_privacy_metrics` ➕.

### 5.3 Privacy Accounting (`metrics/privacy_accounting*.py`) — ➕ AÑADIDO

Concepto **ausente en la versión original de la guía** y central al framework: ε no es solo una perilla de config; para varios métodos es una cantidad **derivada** de los parámetros de ruido/clipping vía un accountant formal.

- `privacy_accounting.py` — composición RDP / avanzada / básica para DP-RAG (`ε_total ≤ ε_r + T·ε_g` como cota superior; RDP da cota más ajustada).
- `privacy_accounting_autodp.py` — `AutodpLDAccountant`: EM-peeling + RDP + conversión BBGHS para DPVoteRAG/Sparse (port del repo oficial). Accountant **online**: corta la generación al agotar `epsilon_total`/`delta_total`.
- `privacy_accounting_zcdp.py` — `zcdp_dp_synrag` + `print_zcdp_analysis`: composición zCDP → conversión (ε,δ) para DP-SynRAG.

Regla práctica: **DP-RAG y DPVote** consumen ε como presupuesto (input) y el accountant verifica/compone; **DP-SynRAG** produce ε como salida del accountant zCDP.

---

## 6. Módulo de Datasets (`bench_datasets/`)

### 6.1 `BaseDataset` (`base_dataset.py`) — ✅
Métodos abstractos (firma real, sin cambios respecto al diseño original):
- `load() -> None`
- `get_documents() -> list[str]`
- `get_queries() -> list[dict]` — cada item con claves `query` y `answer`.
- `get_private_entities() -> list[str]` — vacío para datasets no privados.

### 6.2 Datasets concretos — ✅ + ➕
- `natural_questions.py`, `triviaqa.py`, `hotpotqa.py`, `squad.py`
- `chatdoctor.py` — dominio médico (**prioritario**)
- ➕ `searchqa.py`, `movielens.py`, `medical_synth.py`, `medical_dirichlet_phi3.py` — datasets del paper DP-SynRAG.

### 6.3 Módulo de LLMs (`bench_llms/`) — ✅
- `BaseLLM` expone la interfaz para construir el generador del pipeline.
- `HuggingFaceLLM` recibe `model_name`. Selección por `llm.name` en YAML.

---

## 7. Experimentos (`experiments/`)

### 7.1 Configuración YAML — esquema real

> El esquema plano del diseño original (`epsilon_values`, `metrics:{utility,privacy}`, `output_dir`) **fue reemplazado** por bloques anidados. Usar este esquema al crear configs nuevas (incluidos los YAML stub de lprag/rp_rag/private_rag).

```yaml
# Ejemplo real: experiments/configs/vanilla.yaml
method: vanilla
dataset:
  name: squad                 # squad|chatdoctor|natural_questions|triviaqa|hotpotqa|…
  index_split: train
  test_split: validation
  max_index_docs: 200
  max_test_queries: 25
  docs_path:                  # opcional: jsonl local
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

- Los métodos con barrido de ε (DP-RAG, DPVote) llevan bloques de parámetros propios (p. ej. `epsilon_retrieval`, `epsilon_generation`, `m`, `k`, `epsilon_token`, …).
- **DP-SynRAG** no lleva `epsilon_values`: lleva `keyword_extractor / clustering / cluster_retrieval / synthetic_generation / self_filter / delta` y la ε se imprime derivada (ver §4.6).
- Cada config tiene una variante `_quick` para smoke runs.

### 7.2 Scripts por Método (`experiments/run_*.py`) — ✅
Cada script: carga su YAML, instancia componentes (vanilla + DP donde aplica), construye el pipeline, itera la grilla DP del config, indexa, ejecuta queries, calcula métricas y guarda JSON en `results/{method}/...`. Invocación directa:

```
python experiments/run_dp_rag.py --config experiments/configs/dp_rag.yaml [--verbose] [--verbose-max-examples N]
```

`run_dpsparsevoterag.py` es un script **propio** (no comparte con `run_dpvoterag.py`).

### 7.3 Runner Central (`experiments/runner.py`) — 🚧 PENDIENTE

> Implementar **como sale aquí**. Hoy los experimentos se lanzan invocando cada `run_*.py` directamente; falta el CLI central.

CLI con: `--method` (o `all`), `--dataset` (o `all`), `--llm-name`, `--llm-model`, `--config` (opcional, sobreescribe defaults), `--dry-run`. Internamente importa cada `run_*.py` y ejecuta su `main()` (o lo lanza como subprocess).

---

## 8. Análisis y Visualización (`analysis/`)

- `plot_tradeoff.py` — lee `results/`, genera curva ε vs utilidad y ε vs privacidad (AUC-MIA) por dataset; figuras en `results/figures/`. Para RP-RAG (sin ε) tratar como punto/serie sin eje ε.
- `summary_table.py` — tabla CSV/LaTeX: Método, Dataset, ε, MatchAccuracy, BERTScore F1, ROUGE-L, MIA-AUC, EntityLeakage. Para DP-SynRAG usar el ε **derivado** del accountant.

---

## 9. Tests (`tests/`)

pytest, no cobertura exhaustiva. Estado real (✅ existen; los del diseño original que faltaban se ampliaron):
- `test_core.py`, `test_vanilla.py`, `test_metrics.py`, `test_datasets.py` — ✅
- ➕ `test_dp_methods.py`, `test_dpvoterag.py`, `test_limited_domain_gumbel.py`, `test_llms.py`, `test_runner_smoke.py`, `test_seed_reproducibility.py`

Al implementar LPRAG/RP-RAG/PRIVATE-RAG, añadir sus tests siguiendo el patrón de `test_dp_methods.py`.

---

## 10. TO-DO List de Implementación

Estado real por bloque:

- **BLOQUE 0** Setup — ✅ HECHO
- **BLOQUE 1** Clases abstractas `core/` — ✅ HECHO (criterio: `@abstractmethod` + pipeline con mocks)
- **BLOQUE 2** Vanilla — ✅ HECHO
- **BLOQUE 3** Datasets — ✅ HECHO (+ 4 datasets extra DP-SynRAG)
- **BLOQUE 4** Métricas — ✅ HECHO (+ privacy accounting, §5.3)
- **BLOQUE 5** Métodos DP:
  - 5.1 **LPRAG** — 🚧 PENDIENTE (NER → exp-mech entidades → `EntityPerturbedChunker` → `run_lprag.py` + `lprag.yaml` → piloto ε=5, NQ, 20 queries)
  - 5.2 **DP-RAG** — ✅ HECHO
  - 5.3 **DPVoteRAG / DPSparseVoteRAG** — ✅ HECHO
  - 5.4 **RP-RAG** — 🚧 PENDIENTE (matriz proyección → `RandomProjectionIndexer` → `RPRetriever` → run + yaml; documentar "sin ε formal")
  - 5.5 **PRIVATE-RAG** — 🚧 PENDIENTE (`BudgetManager` Rényi → `BudgetAwareRetriever` (MURAG) → `AdaptiveBudgetAwareRetriever` (MURAG-ADA) → run + yaml)
  - 5.6 **DP-SynRAG** — ✅ HECHO
- **BLOQUE 6** Runner + tests:
  - 6.1 `runner.py` — 🚧 PENDIENTE
  - 6.3–6.5 tests core/vanilla — ✅ HECHO
- **BLOQUE 7** Experimentos completos — pendiente (depende de completar Bloque 5)
- **BLOQUE 8** Análisis y figuras — `plot_tradeoff.py` / `summary_table.py` presentes; validación manual pendiente

Próximos pasos sugeridos: 5.1 LPRAG → 5.4 RP-RAG → 5.5 PRIVATE-RAG → 6.1 runner → Bloque 7 → Bloque 8.

---

## 11. Notas Importantes para el Agente

1. **Orden estricto de bloques**: no iniciar Bloque 5 (métodos pendientes) sin Bloque 2 funcional. Bloque 4 puede ir en paralelo a Bloque 3. (Ambos ✅.)

2. **Inyección de dependencias — con excepción explícita**: el `RAGPipeline` nunca importa directamente desde `vanilla/` ni `dp_methods/`; siempre recibe instancias desde afuera. **Excepción permitida**: cuando un método cambia el *flujo* de `query()` (no solo un componente), se permite **subclasear** `RAGPipeline` dentro del módulo del método. Caso de referencia: `dp_methods/dpvoterag/pipeline.py::DPVoteRAGPipeline`, que recalcula el `k` de recuperación como `m·k` y delega en `generator.generate_vote(...)` porque el voting no encaja en la firma `retrieve→generate` estándar. La subclase sigue recibiendo sus componentes inyectados; lo único que cambia es la orquestación. Mantener este patrón (subclase de `RAGPipeline`, no un pipeline ad-hoc paralelo) para futuros métodos que lo requieran.

3. **RP-RAG no es DP formal**: al implementarlo (🚧), dejar explícito que no tiene parámetro ε; en la tabla comparativa sus columnas de ε van "N/A".

4. **DPSparseVoteRAG — limitación formal conocida**: el ahorro de budget cuando el token coincide con el LLM público sigue la convención del repo oficial (`epsilon_token` y `sparse_eps` independientes). Documentado en su README; para reproducir el paper estrictamente fijar `sparse_eps = epsilon_token/2`.

5. **Reproducibilidad**: todos los experimentos fijan `random_seed`, propagado a PyTorch/NumPy/Python. Además se usa RNG por-query derivado de `hash(query)` para independencia entre consultas (ver tests `test_seed_reproducibility.py`). Mantener esta convención en métodos nuevos.

6. **Escalabilidad**: DPVoteRAG requiere K×M inferencias por consulta (piloto: K=5, M=2 máx; reportar tiempo/consulta). DP-SynRAG es el de mayor coste offline (un forward por doc en extracción + `O(R·max|S_r|·T)` en generación) — usar `max_samples` y la caché FAISS (`experiments/index_cache.py`, keyed en hiperparams DP) para evitar rebuilds.

7. **ε como entrada vs salida**: DP-RAG y DPVote consumen ε como presupuesto (el accountant compone/verifica); **DP-SynRAG deriva ε** del accountant zCDP a partir de los parámetros de ruido. El análisis (§8) debe leer el ε correcto según el método.
