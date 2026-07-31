# Auditoría del estado actual — inventario por experimento

Levantada leyendo el código (no los README, varios de los cuales están desactualizados).
Base para la homogeneización de `PLAN_EVALUACION_HOMOGENEA.md`.

---

## 1. Matriz método × dataset

### Config `full/`

| Método | Datasets | `max_index_docs` | Q (queries) |
|---|---|---|---|
| dp_rag | nq, tqa, chatdoctor, searchqa, medical, movielens | 20 000 (medical `null` ≈10k) | 40 |
| dpvoterag | los 6 | 20 000 | 40 |
| dpsparsevoterag | los 6 | 20 000 | 40 |
| dpksa | los 6 | 20 000 | 40 |
| private_rag | los 6 | 20 000 | 40 |
| private_rag_ada | los 6 | 20 000 | 40 |
| dp_synrag | los 6 | 1600 / 816 (movielens) / 1410 (searchqa) | 40 |
| vanilla | **solo squad** | 200 | 25 |

### Config `paper_faithful/`

| Método | Datasets | Corpus | Q |
|---|---|---|---|
| dp_rag | medical | `null` (~10k) | 100 |
| dpvoterag | nq, tqa, chatdoctor | 20 000 | 40 |
| dpsparsevoterag | nq, tqa (**21M**), chatdoctor (20k) | 21M / 20k | 100 / 40 |
| dpksa | nq, tqa | **21M** | 40 |
| private_rag(+ada) | nq, tqa, chatdoctor | **21M** / 20k | 40 |
| dp_synrag | medical, movielens, searchqa | 8000 / 4083 / 7054 | 200 / 200 / 102 |

**Corpus incomparables entre métodos en `paper_faithful`**: 21M vs 20 000 vs 8 000 vs ~10k.
`squad` y `hotpotqa` no aparecen en ninguna config DP.

---

## 2. Filtros por dataset (dónde están y quién los activa)

| Filtro | Implementado en | Lo activa | Efecto |
|---|---|---|---|
| **Relevance-binning de Koga** (queries con `len(positive_ctxs)` DPR ∈ [20,60), 4 bins × 25) | `bench_datasets/dpr_relevance.py:1-20,94-108`; `natural_questions.py:118-133,157-216`; `triviaqa.py:118-126,155-209` | **solo** dpvoterag y dpsparsevoterag, y **solo** en `paper_faithful` (`run_dpvoterag.py:151-163`) | Es el "filtra las queries bien soportadas". Construye un set de queries **disjunto** del resto de métodos |
| **`top_movies` de MovieLens** (conserva las N películas más valoradas, poda perfiles y golds) | `bench_datasets/movielens.py:94-121` | **solo** dp_synrag, y **solo** en `paper_faithful` (`top_movies: 30`) | Es el "synrag filtra las películas". En `full/` NO se aplica |
| **Support-binning de SearchQA** (≥40 snippets, 6 bins × 17) | `searchqa.py:112-115,165-225` | solo dp_synrag `paper_faithful` | |
| **Self-filter YES/NO** de DP-SynRAG | `dp_methods/offline/dp_synrag/self_filter.py:64-115` | dp_synrag: ON para medical, chatdoctor, movielens; OFF para searchqa/nq/tqa. **El prompt de movielens difiere entre `full` y `paper_faithful`** | |
| `min_high_ratings=5`, `top_k_favorites=10`, `seed_size=3` | `movielens.py:60,114,127-128,154-157` | siempre (movielens) | |
| Medical: solo `doctor_response` indexado; descarta filas sin síntoma/enfermedad; query = plantilla de síntomas | `medical_dirichlet_phi3.py:67-77,90-107` | siempre | |
| SearchQA: snippets con respuesta primero al truncar | `searchqa.py:240-271` | siempre | |
| SQuAD: siembra el índice con el contexto gold de cada query de test | `squad.py:93-120` | siempre (solo vanilla) | leakage |
| Reservoir sampling sembrado del corpus DPR | `dpr_corpus.py:38-73` | siempre que `max_index_docs` < N | |
| **Grounding filter del paper DP-SynRAG** | — | **no existe** (`grep grounding` → 0 hits) | |
| Filtro DP-KSA de stopwords | `keyword_selector.py:36-55` | `false` en ambas configs | |

**Bloqueante local:** `data/dpr/` no existe → NQ/TriviaQA caen a un fallback degenerado
(`natural_questions.py:100-117`, `triviaqa.py:89-112`) **y el relevance-binning se salta con un warning**
sin registrarlo en el JSON.

**Bug de propagación:** `run_private_rag.py:114-119` y `run_dpksa.py:98` hardcodean los kwargs del loader,
así que cualquier filtro puesto en su YAML se ignora en silencio.

---

## 3. Grillas de ε y semántica

| Método | Qué se barre | δ | Semántica del ε | Accountant |
|---|---|---|---|---|
| dp_rag | `eps_r ∈ {0.5}` × `eps_g ∈ {2, 5, 7.5, 10}` | 1e-3 | **por consulta** (retrieval ⊕ T tokens) | PLD exacta, `privacy_accounting.py:16-65` |
| dpvoterag | `epsilon_total ∈ {2,4,8,10,20,40}`, `epsilon_token=2` | 1e-4 / token 1e-6 | **por consulta** (`accountant.reset()`, `generator.py:397`) | RDP EM-peeling + BBGHS |
| dpsparsevoterag | `{2,5,10,20,40} × epsilon_token {1,2,5}` (pf quita ε=2 → 12 celdas) | 1e-4 / token 1e-5 | **por consulta** (`sparse_vote_generator.py:192`) | RDP + SVT por época |
| dpksa | `{1, 2, 3, 5, 8}` | 1e-4 | **por consulta** | RDP aditiva EM + PTR |
| private_rag / _ada | `eps_q ∈ {2,5,10,15,20,30,40}` × `eps_0 ∈ {0.5,1,2}`, **M=1 fijo** | **0** (DP puro) | **total sobre el corpus**, ε = M·ε_q | cerrado, `privacy_accounting_private_rag.py:30-36` |
| dp_synrag | `c ∈ {1, 0.5, 0.25, 0.1, 0.05}` → ε ≈ 245/79/29/10/5 (pf: `c=0.1057` → ε≈10) | 1e-3 | **total offline**, constante en Q (verificado: `run_dp_synrag.py:430` fuera del bucle) | zCDP Bun-Steinke |
| vanilla | — | — | — | — |

Tres δ distintos (1e-3 / 1e-4 / 0) y ninguno es el 1e-5 que compromete la tesis.

---

## 4. LLMs

- `full/`: los 7 configs DP usan **los mismos 7 modelos** (opt-1.3b, pythia-1.4b, Llama-3.2-1B/3B, Qwen2.5-3B, Gemma-2-2B, Phi-4-mini). vanilla usa `tiny-gpt2`.
- `paper_faithful/`: divergen — Phi-3.5 (dp_rag), Phi-4 (dp_synrag), Qwen+Llama (dpksa), opt+pythia+Llama-3.1-8B (voterag/sparse), opt+pythia+Mistral-7B (private_rag).
- Embedder **homogéneo** en las 15 configs: `Snowflake/snowflake-arctic-embed-m-v1.5`.
- Mecánica de generación incomparable: `generate()` greedy (vanilla, dpksa ×80 por consulta), forward token a token (dp_rag, dpvoterag, dpsparse, private_rag), 4 etapas con un solo LLM (dp_synrag).
- `max_new_tokens`: 70 (dp_rag) / 50 pasos de voto (voterag, sparse, private_rag) / 16-32 (dpksa) / 64 lector (dp_synrag).

---

## 5. Chunking — rompe la unidad de privacidad

| Método | Chunker |
|---|---|
| dp_rag | `NoChunker` (`run_dp_rag.py:87`) |
| dp_synrag | `NoChunker` (`run_dp_synrag.py:457`) |
| dpvoterag | `FixedSizeChunker(200,20)` **hardcodeado** (`:331`) |
| dpsparsevoterag | ídem (`:339`) |
| dpksa | ídem (`:256`) |
| private_rag | ídem (`:310`) |
| vanilla | `FixedSizeChunker(120,20)` |

4 de 8 reportan ε_chunk como si fuera ε_doc. Con c chunks por documento, la garantía real es ε_doc = c·ε_chunk.

---

## 6. Métricas

- `experiments/score_results.py:36-59` computa **siempre** el mismo set canónico (8 utility + 4 MIA) e **ignora el bloque `metrics:` de los YAML** (el orquestador fuerza `BENCHDP_DEFER_METRICS=1`, `runner.py:232`).
- Dos normalizaciones conviven: SQuAD en `match_accuracy`/`f1_score`; laxa en `contains_accuracy`/`hit_rate`/`levenshtein` (`utility.py:14-25,32-44`).
- `bert_score_f1` es la única que **ignora los alias** (`utility.py:587`).
- **`hit_rate` de MovieLens solo funciona en dp_synrag** — es el único runner que persiste `favorites`. En los otros 6, MovieLens se puntúa por `match_accuracy` contra un título ⇒ ≈0 no comparable.
- **Faltan** `answer_rate`/`empty_rate` y `retrieval_precision@50`. `recall_at_k` existe (`utility.py:371-400`) pero nunca se invoca y **ningún runner persiste `retrieved_ids`**.
- **Bug crítico en vanilla**: escribe `"answer"` (`run_vanilla.py:465`), el scorer lee `"prediction"` (`score_results.py:72-74`) ⇒ el baseline no privado se puntúa con strings vacíos y sale **0.0** en `metrics_summary.csv`.
- `metrics_summary.csv` no agrega (una fila por fichero, sin n ni CI) y colapsa `epsilon_total`/`epsilon`/`zcdp.epsilon` en una sola columna, mezclando semánticas.
- `analysis/load_results.py:248` está caducado: busca `privacy_accounting["advanced"|"rdp"|"basic"]` pero dp_rag escribe `{"pld","naive_sum"}` ⇒ reporta el naive sum en vez del ε PLD.
- `autodp` está en `requirements.txt:22` pero **no está instalado ni se importa**; `privacy_accounting_autodp.py` es implementación a mano. `dp_accounting-0.6.0` sí está instalado.

---

## 7. Orquestación y semillas

- `runner.py` recorre **dataset-major**, un subproceso por (método, dataset) y otro por LLM. No ve LLMs ni ε: viven en cada YAML.
- CLI: solo `--paper-faithful`, `--methods`, `--mia-only`. **No hay `--dataset`, `--llm`, `--seed` ni `--repeats`**.
- **No existe noción de repeticiones**: la semilla es escalar y se usa a la vez para muestrear queries y para el ruido DP ⇒ repetir un run da salida idéntica. Hace falta separar `data_seed` (entra en la clave del cache de índice, `index_cache.py:113`) de `run_seed` (ruido DP).
- `common.py` tiene 44 líneas; el contrato real está duplicado en los 7 runners, con **dos firmas distintas** de carga de dataset (`_resolve_dataset` vs `load_dataset`) y claves divergentes (`max_samples` vs `max_index_docs` + `max_test_queries` + `max_queries`).
- `random_seed` es top-level en dp_rag/dp_synrag/vanilla y anidado en el bloque del método en los otros 4.
- Cache de índice: hashea chunker + `store_embeddings_matrix`/`build_faiss_index` ⇒ **hoy ningún par de métodos comparte índice**.
- Selección de queries: en `full/` los 7 DP comparten set (Q=40, seed 42); en `paper_faithful/` los sets son **disjuntos** entre voterag/sparse (binned) y el resto. No hay query-id ni test que verifique la igualdad.

---

## 8. Viabilidad del presupuesto multiconsulta ε_total(Q)

| Método | ¿Acepta ε_total repartido entre Q? |
|---|---|
| dp_rag | Sí sin correr nada extra (sin estado entre consultas). Requiere invertir `pld_dp_rag_total` por bisección (patrón ya existe en `generator.py:187-221`) y decidir el reparto entre `eps_r` y `eps_g` |
| dpvoterag | Requiere código: el accountant se **resetea por consulta** (`generator.py:397`). Las primitivas de composición existen; falta el helper de inversión |
| dpsparsevoterag | Requiere código **y cuidado**: `epsilon_token` es independiente del tope, así que con ε_query pequeño el primer cobro de SVT deja `t_max()=0` → predicción vacía (es lo que ya pasa a ε=2). Hay que escalar ε_token junto con el tope |
| dpksa | Trivial: la calibración es función pura, sin estado entre consultas. Riesgo: con Q grande casi todo el barrido cae en ε_query≤1, donde el PTR nunca pasa y el método colapsa a zero-shot ⇒ hay que reportar `ptr_pass_rate` |
| private_rag / _ada | Ya vive en ese eje (ε = M·ε_q, constante en Q). Falta exponer `M_values` y emitir la utilidad acumulada por prefijo del stream |
| dp_synrag | Ya vive en ese eje. La inversión ε_total → c es **closed-form** (verificado: ε=10 → c=0.10573, que es el 0.1057 del config); falta el código (`solve_c_for_epsilon` no existe) |

Como ninguno de los 4 per-query tiene estado entre consultas, **una sola grilla de ε_query se puede reetiquetar
a cualquier par (ε_total, Q) sin corridas adicionales**.
