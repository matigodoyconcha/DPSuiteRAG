# Plan de evaluación homogénea: DP-Benchmark-RAG

**Fecha:** 2026-07-24
**Objetivo:** definir y ejecutar un protocolo que permita comparar los 7 algoritmos DP del benchmark bajo condiciones parejas, resolviendo específicamente el problema de que **Private-RAG (MuRAG) y DP-SynRAG no viven en el mismo eje de privacidad que los métodos per-query**.

---

## Parte I. Diagnóstico

### 1. El problema real no es de configuración, es de semántica de ε

Los 7 algoritmos declaran un número llamado "ε", pero ese número **mide cosas distintas**:

| Familia | Algoritmos | Qué mide su ε | ¿Depende de cuántas consultas se hagan? |
|---|---|---|---|
| Per-query | DP-RAG, DPVoteRAG, DPSparseVoteRAG, DP-KSA | costo de **una** consulta | No lo reporta. El costo real crece con Q |
| Multi-query con filtro individual | MuRAG, MuRAG-Ada | costo **total sobre el corpus**, para todo el stream | No. Es `M·ε_q`, constante en Q |
| Offline amortizado | DP-SynRAG | costo **total one-shot** de construir el corpus sintético | No. Después es post-procesamiento gratis |

Consecuencia directa y verificable en el repo: `experiments/configs/full/private_rag.yaml` corre con `M: 1`, es decir su garantía titular `ε = M·ε_q` cubre **las 40 consultas completas**. Cuando el plot actual pone MuRAG a ε=10 al lado de DP-RAG a ε=10, le está pidiendo a MuRAG responder 40 consultas con el presupuesto que DP-RAG gasta en **una sola**. El "MURAG-ADA 100% vacías" y el "plano en ε_q" de `results/METHODS_OVERVIEW.md` no son hallazgos sobre el método: son el artefacto de un eje mal definido.

### 2. Cómo resuelven esto los propios papers

Fui a leer los seis papers. **Los dos métodos "raros" ya resolvieron el problema, y con la misma solución.**

#### MuRAG (§5.2 del paper, define las dos fórmulas del eje x)

> "For DP-Multi-RAG, the **total privacy budget is T · ε_q**, where T is the number of questions. For MuRAG and MuRAG-Ada, the **total budget is M · ε_q**, where M is the number of retrieved documents with nonzero privacy loss. In our main results, we conservatively set **M = 1**."

Su eje x es **`ε_total` en escala logarítmica**, y su famoso "100× mejor" es literalmente `T/M = 100/1`. El ε=1000 del abstract es exactamente `100 consultas × ε_q=10`. Comparan **iso-utilidad**: fijan un nivel de utilidad por encima del baseline Non-RAG y leen cuánto presupuesto necesita cada método para alcanzarlo.

#### DP-SynRAG (Figura 3, la comparación honesta)

Su Tabla 1 headline **no** es comparable: pone `ε_total=10` de DP-SynRAG contra `ε_query=10` de DP-RAG, y en el propio caption anota que el total compuesto de DP-RAG es `≈10000 / 5000 / 1000`. Le regalan a DP-RAG entre 100× y 1000× más presupuesto.

Pero su **Figura 3 sí es el eje correcto**, y es el que hay que replicar:

- Eje x: **número de consultas m ∈ {1, 5, 10, 15, 20}**
- Eje y: accuracy
- Lo que se mantiene fijo: **ε_total ∈ {3, 5, 10, 20}**
- DP-SynRAG: **línea horizontal** (el corpus se reutiliza sin costo)
- DP-RAG: recibe `ε_query = ε_total/m` y **decae**
- Cruce declarado: **m=1** (empate). DP-SynRAG domina para todo m>1

#### Los cuatro per-query: ninguno compone sobre el workload

Verificado paper por paper:

- **DP-RAG (Grislain):** un solo punto operativo (ε≈5, δ=1e-3), sin baselines, sin semillas, sin número de consultas reportado. Cero menciones a composición entre consultas.
- **Koga (DPVoteRAG / DPSparseVoteRAG):** ε_total=2..40 **por pregunta**, δ_total=1e-4. Discute composición extensamente, pero **solo sobre tokens dentro de una consulta**. Cero menciones al workload.
- **DP-KSA:** ε ∈ {1,2,3,5,8} **por consulta**, δ=1e-4. Critica explícitamente la composición por token de los otros dos, pero **no contabiliza la suya entre consultas**.

Con Q=100 consultas de test y composición básica, sus presupuestos reales sobre el corpus serían **~500 (Grislain), ~1000-4000 (Koga), ~100-800 (DP-KSA)**.

### 3. La conclusión que ordena todo el plan

> **Existe un solo eje en el que los tres regímenes son comparables: el presupuesto total sobre el corpus para un workload de Q consultas, `ε_total(Q)`, bajo una relación de vecindad a nivel documento y un accountant común.**

En ese eje:
- Per-query: `ε_total(Q)` **crece con Q**
- MuRAG: `ε_total = M·ε_q`, **constante en Q** (paga con documentos que se apagan)
- DP-SynRAG: `ε_total = ε_build`, **constante en Q** (paga con fidelidad del corpus)

Y la historia del benchmark deja de ser "quién gana a ε=10" y pasa a ser **"cómo escala el costo de privacidad con el tamaño del workload, y dónde están los puntos de cruce"**. Eso es una contribución, no un rescate.

### 4. La asimetría que ambos papers cometen (y que podemos corregir)

MuRAG y DP-SynRAG usan **composición básica** (`ε_total = Q·ε_query`) para el baseline per-query, mientras usan contabilidad ajustada (RDP / zCDP) para sí mismos. Eso infla artificialmente el costo de los per-query por un factor grande.

Cuantificación concreta, con δ=1e-5 y `ln(1/δ)=11.513`:

Partiendo de `ε_total = 10`, la inversa zCDP da `ρ_total = (√(ε+ln(1/δ)) − √(ln(1/δ)))² = 1.550`.

| Q | ε_query bajo composición básica | ε_query bajo zCDP (`ρ_q = ρ_total/Q`) | Ganancia |
|---|---|---|---|
| 1 | 10.0 | 10.0 | 1.0× |
| 10 | 1.0 | 2.86 | 2.9× |
| 100 | 0.1 | 0.86 | **8.6×** |

El repo ya tiene `metrics/privacy_accounting_autodp.py` y el accountant PLD (que en el run de DP-RAG resultó "~2× más ajustado que la composición naive"). **Corregir esta asimetría es defendible, barato y es exactamente lo que `metodologia.tex:58-61` ya prometió por escrito.**

### 5. Un problema de corrección, no de homogeneidad: el chunking rompe la garantía a nivel documento

`metodologia.tex:44` compromete la vecindad como *"two versions of the corpus that differ in exactly one document"*.

Pero en el código:
- `run_dp_rag.py` y `run_dp_synrag.py` usan **NoChunker** (1 documento = 1 unidad de privacidad). Correcto.
- `run_dpvoterag.py:331`, `run_dpsparsevoterag.py:339`, `run_dpksa.py:256`, `run_private_rag.py:310` **hardcodean chunking 200/20**.

Si un documento se parte en c chunks y el mecanismo garantiza DP respecto a **un chunk**, entonces la garantía a nivel **documento** requiere privacidad de grupo: `ε_doc = c · ε_chunk`. Los cuatro runners reportan hoy `ε_chunk` como si fuera `ε_doc`.

Esto **no es un detalle de homogeneización, es un error en la garantía reportada**, y afecta a 4 de los 7 algoritmos. Hay que resolverlo antes de correr nada.

### 6. Inventario del estado actual

| Dimensión | Estado |
|---|---|
| Embedder | ✅ **único knob realmente homogéneo** hoy (`arctic-embed-m-v1.5` en los 8 configs) |
| Scoring | ✅ centralizado en `score_results.py` con normalización SQuAD |
| δ | ❌ 1e-3 / 1e-4 / 0 según método. La tesis compromete **1e-5**. Ninguno coincide |
| LLM generador | ❌ tiny-gpt2 / opt-1.3b / Qwen-3B / Phi-3.5 / Phi-4-mini según método |
| Tamaño de corpus | ❌ 200 / 1.600 / 20.000 / 21M según método |
| #consultas de test | ❌ 40 / 100 / 200 según método y config |
| Chunking | ❌ NoChunker en 2, hardcodeado 200/20 en 4, 120/20 en vanilla |
| `random_seed` | ❌ top-level en 3 runners, dentro del bloque del método en 4 |
| `max_new_tokens` | ❌ 32 a 70 según método |
| Semillas | ❌ una sola (42) en todos |
| **Run con todos los métodos** | ❌ **no existe**. El mejor (`run_20260702_114726`) tiene 6 de 8 |
| **`dpvoterag`** | ❌ **nunca se ha corrido**. Cero JSON en todo `results/` |
| **`vanilla`** | ❌ nunca se ha corrido, y **no está en `DEFAULT_METHODS`** (`runner.py:87-95`) |
| Run de `private_rag` | ❌ **declarado inválido** en `results/reuMartes/LEEME.md` (τ=0.3 agotó el corpus). Invalida 104 de 182 JSON |

---

## Parte II. El protocolo unificado

### 7. Definición del eje

**Unidad de privacidad:** un documento del corpus, relación add/remove. Aplica a los 7 algoritmos. Los seis papers coinciden en esto (verificado uno por uno), así que no hay conflicto de fondo.

**δ común:** `1e-5`, como compromete `metodologia.tex:53-55`.

**Accountant común:** RDP/zCDP como marco declarado (coincide con el texto del manuscrito), componiendo **al nivel de los mecanismos base** (exponencial, Gumbel/LimitedDomain, Gaussiano, Laplace, SVT) sobre el workload completo de `Q × T` releases, no encadenando cajas negras `(ε,δ)` por consulta. PLD se reporta como columna secundaria más ajustada donde `autodp` lo permita.

**Excepción importante y deliberada:** MuRAG es **ε-DP puro (δ=0)**. Convertir pure DP a zCDP y de vuelta a δ=1e-5 **empeora** el número (para ε₀=10 daría 36.5). Como `(ε₀, 0)-DP ⟹ (ε₀, δ)-DP` para cualquier δ, **MuRAG se reporta con su ε₀ directo**, que es válido a δ=1e-5 y es conservador con él. Se anota en el texto que existe una versión más ajustada vía el filtro de Feldman-Zrnic a orden Rényi finito, pero eso requiere una demostración nueva en el Anexo A y queda como refinamiento opcional.

**Definición operativa:**

```
ε_total(Q) := pérdida de privacidad en el peor caso respecto de un documento
              del corpus, acumulada sobre un workload de Q consultas,
              a δ = 1e-5, bajo el accountant común.

  per-query:   ε_total(Q) = compose_RDP(mecanismos_de_una_consulta, Q)
  MuRAG/Ada:   ε_total(Q) = M · ε_q                    (constante en Q, δ=0)
  DP-SynRAG:   ε_total(Q) = ε_build                    (constante en Q)
```

### 8. Las tres salidas que hay que producir

**Figura A — trade-off clásico, sobre el eje corregido.**
Utilidad vs `ε_total`, a **Q fijo = 100**. Una curva por método. Esto es lo que `metodologia.tex:215-223` ya promete, pero con el eje x arreglado. Aquí MuRAG y DP-SynRAG dejan de estar penalizados 100×.

**Figura B — escalamiento con el workload (el estilo Figura 3 de DP-SynRAG). Es la figura nueva y la más importante.**
Utilidad vs **Q ∈ {1, 5, 10, 25, 50, 100}**, a `ε_total` fijo (una subfigura por valor de `ε_total ∈ {1, 3, 10, 30}`).
- DP-SynRAG: línea horizontal
- MuRAG/Ada: casi horizontal, con degradación por documentos apagados
- Los cuatro per-query: decaen

Aquí se leen los **puntos de cruce**, que es el resultado que ningún paper ha producido para más de dos métodos a la vez.

**Tabla C — iso-utilidad (el criterio de MuRAG).**
Para cada método y dataset: `ε_total` mínimo necesario para superar el baseline Non-RAG en un margen fijo, a Q=100. Es la lectura que hace MuRAG y la que produce titulares interpretables ("el método X necesita 40× más presupuesto que el método Y para el mismo nivel de utilidad").

### 9. Truco de cómputo que hace viable la Figura B

Las respuestas de un método per-query dependen **solo de `ε_query`**, no de Q. Y las de MuRAG/DP-SynRAG dependen solo de su `ε_total`.

Por lo tanto:

- **Per-query:** se barre una grilla de `ε_query` una sola vez, y cada punto `(ε_total, Q)` de la Figura B se obtiene **reetiquetando / interpolando** sobre esa curva. No hay que correr un experimento por cada par `(ε_total, Q)`.
- **MuRAG:** una sola corrida por `ε_total` con el stream completo de Q=100. La curva de la Figura B es la **media acumulada sobre prefijos del stream**. Sale gratis del mismo run.
- **DP-SynRAG:** un solo build por `ε_total`, luego se responden las 100 consultas. La curva es horizontal por construcción.

Esto reduce la Figura B de "|ε_total| × |Q| corridas por método" a "cero corridas adicionales". Es el motivo por el que este protocolo es ejecutable.

### 10. Qué se fija y qué varía

| Factor | Decisión | Justificación |
|---|---|---|
| Embedder | `Snowflake/snowflake-arctic-embed-m-v1.5` | ya es homogéneo. El Anexo B permite intercambiar retriever en los 4 per-query (κ≤2 absorbido) |
| LLM generador (grid principal) | **`Qwen/Qwen2.5-3B-Instruct`** | cabe en 24 GiB con m=30 votantes, instruct, ya es el default de dpksa. **Decisión abierta**, ver §16 |
| LLM (robustez) | `facebook/opt-1.3b` + `meta-llama/Llama-3.1-8B` | opt-1.3b es nativo de Koga y MuRAG. Llama-3.1-8B es el **único LLM compartido por dos papers** (Koga y DP-KSA) |
| Chunking | **NoChunker en los 7** (1 documento = 1 unidad de privacidad) | única forma de que el ε reportado sea el ε a nivel documento que compromete la tesis. Ver §5 |
| Corpus | mismo N y mismos documentos para los 7 | DP-SynRAG **indexa un corpus sintético**: comparte el corpus de **entrada**, no el índice. Es intrínseco (locus S1) |
| Consultas de test | **mismo set de Q=100**, mismo orden, para los 7 | Q=100 coincide con Koga y con MuRAG |
| δ | `1e-5` en todos (δ=0 en MuRAG, que es más fuerte) | `metodologia.tex:53-55` |
| Semillas | **3** (5 si el cómputo alcanza) | Koga usa 3, DP-SynRAG usa 5. Grislain y DP-KSA no reportan ninguna |
| `max_new_tokens` | **32** común | el costo de composición por token escala con T. Dejarlo libre sesga el eje ε |
| `k` (docs recuperados) | **NO se iguala.** Se reporta el k efectivo de cada uno | es intrínseco al mecanismo: DP-RAG usa umbral DP (k aleatorio), Koga necesita m votantes, DP-KSA usa N=80 ensembles. Forzarlo rompería los métodos |
| `m` (votantes) | **30** en el grid principal | está en la grilla de Koga {30,40,50} y de MuRAG. Es el que domina el costo de cómputo. Barrer m queda para Tier C |
| Filtrado de consultas | **ninguno** (se mantiene la política actual del README) | Koga descarta las preguntas con <20 docs relevantes "porque los DP fallarían fácilmente". Ese filtro esconde justamente lo que el benchmark quiere medir. Se reporta **estratificado por bins de redundancia** en vez de filtrar |
| Selección de hiperparámetros | **fija a priori**, no best-of-grid | Koga y MuRAG reportan el máximo sobre hiperparámetros **medido en el test set**. Es un sesgo optimista que no queremos heredar |

### 11. Datasets

`metodologia.tex:163-170` prioriza ChatDoctor y MovieLens-1M como escenarios con sensibilidad real, y permite NQ/TriviaQA solo como *reference condition*. El análisis de generalización prometido (`metodologia.tex:227-229`) exige **al menos dos dominios**.

| Dataset | Rol | Dominio | Por qué |
|---|---|---|---|
| **ChatDoctor** | principal | médico | prioritario en la tesis. Usado por Koga y MuRAG. Set "dependiente" (alta redundancia) |
| **medical_dirichlet_phi3** (Medical Synth) | principal | médico sintético | **único dataset compartido por DP-RAG y DP-SynRAG**. Redundancia controlada por construcción (bins de frecuencia de enfermedad) |
| **MovieLens-1M** | principal | recomendación | prioritario en la tesis. Usado por DP-SynRAG. Único con gold set-valued (`hit_rate`) y unidad = usuario |
| **Natural Questions** | reference condition | conocimiento general | permite el análisis de generalización. Set "independiente" (baja redundancia). Solo con el LLM primario |

### 12. Métricas

**Utilidad (existentes):** `contains_accuracy` (headline, es la "match accuracy" de todos los papers), `f1_score` con normalización SQuAD, `rouge_l`, `hit_rate` (MovieLens), `bert_score_f1` (ChatDoctor, es la métrica que usa MuRAG ahí).

**Utilidad (a implementar):**
- **`answer_rate` / `empty_rate`.** Cuatro de los siete algoritmos degeneran a salida vacía en algún régimen (Sparse-Vote, MURAG-ADA, DP-KSA cuando PTR no libera, DP-SynRAG cuando el self-filter vacía el corpus). Hoy esas vacías se promedian como F1=0 y se confunden con "respondió mal". Hay que separar **"no respondió"** de **"respondió mal"**.
- **`retrieval_precision@50`.** Es la Tabla 1 de MuRAG: % de documentos verdaderamente top-50 entre los recuperados. Es **la única métrica que hace visible el agotamiento de presupuesto** de MuRAG (cae a 13.3% en ChatDoctor con M=1, contra 78.8% en NQ). Barata: solo requiere el ranking no privado como gold.

**Privacidad (existentes):** `s2mia_auc`, `rag_mia_auc` (yes/no), `synth_retention_auc` (solo DP-SynRAG), consumo de presupuesto. Todas **standalone**, como compromete `metodologia.tex:100`.

**Caracterización del corpus (a implementar, offline, no requiere correr ningún método):**
- **Histograma de reutilización de documentos** (Figura 1 de MuRAG): en cuántas consultas aparece cada documento en el top-50. Clasifica cada dataset como **independiente** (NQ, TriviaQA) o **dependiente** (ChatDoctor, MovieLens). Es un script de 30 líneas sobre el índice, y es **la evidencia directa del hilo de redundancia/S3 del paper WSDM**. Predice cuáles métodos van a sufrir antes de correrlos.

---

## Parte III. Plan accionable

### Fase 0 — Reparaciones bloqueantes

Nada de esto es opcional: sin esto el run homogéneo no produce resultados citables.

| # | Tarea | Archivo | Detalle |
|---|---|---|---|
| 0.1 | Arreglar `run_vanilla.py` | `experiments/run_vanilla.py` | escribe `"answer"` y `score_results.py:73` lee `"prediction"` → **hoy puntúa vanilla con predicciones vacías**. Además: soportar listas `datasets:`/`llms:`, escribir `dataset` como string y no dict, honrar `BENCHDP_DEFER_METRICS` |
| 0.2 | Meter `vanilla` en `DEFAULT_METHODS` | `experiments/runner.py:87-95` | hoy el sweep por defecto **no produce baseline no privado**, que es la condición de referencia que promete `metodologia.tex:210-213` |
| 0.3 | **Política única de chunking** | los 5 runners que lo hardcodean | exponer `chunk_size`/`chunk_overlap` y ponerlos en NoChunker. Ver §5: hoy 4 métodos reportan ε de chunk como si fuera ε de documento |
| 0.4 | δ=1e-5 en los 8 configs | `experiments/configs/**` | unificar `delta` vs `delta_total`+`delta_token`. MuRAG queda en δ=0 documentado |
| 0.5 | `random_seed` al top-level + lista de semillas | 4 runners | hoy está enterrado en `dpvoterag.random_seed`, `dpksa.random_seed`, etc. |
| 0.6 | Unificar `max_samples` → `max_index_docs` | `run_dp_rag.py:149-150`, `run_dp_synrag.py:154-155` | y **desacoplar corpus de #consultas**: hoy si falta `max_test_queries`, el número de consultas cae al tamaño del corpus |
| 0.7 | Limpiar `results/` | `results/` | `load_results._method_from_path` solo reconoce `run_YYYYmmdd_HHMMSS`. Hoy `run_20260712_224716_maquinatoro`, `reuMartes/`, `dpksa_comparison/` y 2 más se interpretan como **nombres de método** |

### Fase 1 — El eje común (el corazón del plan)

| # | Tarea | Dónde | Detalle |
|---|---|---|---|
| 1.1 | Módulo `metrics/workload_accounting.py` | nuevo | función única `epsilon_total(method, params, Q, delta) -> float`, componiendo mecanismos base con `autodp` para los 7. Es la implementación de la promesa de `metodologia.tex:58-61` |
| 1.2 | **Inversión del accountant de DP-SynRAG** | `metrics/privacy_accounting_zcdp.py` | hoy ε es un **output** y el sweep se hace sobre `c_values`. Hay que invertirlo: dado `(ε_total, δ)` → `ρ = (√(ε+ln(1/δ)) − √(ln(1/δ)))²` → repartir a las etapas → `c/τ = √(2ρ_pred/T)`. Es **closed-form, no requiere búsqueda binaria**. Sin esto DP-SynRAG no puede ponerse en el eje x |
| 1.3 | Calibración inversa per-query | `metrics/workload_accounting.py` | dado `(ε_total, Q, δ)` → `ε_query`. Define la grilla de barrido de los cuatro per-query |
| 1.4 | Campos `epsilon_workload` y `Q` en todos los JSON | los 7 runners | y extender `analysis/load_results.py`. Hoy `dpksa` y `private_rag` caen al normalizador genérico, así que **las celdas de `ε_0` de private_rag colapsan al mismo `epsilon_value`** |
| 1.5 | Curva prefijo de MuRAG | `run_private_rag.py` | emitir la utilidad acumulada tras cada consulta del stream, no solo la media final. Es lo que hace gratis la Figura B |

### Fase 2 — Métricas faltantes

| # | Tarea | Dónde |
|---|---|---|
| 2.1 | `answer_rate` / `empty_rate` | `metrics/utility.py` + `score_results.py` |
| 2.2 | `retrieval_precision_at_50` | `metrics/utility.py`. Gold = ranking no privado del mismo índice |
| 2.3 | Histograma de reutilización de documentos | script offline en `analysis/`. **Correr esto primero: predice los resultados sin gastar GPU** |

### Fase 3 — Descargas y calibración de costo

| # | Tarea |
|---|---|
| 3.1 | Descargar `arctic-embed-m-v1.5` + `Qwen2.5-3B-Instruct` (**ninguno está en el caché de 5.2 GB actual**) |
| 3.2 | Decidir sobre el corpus DPR (`data/dpr/` **no existe**, ~13 GB comprimidos, ~21M pasajes). Solo hace falta si NQ entra como reference condition con corpus realista |
| 3.3 | **Smoke run de calibración:** 1 dataset, 1 LLM, Q=10, 1 punto de ε, **los 8 métodos**. Verifica que `metrics_summary.csv` sale con las 8 filas, y **mide segundos/consulta por método** para poder dimensionar el grid real |

Esta fase 3.3 no es opcional. El grid principal se dimensiona **con los números medidos ahí**, no con estimaciones.

### Fase 4 — Grid principal (Tier A)

```
3 datasets (ChatDoctor, Medical Synth, MovieLens)
× 1 LLM (Qwen2.5-3B-Instruct)
× 3 semillas
× Q = 100 consultas
× {7 configuraciones DP con su grilla de ε} + vanilla + non-RAG
```

Grillas de ε por método, todas calibradas a los mismos objetivos de `ε_total ∈ {1, 3, 10, 30, 100}` a Q=100:

| Método | Qué se barre | Cómo se obtiene |
|---|---|---|
| DP-RAG, DPVoteRAG, DPSparseVoteRAG, DP-KSA | `ε_query` | inversión 1.3 desde cada `ε_total` objetivo |
| MuRAG, MuRAG-Ada | `ε_q` (con M=1, `ε_total = ε_q`) | directo |
| DP-SynRAG | parámetros de ruido | inversión 1.2 desde cada `ε_total` objetivo |

### Fase 5 — Robustez (Tier B, si el cómputo alcanza)

- NQ como reference condition, LLM primario, mismas grillas
- Barrido de LLM (`opt-1.3b`, `Llama-3.1-8B`) sobre **un** dataset, **una** semilla
- Barrido de `m ∈ {30,40,50}` para los métodos de votación, un dataset
- `M ∈ {1,5}` para MuRAG (es el ablation de su Figura 3)

### Fase 6 — MIA standalone

`python experiments/runner.py --mia-only`, sobre el mismo split member/non-member para los 7. Las cuatro métricas de `metodologia.tex:99-125`.

### Fase 7 — Análisis

- Figuras A, B y Tabla C de §8
- Curvas estratificadas por bin de redundancia (usa `data/dpr/biencoder-*-dev.json` o el histograma de 2.3)
- Actualizar `analysis/plot_tradeoff.py`, que hoy tiene colores y series hardcodeadas y ancla vanilla en ε=0

---

## Parte IV. Decisiones abiertas y riesgos

### 16. Decisiones que requieren tu criterio

1. **LLM primario del grid.** Recomiendo `Qwen2.5-3B-Instruct`. La tensión: Koga y MuRAG usan modelos **base** (OPT, Pythia, Llama base) porque la votación token a token con modelos instruct se comporta distinto. Si prefieres fidelidad a esos papers, el primario debería ser `opt-1.3b`, con el costo de un techo de utilidad bajo (su Non-RAG en ChatDoctor ya está en 0.838 de BERTScore, casi sin headroom).

2. **Q = 100 vs Q = 40.** Q=100 coincide con Koga, MuRAG y DP-KSA, y hace la Figura B mucho más legible. Q=40 es lo que corren los configs hoy y cuesta 2.5× menos.

3. **MuRAG a δ=0 (conservador, sin demostración nueva) vs re-derivación a orden Rényi finito** (más ajustado, favorece a MuRAG, requiere extender el Anexo A). Recomiendo δ=0 para el grid principal.

4. **Corpus DPR.** Descargarlo (~13 GB + horas de indexado) o quedarse con corpus stand-in de 20k. Ojo: `results/_review_archive_20260706/TODO.md` dice que el corpus de 20k **desactiva de facto el mecanismo DP de DP-KSA**.

### 17. Riesgos de contradicción con el manuscrito

| # | Riesgo | Dónde | Acción |
|---|---|---|---|
| 1 | `metodologia.tex:204-207` promete evaluar **todos** los LLMs del estado del arte ("Llama 3.1 8B, Llama 3.2 1B, OPT 1.3B y 2.7B, Pythia 1.4B, Mistral-7B, Phi-4-mini, Gemma-2 2B") | tesis | **suavizar el texto**: un LLM primario para el grid + un subconjunto como robustez |
| 2 | δ=1e-5 comprometido, configs en 1e-3/1e-4 | configs | Fase 0.4 |
| 3 | MIA standalone comprometida | ya se cumple (`--mia-only`) | ninguna |
| 4 | Vanilla como condición de referencia comprometida | `runner.py` | Fase 0.1 y 0.2 |
| 5 | Anexo B prohíbe tocar los mecanismos que producen el objeto liberado | anexos | el plan solo homogeneiza retriever, embedder, chunking, LLM y scoring, que el Anexo B marca como intercambiables |

### 18. Lo que este plan aporta que ningún paper tiene

1. **Los 7 algoritmos en un solo eje `ε_total(Q)`.** Hoy solo existen 2 comparaciones experimentales reales en toda la familia (MuRAG→Koga y DP-SynRAG→Grislain).
2. **La Figura B con 7 curvas.** DP-SynRAG la hizo con 2 métodos, MuRAG con 2. Con 7 aparecen los puntos de cruce reales.
3. **La corrección de la asimetría de contabilidad** (§4): ambos papers usan composición básica para el rival y contabilidad ajustada para sí mismos. A Q=100 eso es un factor 8.6×.
4. **La corrección del chunking** (§5): 4 de 7 métodos hoy reportan una garantía a nivel chunk como si fuera a nivel documento.
5. **Separar "no respondió" de "respondió mal"** (§12), que es la diferencia entre un método que colapsa y uno que se equivoca.
