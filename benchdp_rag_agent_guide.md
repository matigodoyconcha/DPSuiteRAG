# BenchDP-RAG — Guía de Implementación para Agente

> Este documento describe **cómo estructurar y construir** el framework de evaluación de métodos de Privacidad Diferencial en sistemas RAG. No contiene código implementado. Su propósito es servir como instrucción completa y ordenada para un agente que ejecutará la implementación paso a paso.

---

## 1. Propósito del Proyecto

Construir un framework Python modular, reproducible y extensible que permita:

1. Ejecutar un pipeline RAG completo con componentes intercambiables.
2. Substituir cualquier componente del pipeline por una versión que aplique Privacidad Diferencial (DP).
3. Evaluar y comparar 6 métodos del estado del arte bajo condiciones homogéneas.
4. Producir tablas y curvas del trade-off privacidad-utilidad para distintos valores de ε.

Los 6 métodos a implementar son: **LPRAG, DP-RAG, DPVoteRAG/DPSparseVoteRAG, RP-RAG, PRIVATE-RAG, DP-SynRAG**.

---

## 2. Árbol de Directorios

```
benchdp_rag/
│
├── core/                          # Clases abstractas del pipeline RAG
│   ├── __init__.py
│   ├── chunker.py                 # Abstract: BaseChunker
│   ├── embedder.py                # Abstract: BaseEmbedder
│   ├── indexer.py                 # Abstract: BaseIndexer
│   ├── retriever.py               # Abstract: BaseRetriever
│   ├── reranker.py                # Abstract: BaseReranker
│   ├── prompt_builder.py          # Abstract: BasePromptBuilder
│   ├── generator.py               # Abstract: BaseGenerator
│   ├── postprocessor.py           # Abstract: BasePostprocessor
│   └── pipeline.py                # RAGPipeline: orquesta todos los componentes
│
├── vanilla/                       # Implementación RAG estándar (sin DP)
│   ├── __init__.py
│   ├── chunker.py                 # SentenceChunker, FixedSizeChunker
│   ├── embedder.py                # HFEmbedder (HuggingFace sentence-transformers)
│   ├── indexer.py                 # FAISSIndexer
│   ├── retriever.py               # CosineSimilarityRetriever
│   ├── reranker.py                # IdentityReranker (pass-through, sin reranking)
│   ├── prompt_builder.py          # SimplePromptBuilder
│   ├── generator.py               # HFGenerator (HuggingFace causal LM)
│   └── postprocessor.py           # IdentityPostprocessor (pass-through)
│
├── dp_methods/                    # Un módulo por cada método DP
│   ├── lprag/
│   │   ├── __init__.py
│   │   ├── chunker.py             # Hereda BaseChunker — añade perturbación de entidades
│   │   └── README.md              # Descripción del método, paper, parámetros
│   │
│   ├── dp_rag/
│   │   ├── __init__.py
│   │   ├── retriever.py           # Hereda BaseRetriever — selección de docs con DP
│   │   ├── generator.py           # Hereda BaseGenerator — generación token a token con DP
│   │   └── README.md
│   │
│   ├── dpvoterag/
│   │   ├── __init__.py
│   │   ├── generator.py           # Hereda BaseGenerator — histograma de votos + SVT
│   │   ├── sparse_vote_generator.py  # Variante DPSparseVoteRAG
│   │   └── README.md
│   │
│   ├── rp_rag/
│   │   ├── __init__.py
│   │   ├── indexer.py             # Hereda BaseIndexer — proyección aleatoria gaussiana
│   │   ├── retriever.py           # Hereda BaseRetriever — búsqueda en espacio proyectado
│   │   └── README.md
│   │
│   ├── private_rag/
│   │   ├── __init__.py
│   │   ├── retriever.py           # Hereda BaseRetriever — filtro por budget Rényi DP
│   │   ├── budget_manager.py      # Lleva cuenta del budget por documento
│   │   └── README.md
│   │
│   └── dp_synrag/
│       ├── __init__.py
│       ├── indexer.py             # Hereda BaseIndexer — genera docs sintéticos DP antes de indexar
│       ├── generator.py           # Hereda BaseGenerator — generación token a token DP por cluster
│       └── README.md
│
├── metrics/                       # Módulo de métricas de utilidad y privacidad
│   ├── __init__.py
│   ├── utility.py                 # MatchAccuracy, BERTScore F1, ROUGE-L, BLEU
│   └── privacy.py                 # MIA_AUC (BLEU-based), PrivateEntityLeakage
│
├── datasets/                      # Carga y normalización de datasets
│   ├── __init__.py
│   ├── base_dataset.py            # Abstract: BaseDataset
│   ├── natural_questions.py
│   ├── triviaqa.py
│   ├── hotpotqa.py
│   ├── squad.py
│   └── chatdoctor.py
│
├── llms/                          # Registro de LLMs configurables por nombre
│   ├── __init__.py
│   ├── base_llm.py                # Abstract: BaseLLM
│   └── huggingface.py             # HuggingFaceLLM (model_name configurable)
│
├── experiments/                   # Scripts de experimento por método + runner central
│   ├── configs/                   # Archivos YAML de configuración por experimento
│   │   ├── vanilla.yaml
│   │   ├── lprag.yaml
│   │   ├── dp_rag.yaml
│   │   ├── dpvoterag.yaml
│   │   ├── rp_rag.yaml
│   │   ├── private_rag.yaml
│   │   └── dp_synrag.yaml
│   │
│   ├── run_vanilla.py             # Ejecuta pipeline vanilla y guarda resultados
│   ├── run_lprag.py
│   ├── run_dp_rag.py
│   ├── run_dpvoterag.py
│   ├── run_rp_rag.py
│   ├── run_private_rag.py
│   ├── run_dp_synrag.py
│   └── runner.py                  # CLI central: orquesta todos los scripts
│
├── results/                       # Resultados generados (NO se commitean al repo)
│   └── .gitkeep
│
├── analysis/                      # Scripts para generar tablas y figuras
│   ├── plot_tradeoff.py           # Curvas ε vs utilidad y ε vs privacidad
│   └── summary_table.py           # Tabla comparativa LaTeX/CSV
│
├── tests/
│   ├── test_core.py               # Tests de las clases abstractas y pipeline
│   ├── test_vanilla.py            # Tests de la implementación vanilla
│   ├── test_metrics.py            # Tests de métricas con valores conocidos
│   └── test_datasets.py           # Tests de carga de datasets
│
├── requirements.txt
├── README.md
└── setup.py
```

---

## 3. Descripción de las Clases Abstractas (`core/`)

Cada clase abstracta define el **contrato** que deben cumplir tanto la implementación vanilla como cada método DP. El agente debe implementar **primero** estas clases, ya que todo lo demás depende de ellas.

### 3.1 `BaseChunker` (`core/chunker.py`)

Responsabilidad: recibir un documento de texto y devolver una lista de fragmentos (chunks).

Métodos abstractos que debe exponer:
- `chunk(document: str) -> list[str]` — divide el documento en chunks.
- `get_config() -> dict` — devuelve los parámetros usados (tamaño de chunk, overlap, etc.).

Implementaciones esperadas:
- **Vanilla**: `FixedSizeChunker` (por número de tokens) y `SentenceChunker` (por oraciones).
- **LPRAG**: `EntityPerturbedChunker` — igual que vanilla pero aplica perturbación DP sobre entidades antes de devolver los chunks.

---

### 3.2 `BaseEmbedder` (`core/embedder.py`)

Responsabilidad: convertir una lista de strings (chunks o queries) en vectores numéricos.

Métodos abstractos:
- `embed(texts: list[str]) -> list[vector]` — devuelve una lista de embeddings.
- `embed_query(query: str) -> vector` — caso especial para queries en tiempo de inferencia.

Implementaciones esperadas:
- **Vanilla**: `HFEmbedder` usando `sentence-transformers`.
- **RP-RAG**: el embedder en sí es vanilla; la proyección ocurre en el Indexer.

---

### 3.3 `BaseIndexer` (`core/indexer.py`)

Responsabilidad: recibir vectors + metadatos de los chunks y construir/persistir el índice de búsqueda.

Métodos abstractos:
- `build(vectors, metadata: list[dict])` — construye el índice desde cero.
- `save(path: str)` / `load(path: str)` — persistencia del índice.
- `get_index()` — devuelve la estructura interna del índice.

Implementaciones esperadas:
- **Vanilla**: `FAISSIndexer` — índice FAISS plano (IndexFlatIP o IndexFlatL2).
- **RP-RAG**: `RandomProjectionIndexer` — aplica una matriz gaussiana aleatoria sobre los vectors antes de construir el índice FAISS.
- **DP-SynRAG**: `DPSyntheticIndexer` — en lugar de indexar los vectores de documentos originales, genera documentos sintéticos con DP y los indexa en su lugar.

---

### 3.4 `BaseRetriever` (`core/retriever.py`)

Responsabilidad: dado un vector de query y un índice, devolver los top-K chunks más relevantes.

Métodos abstractos:
- `retrieve(query_vector, k: int) -> list[dict]` — devuelve los K chunks con sus metadatos y scores.

Implementaciones esperadas:
- **Vanilla**: `CosineSimilarityRetriever` — búsqueda exacta por similitud coseno en FAISS.
- **DP-RAG**: `DPRetriever` — selecciona los K documentos usando un umbral ruidoso (mecanismo DP para el threshold τ).
- **RP-RAG**: `RPRetriever` — hace la búsqueda en el espacio proyectado usando el mismo vector de transformación que el Indexer.
- **PRIVATE-RAG**: `BudgetAwareRetriever` — consulta el `BudgetManager` antes de incluir cada documento; excluye documentos con budget agotado.

---

### 3.5 `BaseReranker` (`core/reranker.py`)

Responsabilidad: reordenar la lista de chunks recuperados según un criterio secundario.

Métodos abstractos:
- `rerank(query: str, chunks: list[dict]) -> list[dict]` — devuelve los chunks reordenados.

Implementaciones esperadas:
- **Vanilla**: `IdentityReranker` — devuelve la lista sin modificar (pass-through). Útil para mantener la interfaz sin overhead cuando no se necesita reranking.
- En los métodos DP actuales no se requiere reranker propio; todos usan `IdentityReranker`.

---

### 3.6 `BasePromptBuilder` (`core/prompt_builder.py`)

Responsabilidad: combinar la query del usuario y los chunks recuperados en un prompt listo para el LLM.

Métodos abstractos:
- `build(query: str, chunks: list[dict]) -> str` — devuelve el prompt completo como string.

Implementaciones esperadas:
- **Vanilla**: `SimplePromptBuilder` — template estático tipo `"Contexto: {chunks}\n\nPregunta: {query}\nRespuesta:"`.
- **DP-RAG**: `PerDocPromptBuilder` — genera un prompt independiente por cada documento (uno por doc, no uno con todos concatenados).

---

### 3.7 `BaseGenerator` (`core/generator.py`)

Responsabilidad: recibir un prompt (o lista de prompts) y generar una respuesta de texto.

Métodos abstractos:
- `generate(prompt: str) -> str` — generación estándar a partir de un solo prompt.
- `generate_with_dp(prompts: list[str], epsilon: float, delta: float) -> str` — generación con garantías DP (puede no implementarse en todos los métodos; en vanilla puede levantar `NotImplementedError`).

Implementaciones esperadas:
- **Vanilla**: `HFGenerator` — carga un modelo causal de HuggingFace y genera texto con `model.generate()`.
- **DP-RAG**: `DPAggregatedGenerator` — recibe K prompts (uno por doc), genera distribuciones de tokens en paralelo, las agrega con ruido DP y samplea token a token.
- **DPVoteRAG**: `VoteRAGGenerator` — recibe K×M prompts, construye histograma de votos por token, aplica ruido Laplaciano y SVT para decidir el token a publicar.
- **DPSparseVoteRAG**: `SparseVoteRAGGenerator` — extiende el anterior con un LLM público; si el token del LLM público coincide con el ganador, omite el gasto de privacidad.
- **DP-SynRAG**: la generación DP ocurre durante la fase de indexación (en `DPSyntheticIndexer`), por lo que en inferencia se usa `HFGenerator` vanilla.

---

### 3.8 `BasePostprocessor` (`core/postprocessor.py`)

Responsabilidad: limpiar o transformar la respuesta generada antes de entregarla al evaluador.

Métodos abstractos:
- `process(response: str) -> str` — devuelve la respuesta procesada.

Implementaciones esperadas:
- **Vanilla**: `IdentityPostprocessor` — devuelve la respuesta sin modificar.
- Todos los métodos DP actuales usan `IdentityPostprocessor`.

---

### 3.9 `RAGPipeline` (`core/pipeline.py`)

Clase concreta (no abstracta) que orquesta todos los componentes. Recibe instancias de cada componente en su constructor y expone dos métodos principales:

- `index(documents: list[str])` — ejecuta la fase offline: Chunker → Embedder → Indexer.
- `query(query: str, k: int) -> str` — ejecuta la fase online: Embedder → Retriever → Reranker → PromptBuilder → Generator → Postprocessor.

El `RAGPipeline` no sabe nada de DP; simplemente llama a los componentes que se le inyectan. Si se inyecta un `DPRetriever` en lugar de un `CosineSimilarityRetriever`, el pipeline automáticamente se vuelve DP en esa etapa.

---

## 4. Descripción de los Métodos DP (`dp_methods/`)

Cada módulo solo sobreescribe los componentes del pipeline que modifica. Los demás componentes se toman directamente de `vanilla/`.

### 4.1 LPRAG (`dp_methods/lprag/`)

Componente modificado: **Chunker**.

La lógica de perturbación ocurre en `EntityPerturbedChunker`. El chunker debe:
1. Aplicar chunking estándar (puede delegar internamente en `FixedSizeChunker`).
2. Detectar entidades sensibles en cada chunk (personas, lugares, diagnósticos) usando un NER.
3. Para cada entidad, aplicar el mecanismo de exponenciación con parámetro ε para sustituirla por una alternativa semánticamente similar.
4. Devolver los chunks perturbados.

Parámetros del módulo: `epsilon`, `entity_types`, `candidate_model`.

---

### 4.2 DP-RAG (`dp_methods/dp_rag/`)

Componentes modificados: **Retriever** + **Generator**.

`DPRetriever`: selecciona los K documentos usando un umbral τ estimado con garantías DP (en lugar de tomar simplemente los top-K por score).

`DPAggregatedGenerator`: recibe la lista de K documentos seleccionados y genera la respuesta token a token. Por cada token:
1. Genera un prompt por cada documento.
2. Obtiene las distribuciones de probabilidad sobre el vocabulario de cada prompt.
3. Agrega las distribuciones añadiendo ruido calibrado a ε.
4. Samplea el token ganador de la distribución agregada.
5. Repite hasta generar el EOS o alcanzar max_tokens.

Parámetros del módulo: `epsilon`, `delta`, `num_docs`, `public_dataset_path`.

---

### 4.3 DPVoteRAG / DPSparseVoteRAG (`dp_methods/dpvoterag/`)

Componente modificado: **Generator**.

`VoteRAGGenerator`:
1. Recibe K chunks, los divide en M particiones → K×M sub-documentos.
2. Para cada sub-documento, genera la distribución del próximo token con un LLM.
3. Construye un histograma de votos por token.
4. Añade ruido Laplaciano al histograma.
5. Usa SVT (Sparse Vector Technique) con umbral ruidoso para decidir si publicar el token ganador o continuar.
6. Repite token a token.

`SparseVoteRAGGenerator`: extiende el anterior.
1. Además de los K×M LLMs, corre un LLM público (sin documentos).
2. Si el token del LLM público coincide con el token ganador del histograma, lo publica sin gastar budget DP.
3. Solo gasta ε cuando el token ganador difiere del LLM público.

Parámetros del módulo: `epsilon`, `k`, `m`, `svt_threshold`, `public_model`.

---

### 4.4 RP-RAG (`dp_methods/rp_rag/`)

Componentes modificados: **Indexer** + **Retriever**.

`RandomProjectionIndexer`:
1. Genera una matriz de proyección aleatoria gaussiana R de dimensión `(d_original, d_projected)`. Esta matriz se genera una sola vez y se persiste junto con el índice.
2. Aplica la proyección `v' = v @ R` sobre cada vector de embedding antes de añadirlo al índice FAISS.

`RPRetriever`:
1. Al recibir un vector de query, aplica la misma proyección R.
2. Hace la búsqueda por similitud en el espacio proyectado.

Nota importante para el agente: este método **no tiene garantías DP formales** (no hay ε ni δ). Debe documentarse explícitamente en el `README.md` del módulo y tratarse como baseline heurístico en la evaluación.

Parámetros del módulo: `d_projected`, `random_seed`.

---

### 4.5 PRIVATE-RAG (`dp_methods/private_rag/`)

Componentes modificados: **Retriever** + un componente nuevo **BudgetManager**.

`BudgetManager` (`budget_manager.py`): componente auxiliar (no hereda de ninguna clase abstracta del pipeline). Mantiene un diccionario `{doc_id: remaining_budget}`. Expone:
- `can_query(doc_id) -> bool` — retorna True si el documento aún tiene budget.
- `consume(doc_id, cost: float)` — descuenta del budget del documento.
- `reset()` / `save_state()` / `load_state()` — gestión del estado.

`BudgetAwareRetriever` (variante MURAG):
1. Recupera los top-K candidatos por similitud (igual que vanilla).
2. Filtra los candidatos cuyos `doc_id` tienen budget agotado según `BudgetManager`.
3. Devuelve solo los documentos con budget disponible.

`AdaptiveBudgetAwareRetriever` (variante MURAG-ADA):
1. Usa un umbral de similitud adaptativo por consulta.
2. Sigue recuperando documentos hasta obtener K que superen el umbral dinámico y tengan budget.

Parámetros del módulo: `epsilon_per_query`, `delta`, `initial_budget_per_doc`, `renyi_order`.

---

### 4.6 DP-SynRAG (`dp_methods/dp_synrag/`)

Componente modificado: **Indexer** (la "magia" ocurre en la fase offline, no en la inferencia).

`DPSyntheticIndexer`:
1. **Fase de clustering privado**: agrupa los documentos originales en clusters usando DP-clustering (histograma de keywords con ruido Laplaciano sobre embeddings medios).
2. **Fase de generación sintética**: para cada cluster, genera documentos sintéticos token a token:
   - Un LLM por documento hace rephrase (prompt: "rephrase the following: {doc}").
   - Se acumulan los logits de todos los LLMs del cluster.
   - Se añade ruido calibrado a ε, se clipea, y se samplea el token más probable.
   - Se repite hasta generar T tokens.
3. **Fase de auto-filtrado**: un LLM evaluador (sin acceso a documentos originales) decide si cada documento sintético es útil para la tarea. Los que no pasan el filtro se descartan.
4. Los documentos sintéticos aprobados se embeben y se indexan en FAISS como en vanilla.

En inferencia, el `RAGPipeline` usa `CosineSimilarityRetriever` y `HFGenerator` vanilla sobre el índice sintético.

Parámetros del módulo: `epsilon`, `delta`, `n_clusters`, `tokens_per_doc`, `clip_threshold`, `evaluator_model`.

---

## 5. Módulo de Métricas (`metrics/`)

### 5.1 Métricas de Utilidad (`metrics/utility.py`)

Cada función recibe una lista de respuestas generadas y una lista de respuestas de referencia (gold), y devuelve un escalar.

- `match_accuracy(predictions, references) -> float` — porcentaje de respuestas que contienen exactamente la respuesta de referencia (case-insensitive).
- `bertscore_f1(predictions, references, model_name) -> float` — F1 de BERTScore usando un modelo de referencia.
- `rouge_l(predictions, references) -> float` — F-measure de ROUGE-L.
- `bleu(predictions, references) -> float` — BLEU score estándar.

### 5.2 Métricas de Privacidad (`metrics/privacy.py`)

- `mia_auc_bleu(pipeline, datastore_docs, non_member_docs, n_queries) -> float` — implementa el ataque MIA basado en BLEU-score de DPVoteRAG. Genera queries sobre documentos miembro y no-miembro, calcula el BLEU entre la respuesta generada y el documento, y computa el AUC de la separación. Retorna el AUC (0.5 = sin información, 1.0 = ataque perfecto).
- `private_entity_leakage(responses, private_entities) -> float` — cuenta cuántas entidades privadas del datastore aparecen en las respuestas generadas.

---

## 6. Módulo de Datasets (`datasets/`)

### 6.1 `BaseDataset` (`datasets/base_dataset.py`)

Clase abstracta que estandariza la interfaz de todos los datasets. Métodos abstractos:
- `load() -> None` — descarga o carga el dataset desde disco.
- `get_documents() -> list[str]` — devuelve la lista de documentos para indexar.
- `get_queries() -> list[dict]` — devuelve lista de `{query: str, answer: str}`.
- `get_private_entities() -> list[str]` — devuelve entidades sensibles conocidas (para la métrica de leakage).

### 6.2 Datasets concretos

Cada archivo implementa `BaseDataset` para un dataset específico:

- `natural_questions.py` — NQ abierto, dominio de conocimiento general.
- `triviaqa.py` — TriviaQA, dominio de conocimiento general.
- `hotpotqa.py` — HotpotQA, preguntas multi-hop.
- `squad.py` — SQuAD, comprensión lectora.
- `chatdoctor.py` — ChatDoctor, dominio médico. **Prioritario** por ser el único dominio médico.

### 6.3 Módulo de LLMs (`llms/`)

Para desacoplar la selección del LLM del pipeline, se define un módulo de LLMs configurable de forma análoga a `dataset.name`:

- `BaseLLM` expone la interfaz para construir el generador del pipeline.
- `HuggingFaceLLM` implementa el proveedor inicial y recibe `model_name`.
- La selección se realiza por `llm.name` en YAML, permitiendo extender a nuevos proveedores sin modificar `RAGPipeline`.

---

## 7. Experimentos (`experiments/`)

### 7.1 Archivos de Configuración YAML (`experiments/configs/`)

Cada YAML define un experimento completamente reproducible:

```yaml
# Ejemplo: experiments/configs/lprag.yaml
method: lprag
dataset:
  name: natural_questions
llm:
  name: huggingface
  model_name: mistral-7b-instruct
models:
  embedder: sentence-transformers/all-MiniLM-L6-v2
epsilon_values: [1, 3, 5, 7, 9]
delta: 1.0e-5
k_retrieved: 5
metrics:
  utility: [match_accuracy, bertscore_f1, rouge_l]
  privacy: [mia_auc_bleu, private_entity_leakage]
output_dir: results/lprag/
```

### 7.2 Scripts por Método (`experiments/run_*.py`)

Cada script:
1. Carga la configuración YAML de su método.
2. Instancia los componentes correspondientes (vanilla + DP donde aplica).
3. Construye el `RAGPipeline`.
4. Itera sobre los valores de ε definidos en el YAML.
5. Para cada ε: indexa el dataset, ejecuta todas las queries, calcula todas las métricas.
6. Guarda los resultados en `results/{method}/{dataset}/epsilon_{e}.json`.

### 7.3 Runner Central (`experiments/runner.py`)

CLI que acepta los siguientes argumentos:
- `--method` — nombre del método (`vanilla`, `lprag`, `dp_rag`, etc.) o `all` para correr todos.
- `--dataset` — nombre del dataset o `all`.
- `--llm-name` — proveedor de LLM (ej. `huggingface`).
- `--llm-model` — nombre del modelo dentro del proveedor elegido.
- `--config` — path a un YAML específico (opcional, sobreescribe los defaults).
- `--dry-run` — imprime qué experimentos se ejecutarían sin ejecutarlos.

Internamente llama a cada `run_*.py` como subprocess o lo importa y ejecuta su función `main()`.

---

## 8. Análisis y Visualización (`analysis/`)

### `plot_tradeoff.py`

Lee los JSONs de resultados en `results/` y genera:
- Curva ε vs. utilidad (una línea por método) por dataset.
- Curva ε vs. privacidad (AUC-MIA) por dataset.
- Guarda las figuras en `results/figures/`.

### `summary_table.py`

Genera una tabla comparativa en CSV y LaTeX con columnas: Método, Dataset, ε, MatchAccuracy, BERTScore F1, ROUGE-L, MIA-AUC, EntityLeakage.

---

## 9. Tests (`tests/`)

Tests simples con pytest. No se busca cobertura exhaustiva, solo verificar que los componentes son instanciables y sus interfaces funcionan.

### `tests/test_core.py`
- Verificar que `RAGPipeline` acepta instancias de las clases abstractas.
- Verificar que `RAGPipeline.index()` y `RAGPipeline.query()` ejecutan sin errores con componentes stub (mocks).

### `tests/test_vanilla.py`
- Verificar que el pipeline vanilla completo corre end-to-end con un documento y una query de prueba pequeños.
- Verificar que `FixedSizeChunker` divide correctamente un texto de ejemplo.
- Verificar que `FAISSIndexer` construye un índice y devuelve resultados de búsqueda.

### `tests/test_metrics.py`
- Verificar `match_accuracy` con casos conocidos (predicción == referencia → 1.0, predicción vacía → 0.0).
- Verificar `rouge_l` con un par de strings conocidos.
- Verificar que `mia_auc_bleu` devuelve un valor entre 0.0 y 1.0.

### `tests/test_datasets.py`
- Verificar que cada dataset implementa correctamente la interfaz `BaseDataset`.
- Verificar que `get_documents()` devuelve una lista no vacía.
- Verificar que `get_queries()` devuelve dicts con las claves `query` y `answer`.

---

## 10. TO-DO List de Implementación

El agente debe ejecutar estas tareas **en el orden indicado**. Cada tarea es independiente del siguiente bloque solo cuando el bloque anterior esté completo.

---

### BLOQUE 0 — Setup inicial

- [ ] **0.1** Crear el árbol de directorios completo con los `__init__.py` y `README.md` vacíos.
- [ ] **0.2** Crear `requirements.txt` con las dependencias necesarias: `torch`, `transformers`, `sentence-transformers`, `faiss-cpu`, `datasets` (HuggingFace), `bert-score`, `rouge-score`, `nltk`, `numpy`, `pyyaml`, `pytest`, `tqdm`.
- [ ] **0.3** Crear `setup.py` para instalar el paquete en modo editable (`pip install -e .`).

---

### BLOQUE 1 — Clases abstractas (`core/`)

- [ ] **1.1** Implementar `BaseChunker` en `core/chunker.py`.
- [ ] **1.2** Implementar `BaseEmbedder` en `core/embedder.py`.
- [ ] **1.3** Implementar `BaseIndexer` en `core/indexer.py`.
- [ ] **1.4** Implementar `BaseRetriever` en `core/retriever.py`.
- [ ] **1.5** Implementar `BaseReranker` en `core/reranker.py`.
- [ ] **1.6** Implementar `BasePromptBuilder` en `core/prompt_builder.py`.
- [ ] **1.7** Implementar `BaseGenerator` en `core/generator.py`.
- [ ] **1.8** Implementar `BasePostprocessor` en `core/postprocessor.py`.
- [ ] **1.9** Implementar `RAGPipeline` en `core/pipeline.py`.

> **Criterio de completitud**: cada clase abstracta tiene sus métodos decorados con `@abstractmethod`. `RAGPipeline` instancia con mocks y ejecuta `index()` y `query()` sin errores.

---

### BLOQUE 2 — Implementación Vanilla (`vanilla/`)

- [ ] **2.1** Implementar `FixedSizeChunker` y `SentenceChunker` en `vanilla/chunker.py`.
- [ ] **2.2** Implementar `HFEmbedder` en `vanilla/embedder.py`.
- [ ] **2.3** Implementar `FAISSIndexer` en `vanilla/indexer.py`.
- [ ] **2.4** Implementar `CosineSimilarityRetriever` en `vanilla/retriever.py`.
- [ ] **2.5** Implementar `IdentityReranker` en `vanilla/reranker.py`.
- [ ] **2.6** Implementar `SimplePromptBuilder` en `vanilla/prompt_builder.py`.
- [ ] **2.7** Implementar `HFGenerator` en `vanilla/generator.py`.
- [ ] **2.8** Implementar `IdentityPostprocessor` en `vanilla/postprocessor.py`.
- [ ] **2.9** Ensamblar y probar el pipeline vanilla completo end-to-end con un documento de prueba.

> **Criterio de completitud**: `run_vanilla.py` ejecuta sin errores sobre NaturalQuestions (subconjunto de 10 ejemplos) y produce un JSON de resultados.

---

### BLOQUE 3 — Módulo de Datasets (`datasets/`)

- [ ] **3.1** Implementar `BaseDataset` en `datasets/base_dataset.py`.
- [ ] **3.2** Implementar `NaturalQuestionsDataset`.
- [ ] **3.3** Implementar `TriviaQADataset`.
- [ ] **3.4** Implementar `HotpotQADataset`.
- [ ] **3.5** Implementar `SQuADDataset`.
- [ ] **3.6** Implementar `ChatDoctorDataset`. *(Prioridad alta — único dominio médico)*

> **Criterio de completitud**: `test_datasets.py` pasa para los 5 datasets.

---

### BLOQUE 4 — Módulo de Métricas (`metrics/`)

- [ ] **4.1** Implementar `match_accuracy`, `bertscore_f1`, `rouge_l`, `bleu` en `metrics/utility.py`.
- [ ] **4.2** Implementar `mia_auc_bleu` en `metrics/privacy.py`.
- [ ] **4.3** Implementar `private_entity_leakage` en `metrics/privacy.py`.

> **Criterio de completitud**: `test_metrics.py` pasa con valores de referencia conocidos.

---

### BLOQUE 5 — Implementación de Métodos DP

Cada sub-bloque puede desarrollarse en paralelo una vez que el Bloque 2 esté completo.

#### 5.1 LPRAG
- [ ] **5.1.1** Implementar el NER para detección de entidades (puede usar `spacy` o un modelo HF de NER).
- [ ] **5.1.2** Implementar el mecanismo de exponenciación para sustitución DP de entidades.
- [ ] **5.1.3** Implementar `EntityPerturbedChunker` en `dp_methods/lprag/chunker.py`.
- [ ] **5.1.4** Crear `experiments/run_lprag.py` y `experiments/configs/lprag.yaml`.
- [ ] **5.1.5** Ejecutar experimento piloto (ε=5, NQ, 20 queries) y verificar que los resultados son razonables.

#### 5.2 DP-RAG
- [ ] **5.2.1** Implementar `DPRetriever` con umbral τ ruidoso.
- [ ] **5.2.2** Implementar `PerDocPromptBuilder`.
- [ ] **5.2.3** Implementar `DPAggregatedGenerator` con generación token a token.
- [ ] **5.2.4** Crear `experiments/run_dp_rag.py` y `experiments/configs/dp_rag.yaml`.
- [ ] **5.2.5** Ejecutar experimento piloto.

#### 5.3 DPVoteRAG / DPSparseVoteRAG
- [ ] **5.3.1** Implementar la lógica de partición de chunks en M secciones.
- [ ] **5.3.2** Implementar `VoteRAGGenerator` con histograma Laplaciano y SVT.
- [ ] **5.3.3** Implementar `SparseVoteRAGGenerator` con fuente pública.
- [ ] **5.3.4** Crear `experiments/run_dpvoterag.py` y el YAML correspondiente.
- [ ] **5.3.5** Ejecutar experimento piloto. Medir tiempo por consulta (advertencia: costo computacional alto).

#### 5.4 RP-RAG
- [ ] **5.4.1** Implementar generación y persistencia de la matriz de proyección aleatoria.
- [ ] **5.4.2** Implementar `RandomProjectionIndexer`.
- [ ] **5.4.3** Implementar `RPRetriever`.
- [ ] **5.4.4** Crear `experiments/run_rp_rag.py` y el YAML.
- [ ] **5.4.5** Ejecutar experimento piloto. Nota: documentar que este método no tiene ε formal.

#### 5.5 PRIVATE-RAG
- [ ] **5.5.1** Implementar `BudgetManager` con soporte para Rényi DP composicional.
- [ ] **5.5.2** Implementar `BudgetAwareRetriever` (MURAG).
- [ ] **5.5.3** Implementar `AdaptiveBudgetAwareRetriever` (MURAG-ADA).
- [ ] **5.5.4** Crear `experiments/run_private_rag.py` y el YAML.
- [ ] **5.5.5** Ejecutar experimento piloto.

#### 5.6 DP-SynRAG
- [ ] **5.6.1** Implementar el clustering privado basado en histograma de keywords.
- [ ] **5.6.2** Implementar la generación sintética token a token (fase de logit aggregation + ruido).
- [ ] **5.6.3** Implementar el self-filtering con LLM evaluador.
- [ ] **5.6.4** Ensamblar `DPSyntheticIndexer`.
- [ ] **5.6.5** Crear `experiments/run_dp_synrag.py` y el YAML.
- [ ] **5.6.6** Ejecutar experimento piloto. Nota: este es el método con mayor costo computacional.

---

### BLOQUE 6 — Runner Central y Tests

- [ ] **6.1** Implementar `experiments/runner.py` con la CLI descrita.
- [ ] **6.2** Verificar que `runner.py --method all --dataset natural_questions --dry-run` muestra todos los experimentos correctamente.
- [ ] **6.3** Escribir `tests/test_core.py`.
- [ ] **6.4** Escribir `tests/test_vanilla.py`.
- [ ] **6.5** Verificar que todos los tests pasan con `pytest tests/ -v`.

---

### BLOQUE 7 — Experimentos Completos

- [ ] **7.1** Ejecutar todos los métodos sobre NaturalQuestions con ε ∈ {1, 3, 5, 7, 9}.
- [ ] **7.2** Ejecutar todos los métodos sobre ChatDoctor con ε ∈ {1, 3, 5, 7, 9}.
- [ ] **7.3** Ejecutar el baseline vanilla (sin DP) como referencia en ambos datasets.
- [ ] **7.4** Verificar que todos los JSONs de resultados están completos y sin NaN.

---

### BLOQUE 8 — Análisis y Figuras

- [ ] **8.1** Implementar `analysis/summary_table.py` y generar la tabla comparativa.
- [ ] **8.2** Implementar `analysis/plot_tradeoff.py` y generar las curvas ε-utilidad y ε-privacidad.
- [ ] **8.3** Revisar manualmente que las figuras son coherentes con lo esperado.
- [ ] **8.4** Exportar figuras en formato PDF y PNG para incluir en la tesis.

---

## 11. Notas Importantes para el Agente

1. **Orden estricto de bloques**: no iniciar el Bloque 5 sin que el Bloque 2 esté completamente funcional y probado. El Bloque 4 puede hacerse en paralelo al Bloque 3.

2. **Inyección de dependencias**: el `RAGPipeline` nunca debe importar directamente desde `vanilla/` ni desde `dp_methods/`. Siempre recibe instancias desde afuera. Esto es lo que hace el framework extensible.

3. **RP-RAG no es DP formal**: cuando se implemente y documente RP-RAG, debe quedar explícito que no tiene parámetro ε. En los experimentos, su fila en la tabla comparativa debe dejar las columnas de ε en blanco o marcadas como "N/A".

4. **DPSparseVoteRAG tiene una limitación formal conocida**: el mecanismo de ahorro de budget cuando el token coincide con el LLM público no tiene justificación formal clara en el paper. Documentar esto en el README del módulo y en el análisis de resultados.

5. **Reproducibilidad**: todos los experimentos deben fijar una semilla aleatoria (`random_seed` en el YAML). La semilla debe propagarse a PyTorch, NumPy y Python random.

6. **Escalabilidad**: DPVoteRAG requiere K×M inferencias por consulta. Para los experimentos piloto usar K=5, M=2 como máximo. Para los experimentos completos, medir el tiempo por consulta y reportarlo en la tabla comparativa.
