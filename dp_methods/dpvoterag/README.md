# DPVoteRAG / DPSparseVoteRAG

Implementación de los métodos DPVoteRAG y DPSparseVoteRAG descritos en
Koga et al., *"Privacy-Preserving Retrieval-Augmented Generation with Differential
Privacy"* ([arXiv:2412.04697](https://arxiv.org/html/2412.04697v1)).

Esta implementación replica el comportamiento del repo oficial
[tacchan7412/DPRAG](https://github.com/tacchan7412/DPRAG).

## Componentes

- [`generator.py`](generator.py) — `VoteRAGGenerator` con el mecanismo
  Limited Domain Gumbel (Durfee & Rogers, NeurIPS 2019).
- [`sparse_vote_generator.py`](sparse_vote_generator.py) — `SparseVoteRAGGenerator`
  que extiende el anterior con Sparse Vector Technique (Dwork & Roth).
- [`pipeline.py`](pipeline.py) — orquestación RAG (embedder, indexer, retriever,
  generator, postprocessor).

El accounting `(ε,δ)` se delega en
[`metrics/privacy_accounting_autodp.py::AutodpLDAccountant`](../../metrics/privacy_accounting_autodp.py),
que es un port directo de `utils/dp_accountant.py` del repo oficial (EM-peeling
con composición RDP, conversión BBGHS).

## DPVoteRAG (`VoteRAGGenerator`)

Para cada token generado:

1. Particiona aleatoriamente los `m·k` documentos recuperados en `m` voters
   (cada uno con `k` documentos).
2. Construye `m` prompts (idéntico al referente:
   `"Instruction: ... Context: 1. doc1; 2. doc2 ... Question: ... Answer:"`).
3. Cada voter produce un siguiente token vía greedy decoding.
4. Se agrega un histograma de votos y se aplica **Limited Domain Gumbel**:
   - Se ordenan los conteos en orden descendente y se toman los top `k_bar`.
   - Umbral: `h_⊥ = h_{k_bar} + 1 + log(min(sens, k_bar, dim − k_bar) / δ) / ε`.
   - Ruido Gumbel(0, 1/ε) sobre los conteos y sobre el umbral.
   - Si `max(v_counts) ≥ v_⊥` ⇒ se elige el token ganador (success).
   - En caso contrario ⇒ `NotFoundLDTop1`. El comportamiento depende de `fail_mode`:
     - `'ld_pate'` (default del paper, `utils/ensemble.py:127`): reintenta una vez
       con `k_bar = dim − 1` (1-excluded LD); si vuelve a fallar, detiene
       (`exit_status=1`).
     - `'stop'`: detiene la generación inmediatamente (`exit_status=1`).
5. El accountant registra el costo `ε` por token; si se excede `epsilon_total`
   o `delta_total`, la generación se detiene con `exit_status=2`.

### Parámetros relevantes

| Parámetro | Descripción | Default |
|---|---|---|
| `m` | Número de voters | 30 |
| `k` | Documentos por voter | 1 |
| `epsilon_token` / `delta_token` | Presupuesto LD por token | 2.0 / 1e-5 |
| `epsilon_total` / `delta_total` | Presupuesto total por query (online accountant) | 10.0 / 1e-4 |
| `k_bar` | Tamaño del dominio reducido en LD | `m` |
| `sensitivity` | l₀-sensibilidad (add/remove ⇒ 2) | 2.0 |
| `tmax_absolute` | Máximo de tokens por respuesta | 50 |
| `n_peels` | Top-i peels por token (1 = top-1, como el referente) | 1 |
| `fail_mode` | `'ld_pate'` (paper default) o `'stop'` | `'ld_pate'` |
| `random_seed` | Semilla para particionado y ruido | 42 |

> **Nota:** si `len(docs) < m·k` se lanza `ValueError`. El paper asume
> `n_split·n_docs` recuperados; ajusta `k_retrieved` o reduce `m`/`k` en el config.

### Salida (`last_run_stats`)

- `exit_status`: 0 = OK, 1 = LD falló (stop), 2 = budget agotado.
- `tokens_generated`, `n_private_tokens`, `n_nonprivate_tokens`.
- `dp_spent_epsilon`, `dp_spent_delta`, `dp_summary`.

## DPSparseVoteRAG (`SparseVoteRAGGenerator`)

Extiende DPVoteRAG con un Sparse Vector Technique formal. En cada paso:

1. Se genera un token público `y_nonRAG` con un prompt sin contexto
   (no consume presupuesto LD).
2. Se mide su nivel de acuerdo `a_t` con los voters privados.
3. Se compara `a_t + Lap(0, 4/sparse_eps)` contra `τ̂ = τ + Lap(0, 2/sparse_eps)`.
4. **Si `a_t_noisy ≤ τ̂` ⇒ aplica LD-Gumbel** (token privado, gasta `epsilon_token`).
   El umbral `τ̂` se reservoea sólo después de un paso privado, siguiendo SVT.
5. **Si `a_t_noisy > τ̂` ⇒ acepta el token público** (sin costo LD).

### Exclusión opcional de tokens triviales

- `exclude_stopwords`: stopwords cargadas de `data/stopword_words.pt`
  (convención del paper: el id se toma como `tokenizer(word)['input_ids'][1]`,
  saltando `<bos>`).
- `exclude_question_words`: ids de la pregunta (incluye `<bos>`, como el referente).

Cuando un token cae en la lista de exclusión, se acepta directamente y se
registra `flag=None` en `dp_flags_per_token`.

### Composición de presupuestos

- `sparse_eps`: presupuesto del SVT por paso (se contabiliza como `2·sparse_eps`
  por iteración, composición clásica de Dwork-Roth).
- `epsilon_token`: presupuesto LD-Gumbel por cada paso privado.

El accountant suma ambos: `total_spent_epsilon = ε_LD + ε_SVT`.

### Parámetros adicionales

| Parámetro | Descripción | Default |
|---|---|---|
| `sparse_eps` | Presupuesto SVT (independiente de `epsilon_token`) | 1.0 |
| `tau` | Umbral de acuerdo (default `m/2`) | `m/2` |
| `exclude_stopwords` | Activa exclusión de stopwords | False |
| `exclude_question_words` | Activa exclusión de tokens de la pregunta | False |
| `stopwords_path` | Ruta a `stopword_words.pt` | `data/stopword_words.pt` |
| `tmax_absolute` | Máximo de tokens por respuesta | 200 |

### Salida adicional

- `dp_flags_per_token`: por token, `True` = LD privado, `False` = SVT público,
  `None` = excluido.
- `n_private_tokens`, `n_nonprivate_tokens`, `n_excluded_tokens`, `total_tokens`.

> Nota sobre `sparse_eps`. El paper (Algorithm 2) divide `ε_token` en
> `(ε_token/2, ε_token/2)` para LD y Laplace. El repo oficial los expone como
> dos parámetros independientes (`--dp_eps`, `--sparse_eps`) y esta
> implementación replica esa convención. Para reproducir el paper estrictamente,
> fijar `sparse_eps = epsilon_token / 2`.

## Hiperparámetros recomendados

| Régimen | m | k | epsilon_token | sparse_eps | τ | epsilon_total |
|---|---|---|---|---|---|---|
| Quick (smoke) | 10 | 1 | 2 | 1 | m/2 = 5 | 10 |
| Paper (NQ) | 30 | 1 | 2 (Vote) / 1 (Sparse) | 1 | m/2 = 15 | 10–40 |
| Alta utilidad | 50 | 1 | 5 | 1 | m/2 = 25 | ≥40 |

Los YAMLs viven en [`experiments/configs/`](../../experiments/configs/):
`dpvoterag.yaml`, `dpsparsevoterag.yaml` y sus contrapartes `_quick`.
