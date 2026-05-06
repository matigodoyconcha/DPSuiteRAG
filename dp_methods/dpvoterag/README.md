# DPVoteRAG

Implementación del método DPVoteRAG basado en Koga et al. (*"Privacy-Preserving Retrieval-Augmented Generation with Differential Privacy"*, arXiv:2412.04697v3).

## Idea General

DPVoteRAG usa votación entre `m` votantes para cada paso de generación token a token.
Cada votante observa un subconjunto de `k` documentos recuperados y propone un siguiente token.
Luego se agrega un histograma de votos y se selecciona el token final con el mecanismo
**Limited Domain (LD) Gumbel** (Durfee & Rogers, NeurIPS 2019), que es formalmente (ε,δ)-DP.

## Mecanismo Limited Domain (LD-Gumbel) — Durfee & Rogers 2019

El núcleo privado de DPVoteRAG es la selección de un token del histograma de votos mediante:

1. **Reducción de dominio**: Se toman solo los `k_bar` tokens con mayor conteo (por defecto `k_bar=m`).
2. **Cálculo de umbral privado**: Se computa
   $$h_{\perp} = h_{k_{\text{bar}}} + 1 + \frac{\log(\min(\text{sens}, k_{\text{bar}}, \text{dim} - k_{\text{bar}}) / \delta)}{\varepsilon}$$
   donde `h_{k_bar}` es el conteo del `k_bar`-ésimo token (umbral del candidato "perpendicular").
3. **Ruido Gumbel**: Se añade ruido Gumbel con escala `sens/ε` tanto a los conteos de los top-`k_bar` como al umbral:
   $$v_{\text{cnts}} = h_{\text{top-k_bar}} + \text{Gumbel}(0, \text{sens}/\varepsilon)$$
   $$v_{\perp} = h_{\perp} + \text{Gumbel}(0, \text{sens}/\varepsilon)$$
4. **Comparación formal**: Si $\max(v_{\text{cnts}}) \geq v_{\perp}$, se selecciona el token ganador. En caso contrario, se dispara un **fail-mode** (ver abajo).

**Sensibilidad**: Por defecto `sens=2` (l₀-neighboring con inserción/eliminación), puede ser ajustado.

**Privacy guarantee**: Cada llamada al mecanismo LD consume `ε` y `δ`. La composición RDP se maneja vía `autodp` (ver `metrics/privacy_accounting_ld.py`).

## Fail Modes

Cuando el mecanismo LD falla (i.e., `max(v_cnts) < v_perp`), se aplica una estrategia configurable:

- **"ld_pate"** (default): Reintenta con `k_bar` reducido (PATE-style retry), y si todo falla, devuelve un token aleatorio.
- **"rand"**: Devuelve un token aleatorio del vocabulario.
- **"stop"**: Detiene la generación y retorna `exit_status=1`.
- **"raise"**: Propaga la excepción para debugging.

Cada fallo se registra en `last_run_stats` y puede consultarse en la salida de experimentos.

## Presupuesto y Accounting Online

DPVoteRAG mantiene un **accountant online** (`LDGumbelAccountant`) que:

1. Calcula el presupuesto RDP disponible usando `autodp` (Theorem 19 para LD, composición, amplificación por subsampling).
2. **Antes de cada token**, verifica que queda presupuesto (`check_budget()`).
3. **Después de seleccionar**, contabiliza el gasto (`account_token()`).
4. Si se agota el presupuesto (`target_eps` o `target_delta`), detiene la generación con `exit_status=2`.

El gasto exacto (`eps_spent`, `delta_spent`) se persiste en la salida JSON por query y agregado.

## Parámetros

- `m`: número de voters. Valores altos tienden a mejorar robustez del voto, pero aumentan costo.
- `k`: documentos por voter. El paper usa principalmente `k=1`.
- `epsilon_token`: presupuesto LD-Gumbel **por token**. Parámetro crítico para privacidad/utilidad.
- `delta_token`: delta asociado a `epsilon_token` (típicamente 1e-5 o 1e-6).
- `epsilon_total`: presupuesto total **por query** (usado solo para logging; el control real es online vía accountant).
- `delta_total`: delta total por query.
- `target_eps`, `target_delta`: umbrales de parada anticipada; al agotarse, `exit_status=2`.
- `k_bar`: tamaño del dominio reducido en LD. Si no se fija, se usa `m`.
- `fail_mode`: estrategia ante fallos LD (por defecto "ld_pate").
- `random_seed`: controla particionado aleatorio y ruido para reproducibilidad.

## Costo Computacional

Cada query requiere aproximadamente `m` inferencias del modelo por token generado.
Por eso DPVoteRAG puede ser mucho más costoso que RAG vanilla, especialmente con `m` grande
y respuestas largas.

Con **batching HF**, se reducen las llamadas al modelo agrupando los `m` prompts de voters
en una sola inferencia por token (si la capacidad de memoria lo permite).

## DPSparseVoteRAG (Sparse Vector Technique)

DPSparseVoteRAG extiende DPVoteRAG con un **Sparse Vector Technique (SVT)** formal en cada paso:

1. **Genera un token público** `y_nonRAG` usando prompt sin contexto (no cuesta presupuesto privado).
2. **Consulta SVT**: Consulta su conteo en el histograma de voters privados con ruido Laplaciano (`Lap(0, 4/sparse_eps)`).
3. **Umbral dinámico**: Compara contra un umbral ruidoso `τ̂ = τ + Lap(0, 2/sparse_eps)`.
4. **Decisión**:
   - Si conteo_ruidoso ≤ `τ̂`: usa el token público **sin gastar presupuesto LD** (fila pública).
   - Si conteo_ruidoso > `τ̂`: aplica LD-Gumbel (fila privada, gasta `epsilon_token`).

### Presupuestos separados

- `sparse_eps`: presupuesto del **SVT** (independiente de `epsilon_token`). Controla el ruido Laplaciano.
- `epsilon_token`: presupuesto del **LD-Gumbel** (por cada voto privado).

**Composición**: El budget total por respuesta es aproximadamente `T·sparse_eps + n_dp·epsilon_token`, donde `T` es la longitud y `n_dp` es el número de tokens que pasaron el SVT.

### Exclusión de tokens triviales

Si el token público `y_nonRAG` está en una lista de **exclusión** (stopwords, tokens de la pregunta, etc.), se usa directamente sin SVT ni DP-vote, registrando flag `None` en `dp_flags_per_token`.

### Tracking per-token

`dp_flags_per_token` registra por cada token:
- `True`: seleccionado por voto privado LD (consumió `epsilon_token`).
- `False`: seleccionado por SVT público (consumió ~`4/sparse_eps`).
- `None`: excluido por lista de exclusión (sin costo privado).

## Hyperparámetros recomendados

| Régimen | m | k | epsilon_token | sparse_eps | threshold (τ) | target_eps |
| --- | --- | --- | --- | --- | --- | --- |
| Piloto rápido | 10 | 1 | 2 | 1 | m/2 = 5 | 10 |
| Balanceado | 30 | 1 | 2 | 1 | m/2 = 15 | 20 |
| Alta utilidad | 40–50 | 1 | 5 | 1 | m/2 = 20–25 | 40 |

Notas prácticas:

- `τ = m/2` funciona como punto inicial robusto para SVT.
- Si `tokens_public` (fila pública SVT) es muy alto, bajar `τ` o subir `sparse_eps` para más privacidad.
- Si hay demasiada varianza, subir `epsilon_token` (pero ojo al presupuesto total).
- Los parámetros se exponen en `experiments/configs/dpvoterag.yaml` y `dpsparsevoterag.yaml`.

## Salida y métricas

Cada experimento genera un JSON con:

- `results`: lista de queries con campos:
  - `predictions`: token IDs generados.
  - `ground_truth`: referencia.
  - `exit_status`: 0=OK, 1=detenido por fail-mode "stop", 2=presupuesto agotado, 3=longitud máxima alcanzada.
  - `dp_spent`: presupuesto real gastado (solo DPVoteRAG por ahora).
  - `dp_flags_per_token`: flags de decisión (solo DPSparseVoteRAG).
- `payload`:
  - `dp_spent`: agregado (suma y conteo por query).
  - `exit_statuses`: lista de todos los exit_status.
  - Meta: configuración, hiperparámetros, tiempo de ejecución.

## Archivos clave

- `generator.py`: Implementación de `VoteRAGGenerator` con mecanismo LD-Gumbel y accountant.
- `sparse_vote_generator.py`: Extensión `SparseVoteRAGGenerator` con SVT y exclusión.
- `pipeline.py`: Orquestación RAG (retriever, embedder, indexer).
- `../metrics/privacy_accounting_ld.py`: Accounting RDP y conversión a (ε,δ)-DP.
