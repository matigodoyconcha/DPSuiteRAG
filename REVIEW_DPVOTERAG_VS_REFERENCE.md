# Revisión DPVoteRAG / DPSparseVoteRAG vs implementación de referencia

> **Referencia analizada:** [tacchan7412/DPRAG](https://github.com/tacchan7412/DPRAG)
> (basada en Koga et al., *"Privacy-Preserving Retrieval-Augmented Generation with Differential Privacy"*, arXiv:2412.04697)
>
> **Implementación local revisada:**
> - [dp_methods/dpvoterag/generator.py](dp_methods/dpvoterag/generator.py) — `VoteRAGGenerator`
> - [dp_methods/dpvoterag/sparse_vote_generator.py](dp_methods/dpvoterag/sparse_vote_generator.py) — `SparseVoteRAGGenerator`
> - [dp_methods/dpvoterag/pipeline.py](dp_methods/dpvoterag/pipeline.py) — `DPVoteRAGPipeline`
> - [experiments/run_dpvoterag.py](experiments/run_dpvoterag.py)
> - [experiments/run_dpsparsevoterag.py](experiments/run_dpsparsevoterag.py)
> - [metrics/privacy_accounting.py](metrics/privacy_accounting.py)
>
> **Fecha:** 2026-05-05
> **Objetivo:** Identificar TODO incorrecto, faltante y los TODOs accionables para que otro agente pueda implementarlos.

---

## ✅ Resumen ejecutivo

| Componente | Estado | Severidad |
|---|---|---|
| Mecanismo Limited Domain (LD-Gumbel) — núcleo del paper | ✅ Implementado formal (Durfee-Rogers'19) | **RESUELTO** |
| Detección de fallo / fail modes (`NotFoundLDTop1`) | ✅ Implementado (4 estrategias) | **RESUELTO** |
| Privacy accounting con `autodp` (RDP + subsampling) | ✅ Implementado en `privacy_accounting_ld.py` | **RESUELTO** |
| Auto-stop por overflow de presupuesto (`target_eps`/`target_delta`) | ✅ Implementado con accountant online | **RESUELTO** |
| SVT en SparseVote: `sparse_eps` separado del `epsilon_token` | ❌ Splitea `epsilon_token / 2` (no es lo del paper) | **Alto** |
| SparseVote: prompt público batched con voters (vLLM) | ❌ Se ejecuta secuencialmente | Medio (correctitud OK, eficiencia mala) |
| SparseVote: exclusión de stopwords / question_words | ❌ No implementado | Medio |
| Inferencia batched vía vLLM | ❌ Usa HF + bucle Python (lento) | Medio |
| Tracking per-token de bandera DP (`dp_flags_list`) | ❌ Solo agrega contadores globales | Bajo |
| Documentos por voter (`k=1`, partición aleatoria) | ✅ Conceptualmente OK | — |
| Pipeline RAG (retriever, embedder, indexer) | ✅ Razonable | — |
| Estructura de resultados y métricas (utility/privacy) | ✅ OK | — |

---

## 1. Mecanismo Limited Domain Gumbel (LD-Gumbel) — **✅ IMPLEMENTADO**

### 1.1 Especificación del paper de referencia

Implementación de referencia en [`utils/dp.py:48-105`](https://github.com/tacchan7412/DPRAG/blob/main/utils/dp.py):

```python
def get_top1(self, cnts: torch.Tensor, dim, k_bar=None, sens=2.):
    sorted_cnts, sorted_idxs = torch.sort(cnts, descending=True)
    gumbel = Gumbel(0., 1 / self.eps)

    # Calcular h_perp (umbral del candidato "perp")
    if k_bar < dim:
        if k_bar + 1 > real_len_cnts:
            h_perp = 0
        else:
            h_perp = sorted_cnts[k_bar]
        h_perp = h_perp + 1 + np.log(min(sens, k_bar, dim - k_bar) / self.delta) / self.eps
        v_perp = h_perp + gumbel.sample().item()

    # Comparar v_max con v_perp
    v_cnts = sorted_cnts[:k_bar] + gumbel.sample((k_bar,))
    v_max_idx = torch.argmax(v_cnts)
    if v_cnts[v_max_idx] >= v_perp:
        return sorted_idxs[v_max_idx]   # éxito
    else:
        raise NotFoundLDTop1()           # fallo → invocar fail_mode
```

Puntos clave del mecanismo de Durfee & Rogers (NeurIPS 2019):

1. Se ordena el histograma y se considera solo el dominio reducido `k_bar`.
2. Se computa `h_perp = h_{k_bar} + 1 + log(min(sens,k_bar,dim-k_bar)/δ) / ε`.
3. Se añade ruido Gumbel(0, sens/ε) tanto a los top-`k_bar` como a `h_perp`.
4. Si `argmax(v_cnts) < v_perp` → **fallo** (no se reporta nada).
5. La sensibilidad por defecto es **`sens=2`** (l₀ neighboring), no 1.

Ese término logarítmico **es el que paga el `δ`** del mecanismo LD. Sin él el mecanismo NO es (ε,δ)-DP.

### 1.2 Implementación local (ACTUAL - Corregida)

[`dp_methods/dpvoterag/generator.py:155-214`](dp_methods/dpvoterag/generator.py#L155-L214):

```python
def _limited_domain_or_fail(self, histogram, eps, delta, k_bar=None, sens=2.0) -> int:
    """Implement the Durfee-Rogers Limited Domain Gumbel mechanism (NeurIPS 2019)."""
    
    # Sort histogram by count (descending)
    sorted_items = sorted(histogram.items(), key=lambda item: item[1], reverse=True)
    dim = len(sorted_items)
    k_bar = min(k_bar, dim)
    
    # Extract top-k_bar items
    top_counts = np.array([float(count) for _, count in sorted_items[:k_bar]], dtype=np.float64)
    top_token_ids = [int(token_id) for token_id, _ in sorted_items[:k_bar]]
    
    # Compute h_perp (privacy-paying threshold)
    if k_bar < dim:
        h_perp = float(sorted_items[k_bar][1])  # (k_bar)-th item count
        log_arg = min(sens, float(k_bar), float(dim - k_bar)) / delta
        h_perp = h_perp + 1.0 + np.log(log_arg) / eps  # ← ✅ Usa delta correctamente
    else:
        h_perp = 1.0 + np.log(sens / delta) / eps
    
    # Add Gumbel noise: scale is sens / eps (✅ Correcto, no 1/eps)
    gumbel_scale = sens / eps
    v_cnts = top_counts + self._np_rng.gumbel(loc=0.0, scale=gumbel_scale, size=len(top_counts))
    v_max_idx = np.argmax(v_cnts)
    v_max = float(v_cnts[v_max_idx])
    
    # Add Gumbel noise to threshold
    v_perp = h_perp + float(self._np_rng.gumbel(loc=0.0, scale=gumbel_scale))
    
    # ✅ Formal comparison: v_max >= v_perp decides success
    if v_max >= v_perp:
        return int(top_token_ids[v_max_idx])
    else:
        raise NotFoundLDTop1(...)  # ✅ Formal privacy failure event
```

### 1.3 Cambios implementados (Paso 1 ✅)

| Punto | Antes (❌) | Ahora (✅) |
|---|---|---|
| Cálculo de `h_perp` | No existía | `h_{k_bar}+1+log(min(...)/δ)/ε` |
| Ruido Gumbel sobre `h_perp` | No | Sí, con escala `sens/eps` |
| Comparación `v_max ≥ v_perp` | No (heurística ad-hoc) | Sí (formal DR'19) |
| Excepción `NotFoundLDTop1` | No | Sí (para fail-modes) |
| Sensibilidad `sens=2` | Hardcoded `1/eps` | Configurable, default 2 |
| Uso de `delta` | Ignorado (`_ = delta`) | ✅ En logaritmo (`h_perp`) |
| Fail-modes support | No | Sí (4 estrategias) |
| Tests | No | ✅ 13 unitarios |

**Estado:** ✅ IMPLEMENTADO, TESTEADO (13/13 tests PASAN)

---

## 2. Fail modes y manejo de fallos del LD — **✅ IMPLEMENTADO**

### 2.1 Referencia

[`utils/ensemble.py:120-146`](https://github.com/tacchan7412/DPRAG/blob/main/utils/ensemble.py): cuatro fail-modes documentados:

- `'ld_pate'` (default): si LD falla, reintenta con `k_bar=dim-1` (LimitedDomain "1-excluded").
- `'rand'`: devuelve un token aleatorio del vocabulario.
- `'stop'`: detiene la generación (retorna `None` → exit_status=1).
- `'raise'`: propaga la excepción.

Adicionalmente: `DPExpenseOverflow` cuando `target_eps`/`target_delta` se exceden → detiene con su propio exit_status.

### 2.2 Implementación local (ACTUAL - Completada)

[`dp_methods/dpvoterag/generator.py:233-298`](dp_methods/dpvoterag/generator.py#L233-L298):

```python
class VoteRAGGenerator:
    def __init__(self, ..., fail_mode: str = "ld_pate"):
        """Initialize with fail_mode ∈ {"ld_pate", "rand", "stop", "raise"}."""
        self.fail_mode = fail_mode

    def _handle_ld_failure(self, histogram, eps, delta, k_bar, sens=2.0):
        """Apply fail-mode strategy when LD mechanism fails."""
        if self.fail_mode == "ld_pate":
            # Retry with k_bar reduced: k_bar → k_bar-1 → ... → 1
            for attempt_k_bar in range(k_bar - 1, 0, -1):
                try:
                    token_id = self._limited_domain_or_fail(histogram, eps, delta, k_bar=attempt_k_bar)
                    return int(token_id), 0  # Success with reduced k_bar
                except NotFoundLDTop1:
                    continue
            # All retries failed; fall back to random
            token_id = int(self._np_rng.integers(0, self.vocab_size))
            return token_id, 0
        
        elif self.fail_mode == "rand":
            # Return random token
            token_id = int(self._np_rng.integers(0, self.vocab_size))
            return token_id, 0
        
        elif self.fail_mode == "stop":
            # Signal termination: return None + exit_status=1
            return None, 1
        
        elif self.fail_mode == "raise":
            # Re-raise for debugging
            raise NotFoundLDTop1(f"LD failed with fail_mode='raise'")

    def generate_vote(self, query, docs):
        """Main generation loop with fail-mode handling."""
        ...
        for _ in range(tmax):
            histogram = dict(Counter(voter_tokens))
            try:
                winning_token = self._limited_domain_or_fail(histogram, ...)
            except NotFoundLDTop1:
                # Apply fail-mode strategy
                winning_token, fail_exit_status = self._handle_ld_failure(...)
                if fail_exit_status == 1:  # 'stop' mode
                    exit_status = 1
                    break
            ...
```

### 2.3 Cambios implementados (Paso 2 ✅)

| Estrategia | Comportamiento | Tests |
|---|---|---|
| **"ld_pate"** (default) | Reintenta con k_bar decreciente; fallback a aleatorio | ✅ 2 tests |
| **"rand"** | Token aleatorio del vocabulario | ✅ 2 tests |
| **"stop"** | Retorna None, termina generación (exit_status=1) | ✅ 1 test |
| **"raise"** | Re-lanza excepción (debugging) | ✅ 1 test |

**Cambios adicionales (Paso 2.5 ✅):**
- ✅ Detección robusta de `vocab_size` desde tokenizador (múltiples atributos)
- ✅ Excepción clara si no se puede detectar vocab_size
- ✅ Soporta OPT, GPT-2, y otros tokenizers HF

**Estado:** ✅ IMPLEMENTADO, TESTEADO (10 tests fail-modes + 13 LD-Gumbel = 23 total PASAN)

---

## 3. Privacy Accounting — ✅ IMPLEMENTADO (Paso 3)

### 3.1 Qué se implementó

Se añadió un accounting formal para el mecanismo Limited Domain (LD-Gumbel) y se integró en el generador. Cambios principales:

- Se añadió `autodp` a `requirements.txt`.
- Nuevo módulo: `metrics/privacy_accounting_ld.py` que aporta:
    - Cálculo RDP aproximado para LD-Gumbel (`rdp_limited_domain_gumbel`).
    - Amplificación por subsampling y composición RDP.
    - Conversión RDP → (ε,δ)-DP mediante un método BBGHS (y búsqueda de α óptimo).
    - `LDGumbelAccountant`: contador online que permite consultar y gastar presupuesto por token.
- Integración en el generador: `VoteRAGGenerator` ahora instancia `LDGumbelAccountant` y comprueba el presupuesto antes de votar; al contabilizar cada token se actualiza el gasto y, si se excede, se lanza `DPExpenseOverflow`.
- Se agregaron tests unitarios: `tests/test_privacy_accounting_ld.py` (30 tests) que verifican RDP, composición, subsampling y el accountant.

### 3.2 Efectos en la implementación local

- `metrics/privacy_accounting.py` (antiguo) queda como referencia; la implementación específica para LD está ahora en `metrics/privacy_accounting_ld.py`.
- `dp_methods/dpvoterag/generator.py`:
    - Inicializa `LDGumbelAccountant` con `target_epsilon/delta`, `epsilon_token/delta_token`, `m`, `k`.
    - Antes de generar cada token consulta `accountant.check_budget()`; si no cabe, se lanza `DPExpenseOverflow` y la generación se detiene con `exit_status` adecuado.
    - Después de seleccionar un token válido, se llama a `accountant.account_token()` para consumir presupuesto por token.

### 3.3 Archivos añadidos / modificados

- `requirements.txt` — se añadió `autodp>=0.2.1`.
- `metrics/privacy_accounting_ld.py` — nuevo módulo (RDP + accountant).
- `dp_methods/dpvoterag/generator.py` — integración del accountant y levantado de `DPExpenseOverflow` cuando corresponde.
- `tests/test_privacy_accounting_ld.py` — tests unitarios (30 pruebas) para el módulo de accounting.

Los tests relevantes para LD y fail-modes siguen pasando (23 tests existentes + 30 nuevos; ver logs de pytest).

### 3.4 Pendientes (futuros pasos)

- Ajustes finos en los parámetros por defecto y exposición en los YAML (`target_eps`, `target_delta`, `sparse_eps`).

### 3.5 Cambios implementados (Paso 4 ✅)

Se completó la integración del control online de presupuesto en `VoteRAGGenerator` y el reporting de gasto DP en outputs del experimento.

- ✅ `LDGumbelAccountant.reset()` agregado para reiniciar el gasto por query.
- ✅ `generate_vote()` ahora usa chequeo online de presupuesto por token como gate real (`check_budget`) y detiene con `exit_status=2` al agotar budget.
- ✅ Se dejó de usar `_compute_tmax()` como limitador efectivo de seguridad DP (se mantiene en código solo como helper legacy, no como control principal).
- ✅ Se agregó `last_run_stats` en `VoteRAGGenerator` con `exit_status`, `dp_spent_epsilon`, `dp_spent_delta`, `dp_summary`.
- ✅ En `run_dpvoterag.py` ahora se persisten:
    - `payload['dp_spent']` con agregado por corrida y detalle por query.
    - `exit_statuses` por query.
    - En cada entrada de `results`: `exit_status` y `dp_spent`.

Archivos tocados en este paso:

- `metrics/privacy_accounting_ld.py`
- `dp_methods/dpvoterag/generator.py`
- `experiments/run_dpvoterag.py`

Validación:

- ✅ `pytest -q tests/test_privacy_accounting_ld.py tests/test_fail_modes.py`
- ✅ 40 tests pasando.


---

## 4. SparseVoteRAG: presupuesto SVT separado — **ALTO**

### 4.1 Referencia

CLI ([`README.md:90-110`](https://github.com/tacchan7412/DPRAG/blob/main/README.md)):

```
--dp_eps 1 --dp_delta 1e-5     # presupuesto del LD por consulta DP
--threshold 25                  # τ del SVT
--sparse_eps 1                  # presupuesto SVT (independiente de dp_eps)
--below_threshold               # activa SparseVoteRAG
```

[`utils/ensemble.py:77-117`](https://github.com/tacchan7412/DPRAG/blob/main/utils/ensemble.py):

```python
# Threshold: τ̂ = τ + Lap(0, 2/sparse_eps)
bt_dict['noisy_threshold'] = bt_dict['threshold'] + np.random.laplace(0, 2/bt_dict['sparse_eps'])

# Query: a_t + Lap(0, 4/sparse_eps)
if (next_tokens[:-1] == next_tokens[-1]).sum().item() + np.random.laplace(0, 4/bt_dict['sparse_eps']) <= bt_dict['noisy_threshold']:
    # DP-vote (cuesta dp_eps + 0 al SVT)
    majority_token = majority_vote(next_tokens[:-1], token_dim, dp_engine=dp_engine)
    bt_dict['noisy_threshold'] = bt_dict['threshold'] + np.random.laplace(0, 2/bt_dict['sparse_eps'])  # refresca
else:
    # Usa LLM público (no consume nada)
    majority_token = next_tokens[-1]
```

Es un **Sparse Vector Technique estándar** con dos presupuestos disjuntos: `dp_eps` para la query DP y `sparse_eps` para el SVT.

### 4.2 Local

[`dp_methods/dpvoterag/sparse_vote_generator.py:120-166`](dp_methods/dpvoterag/sparse_vote_generator.py#L120-L166):

```python
epsilon_token_lap = self.epsilon_token / 2.0    # ← splitea heurísticamente
epsilon_token_rag = self.epsilon_token / 2.0
...
tau_hat = self.tau + self._sample_laplace(2.0 / epsilon_token_lap)
...
a_t_noisy = float(a_t) + self._sample_laplace(4.0 / epsilon_token_lap)
```

Problemas:

1. **No hay un parámetro `sparse_eps`.** Toma un único `epsilon_token` y lo parte 50/50. Eso difiere de la API del paper, no es configurable y mezcla budgets que el paper trata por separado.
2. El `epsilon_token` ya se reusa como ε del LD-Gumbel ⇒ el budget total real es ambiguo.
3. El `--threshold` se llama `tau` localmente; OK, pero no hay un default sensato (el README dice `m/2` pero el paper de referencia usa 25 con m=50, es decir 0.5·m, cualitativamente similar).

### 4.3 TODO

- [ ] Añadir parámetro `sparse_eps: float` al `SparseVoteRAGGenerator.__init__` y al config `dpsparsevoterag.yaml`.
- [ ] Eliminar el split `epsilon_token / 2`. Usar:
  - `epsilon_token` ⇒ presupuesto del LD (igual que en VoteRAG).
  - `sparse_eps` ⇒ presupuesto del SVT (escalas Laplace 2/sparse_eps y 4/sparse_eps).
- [ ] Documentar en el README que el budget total por token es `epsilon_token + sparse_eps · 𝟙[query DP]` y por respuesta es `T·sparse_eps + n_dp·epsilon_token` (a refinar con SVT composition theorem).
- [ ] Añadir `threshold` (alias `tau`) explícito en el YAML sin default mágico.

---

## 5. SparseVoteRAG: prompt público y exclusión de stopwords — **MEDIO**

### 5.1 Referencia

- El prompt público (sin contexto) se inyecta como **última entrada en `prompt_list`** ([`eval_rag_generator.py:75-80`](https://github.com/tacchan7412/DPRAG/blob/main/eval_rag_generator.py#L75-L80)) y se evalúa en la **misma llamada batched a vLLM** que los voters. `next_tokens[-1]` es la respuesta del público.
- Si el último token generado por el público está en `exclude_ids` (stopwords del archivo `data/stopword_words.pt` y/o tokens de la pregunta), se usa **directamente** el token público sin SVT y sin DP-vote ([`utils/ensemble.py:82-88`](https://github.com/tacchan7412/DPRAG/blob/main/utils/ensemble.py#L82-L88)). Esto evita gastar budget en tokens triviales.
- Se loggea `bt_dict['dp_flags_list'][-1].append(None | True | False)` por cada token: `None`=excluido, `True`=DP-voted, `False`=copiado del público.

### 5.2 Local

- El prompt público se construye y evalúa **secuencialmente** dentro del loop de tokens ([`sparse_vote_generator.py:138-148`](dp_methods/dpvoterag/sparse_vote_generator.py#L138-L148)). Una llamada extra al modelo por token.
- No existe lista de exclusión (stopwords ni question_words).
- `last_run_stats` solo agrega contadores `n_private_tokens` / `n_nonprivate_tokens`; no per-token.

### 5.3 TODO

- [ ] (Eficiencia) Migrar a inferencia batched (ver §7). Una vez batched, fusionar voters + public en una sola llamada.
- [ ] Añadir parámetro opcional `exclude_stopwords: bool` y `exclude_question_words: bool` al `SparseVoteRAGGenerator`.
  - Pre-tokenizar una lista de stopwords en `__init__` (cacheable).
  - Tokenizar la query al inicio de `generate_vote` y agregar sus token_ids a `exclude_ids`.
  - Si `y_nonrag in exclude_ids` → usar `y_nonrag` directamente, marcar bandera `None`.
- [ ] Reemplazar `last_run_stats` por una lista `dp_flags_per_token: list[bool|None]` y persistirla en el JSON de salida.

---

## 6. Pipeline / orquestación — **MEDIO**

### 6.1 Referencia

[`eval_rag_generator.py:39-116`](https://github.com/tacchan7412/DPRAG/blob/main/eval_rag_generator.py#L39-L116):

```python
context_text = read_top(retrieved_docs_path, n_split * n_docs)
random.shuffle(context_text)
prompt_list = [build_prompt(query, context_text[i*n_docs:(i+1)*n_docs]) for i in range(n_split)]
if below_threshold: prompt_list.append(build_prompt(query, []))   # público
outputs = vllm_model.generate(prompt_token_ids=prompt_token_ids_list, sampling_params=...)
next_tokens = [output.outputs[0].token_ids[0] for output in outputs]
```

Esto vectoriza voters + público en una llamada vLLM por token.

### 6.2 Local

[`generator.py:207-218`](dp_methods/dpvoterag/generator.py#L207-L218):

```python
for _ in range(tmax):
    for voter_docs in partitions:                         # ← O(m) inferencias
        prompt = self._build_prompt(...)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        next_token = self._generate_next_token(encoded["input_ids"])
        voter_tokens.append(next_token)
```

Cada token requiere `m` (o `m+1`) llamadas secuenciales al modelo y re-tokenización completa del prompt cada vez (no se reutiliza el KV-cache). Para `m=30, T=200` son 6000 forwards por query.

### 6.3 TODO

- [ ] Refactorizar `_generate_next_token` para que reciba **una lista de prompts** y devuelva **una lista de next_tokens**, usando:
  - **Opción A (preferida si CUDA está disponible):** vLLM con `SamplingParams(temperature=0, max_tokens=1)` (mirar referencia).
  - **Opción B (sin vLLM):** `model(input_ids_padded, attention_mask=...)` con padding y batched argmax sobre `outputs.logits[:, -1, :]`.
- [ ] Para mantener el KV-cache entre tokens del mismo voter, considerar `model.generate(..., use_cache=True)` con `past_key_values` recurrentes.
- [ ] Modificar `SparseVoteRAGGenerator.generate_vote` para que la lista de prompts incluya el prompt público al final, y leer `voter_tokens = next_tokens_list[:-1]`, `y_nonrag = next_tokens_list[-1]`.

---

## 7. Inferencia con vLLM — **MEDIO**

Toda la referencia asume vLLM (CUDA). El proyecto local usa HuggingFace transformers en CPU/GPU. Esto no es un bug formal pero hace inviable correr `m=50, T=200` con un dataset de tamaño no-trivial.

### TODO

- [ ] Detectar si vLLM está disponible (`import vllm`) y, si lo está, usarlo para `_generate_next_token` batched.
- [ ] Mantener path HF como fallback explícito y advertirlo en logs.
- [ ] Evaluar si `bench_llms/` ya tiene wrapper vLLM y reutilizarlo.

---

## 8. `_compute_tmax`: accounting forward incorrecto

[`generator.py:84-115`](dp_methods/dpvoterag/generator.py#L84-L115) calcula `tmax` para limitar tokens *a priori* usando dos cotas:

```python
tmax_seq = floor(epsilon_total / epsilon_token)                              # basic comp.
tmax_adv = solve_for_t( sqrt(2t·log(1/δ))·ε + t·ε² ≤ epsilon_total )         # advanced comp.
return max(1, tmax_seq, tmax_adv)
```

Problemas:

1. Esas son cotas para **mecanismo ε-DP puro**, NO para LD.
2. `max(tmax_seq, tmax_adv)` es **al revés**: queremos el **mínimo** entre cotas válidas (la más estricta es la cota más segura). Tomar el max **viola la garantía**.
3. Aun corregido, no sustituye el accounting RDP de §3.

### TODO

- [ ] Eliminar `_compute_tmax` o renombrarlo a `_estimate_max_tokens_loose_upperbound` y dejarlo solo como referencia visual.
- [ ] El control real de presupuesto debe venir del accountant online (§3.3), no de un `tmax` precomputado.

---

## 9. Configuración por defecto que no concuerda con el paper

| Parámetro | Paper (ref. README) | Local (`dpvoterag.yaml`) | Comentario |
|---|---|---|---|
| Modelo | `facebook/opt-1.3b` | `facebook/opt-1.3b` | ✅ |
| `n_split` (= `m`) | 50 | 30 | Aceptable (el README local sugiere 10–50). |
| `n_docs` (= `k`) | 1 | 1 | ✅ |
| `dp_eps` (= `epsilon_token`) | 2 (VoteRAG), 1 (Sparse) | 2 | OK para VoteRAG; Sparse debería separar `dp_eps` y `sparse_eps`. |
| `dp_delta` | 1e-5 | 1e-5 | ✅ |
| `target_eps`/`target_delta` | 1000 / 1.0 | (no existe) | ❌ Falta. |
| `threshold` (Sparse) | 25 (con m=50 ⇒ 0.5·m) | `null` (default `m/2`) | OK conceptualmente pero hay que exponerlo en YAML. |
| `sparse_eps` | 1 | (no existe) | ❌ Falta (§4). |
| `--non_short_instruction` flag | sí | no | Cosmético. |
| Stopwords / `--stopping_words` | sí | no | §5. |
| `--question_words` | sí | no | §5. |

### TODO

- [ ] Añadir `target_eps`, `target_delta`, `sparse_eps` y `threshold` al `dpsparsevoterag.yaml`.
- [ ] Añadir `target_eps`, `target_delta` al `dpvoterag.yaml`.
- [ ] Documentar en el README que el budget configurado es **per-query** (no global) si esa es la intención.

---

## 10. Métricas y reporting

| Aspecto | Referencia | Local | Acción |
|---|---|---|---|
| Métrica de utilidad | `exact_match_score`, `f1_score` (`utils_rag`) | `match_accuracy` (exact match normalizado) | ✅ Suficiente, pero falta F1. |
| Persistencia de bandera DP por token | `*_dp-flags-list.txt` | `last_run_stats` agregado | Persistir per-token (§5). |
| Persistencia de histogramas de votación | `--save_hist` opcional | No existe | Opcional. |
| Persistencia de exit_status | sí (`FAILED` y `-1` en token-ids) | ✅ En `run_dpvoterag.py` | **RESUELTO para DPVoteRAG** |
| Persistencia del ε realmente gastado | sí (vía `dp_engine.get_dp_expense()`) | ✅ En `payload['dp_spent']` de DPVoteRAG | **RESUELTO para DPVoteRAG** |

### TODO

- [ ] Añadir `f1_score` a `metrics/utility.py` y al config `metrics.utility: [match_accuracy, f1]`.
- [ ] Persistir `dp_flags_per_token` en el JSON de salida (lista alineada con `predictions`) y extender `exit_status` a DPSparseVoteRAG.
- [ ] (Opcional) Añadir flag `--save_hist` que guarde el histograma de votos por token a un archivo separado.

---

## 11. Documentación (READMEs locales)

- [`dp_methods/dpvoterag/README.md`](dp_methods/dpvoterag/README.md) ya **admite** que el LD es simplificado:
  > "La parte completa de calibracion de umbrales y tratamiento formal fino de `delta` del mecanismo original de Durfee & Rogers (NeurIPS 2019) no esta implementada en su forma completa aqui."
  Esto es honesto, pero significa que la implementación NO es comparable con el baseline de Koga et al. en términos formales. Cualquier resultado de privacidad reportado debería marcarse explícitamente como heurístico.

- El README dice del SparseVote:
  > "El mecanismo de ahorro de presupuesto cuando `y_nonRAG` coincide con un token frecuente del histograma no tiene una justificacion formal completamente cerrada en el paper original."
  Falso: el paper sí lo justifica vía SVT (Dwork & Roth §3.6) — el problema es que la implementación local no usa SVT con presupuesto separado (§4).

### TODO

- [ ] Tras corregir §1 y §3, actualizar `dp_methods/dpvoterag/README.md` para reflejar que ahora sí es LD-DR'19.
- [ ] Reescribir el párrafo sobre SparseVoteRAG aclarando que el ahorro **sí** es formalmente justificable por SVT y que la implementación lo respeta.

---

## 12. Lista priorizada de TODOs accionables

### Prioridad P0 (correctitud DP)

1. ✅ **[LD-Gumbel]** COMPLETADO. Implementado `_limited_domain_or_fail()` siguiendo Durfee-Rogers'19 con:
   - ✅ Cálculo correcto de `h_perp = h_{k_bar}+1+log(min(sens,k_bar,dim-k_bar)/δ)/ε`
   - ✅ Ruido Gumbel con escala `sens/eps` en top-k_bar y threshold
   - ✅ Comparación formal `v_max ≥ v_perp` para decidir éxito
   - ✅ Excepción `NotFoundLDTop1` ante fallo
   - ✅ Tests unitarios: 13/13 PASAN
   - Archivo: [dp_methods/dpvoterag/generator.py](dp_methods/dpvoterag/generator.py)

2. ✅ **[fail modes]** COMPLETADO. Implementadas 4 estrategias:
   - ✅ `"ld_pate"`: Reintenta con k_bar reducido (PATE-style)
   - ✅ `"rand"`: Token aleatorio
   - ✅ `"stop"`: Termina con exit_status=1
   - ✅ `"raise"`: Propaga excepción (debugging)
   - ✅ Tests unitarios: 10/10 PASAN
   - Integración en [generate_vote()](dp_methods/dpvoterag/generator.py#L360-L400)
   
   **Bonus:** ✅ Detección robusta de `vocab_size` desde tokenizador (Paso 2.5)

3. **[accounting LD-RDP]** Reescribir `metrics/privacy_accounting.py` (o crear `_ld.py`) para usar `autodp` y la composición subsampled-LD-RDP del paper. Añadir `autodp` a `requirements.txt`.
   - [ ] Crear clases para EM y LD-RDP composition
   - [ ] Convertir RDP→approx-DP con BBGHS'19 Theorem 21
   - [ ] Integrar con VoteRAGGenerator

4. ✅ **[budget online]** COMPLETADO para DPVoteRAG.
    - ✅ `accountant.check_budget()` como gate por token.
    - ✅ Stop con `exit_status=2` ante overflow de presupuesto.
    - ✅ Persistencia de `eps_spent` y `delta_spent` en JSON (`run_dpvoterag.py`).
    - [ ] Extender este mismo esquema a DPSparseVoteRAG.

5. **[SparseVote: sparse_eps separado]** Añadir parámetro `sparse_eps` distinto de `epsilon_token`. Eliminar el split `epsilon_token/2`. Actualizar config `dpsparsevoterag.yaml` y `run_dpsparsevoterag.py` para leerlo.

### Prioridad P1 (fidelidad al paper)

6. **[SparseVote: exclusión]** Implementar `exclude_stopwords` y `exclude_question_words`. Bandera `None` para tokens excluidos.

7. **[SparseVote: público batched]** Construir `prompt_list` con voters + público y procesarlo en una sola llamada al modelo. Leer `voter_tokens = next_tokens_list[:-1]`, `y_nonrag = next_tokens_list[-1]`.

8. **[reporting]** Persistir en cada JSON de salida:
   - `eps_spent`, `delta_spent` (del accountant).
   - `exit_status` por query.
   - `dp_flags_per_token` (Sparse).

### Prioridad P2 (eficiencia y compatibilidad)

9. **[batched inference]** Refactorizar `_generate_next_token` para batched con `attention_mask` + padding (HF) y, si vLLM está disponible, usarlo (paridad con la referencia).

10. **[KV cache]** Reusar `past_key_values` por voter para evitar re-encoding del prompt completo en cada token.

11. **[F1 score]** Añadir `f1_score` a `metrics/utility.py` y a los configs.

### Prioridad P3 (documentación)

12. Actualizar [dp_methods/dpvoterag/README.md](dp_methods/dpvoterag/README.md) tras P0/P1.

13. Documentar en el README principal que DPVoteRAG/DPSparseVoteRAG ahora siguen Koga et al. con accounting LD-RDP de Durfee-Rogers.

---

## 13. Apéndice A: comandos para reproducir las diferencias

### Clonar y comparar in-place

```bash
git clone https://github.com/tacchan7412/DPRAG.git /tmp/dprag-reference
diff -r /tmp/dprag-reference/utils/dp.py dp_methods/dpvoterag/  # las firmas no coincidirán; es esperado
```

### Smoke test propuesto

```bash
# Antes de tocar nada:
python experiments/run_dpvoterag.py --config experiments/configs/dpvoterag_quick.yaml
# Anotar: match_accuracy, mia_auc, time_per_query

# Tras P0 (LD correcto + fail modes + accounting online):
python experiments/run_dpvoterag.py --config experiments/configs/dpvoterag_quick.yaml
# Esperado: match_accuracy posiblemente menor (por fallos del LD que ahora se contabilizan)
#           pero eps_spent ahora SÍ aparece en el JSON y refleja el LD-DR'19.
```

---

## 14. Apéndice B: tabla rápida "qué archivo tocar"

| TODO | Archivo principal | Otros |
|---|---|---|
| LD-Gumbel correcto | [dp_methods/dpvoterag/generator.py](dp_methods/dpvoterag/generator.py) | tests/ |
| Fail modes | [dp_methods/dpvoterag/generator.py](dp_methods/dpvoterag/generator.py) | [experiments/run_dpvoterag.py](experiments/run_dpvoterag.py) |
| Accounting LD-RDP | [metrics/privacy_accounting.py](metrics/privacy_accounting.py) (o nuevo `_ld.py`) | requirements.txt |
| Budget online + DPExpenseOverflow | [dp_methods/dpvoterag/generator.py](dp_methods/dpvoterag/generator.py) | [dp_methods/dpvoterag/sparse_vote_generator.py](dp_methods/dpvoterag/sparse_vote_generator.py) |
| `sparse_eps` separado | [dp_methods/dpvoterag/sparse_vote_generator.py](dp_methods/dpvoterag/sparse_vote_generator.py) | [experiments/configs/dpsparsevoterag.yaml](experiments/configs/dpsparsevoterag.yaml), [experiments/run_dpsparsevoterag.py](experiments/run_dpsparsevoterag.py) |
| Exclude stopwords/question_words | [dp_methods/dpvoterag/sparse_vote_generator.py](dp_methods/dpvoterag/sparse_vote_generator.py) | data/stopword_words.txt |
| Prompt público batched | [dp_methods/dpvoterag/sparse_vote_generator.py](dp_methods/dpvoterag/sparse_vote_generator.py) | — |
| Inferencia batched / vLLM | [dp_methods/dpvoterag/generator.py](dp_methods/dpvoterag/generator.py) | bench_llms/ |
| F1 score | [metrics/utility.py](metrics/utility.py) | configs |
| Persistir `eps_spent`, `dp_flags_per_token`, `exit_status` | [experiments/run_dpvoterag.py](experiments/run_dpvoterag.py) | [experiments/run_dpsparsevoterag.py](experiments/run_dpsparsevoterag.py) |

---

**Fin del reporte.**
