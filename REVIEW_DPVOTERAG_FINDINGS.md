# Auditoría DPVoteRAG / DPSparseVoteRAG vs `tacchan7412/DPRAG` y paper

> **Implementación local revisada:**
> - [dp_methods/dpvoterag/generator.py](dp_methods/dpvoterag/generator.py)
> - [dp_methods/dpvoterag/sparse_vote_generator.py](dp_methods/dpvoterag/sparse_vote_generator.py)
> - [dp_methods/dpvoterag/presets.py](dp_methods/dpvoterag/presets.py)
> - [metrics/privacy_accounting_ld.py](metrics/privacy_accounting_ld.py)
> - [metrics/privacy_accounting_autodp.py](metrics/privacy_accounting_autodp.py)
> - [experiments/run_dpvoterag.py](experiments/run_dpvoterag.py)
> - [experiments/run_dpsparsevoterag.py](experiments/run_dpsparsevoterag.py)
> - [experiments/configs/dpvoterag.yaml](experiments/configs/dpvoterag.yaml), [experiments/configs/dpsparsevoterag.yaml](experiments/configs/dpsparsevoterag.yaml) (+ `_local.yaml`)
>
> **Referencias:**
> - `tacchan7412/DPRAG` (clonado en `/tmp/dprag-ref/` durante la auditoría).
> - Koga et al., *"Privacy-Preserving Retrieval-Augmented Generation with Differential Privacy"*, [arXiv:2412.04697v1](https://arxiv.org/html/2412.04697v1).
>
> **Fecha:** 2026-05-08.
> **Alcance:** verificación cruzada de los pasos descritos en los documentos previos (`REPLICABILITY_ANALYSIS_DPRAG.md`, `REVIEW_DPVOTERAG_VS_REFERENCE.md`, `REVIEW_DPRAG_IMPLEMENTATION.md`) contra el código real, el repo de referencia y el paper. Esos MDs ya fueron eliminados.

---

## 0. TL;DR

| # | Severidad | Título | Estado real |
|---|---|---|---|
| **B1** | **P0 — bloqueante** | `run_dpsparsevoterag.py` falla con `TypeError` por `tmax_absolute` y `stopwords_path` duplicados en kwargs | 🔴 No corregido |
| **B2** | **P0 — funcional** | `SparseVoteRAGGenerator.generate_vote` ignora `prompt_construction='token_ids_append'` (siempre re-tokeniza) | 🔴 No corregido |
| **B3** | P1 — correctitud | `fail_mode='ld_pate'` diverge del referente (local hace retry progresivo + random; ref hace 1 retry y stop) | 🟡 Divergencia activa |
| **B4** | P1 — datos | `data/stopword_words.pt` no existe en el repo; con `preset: paper` cae a fallback `local_txt` con warning | 🟡 Falla silenciosa |
| **B5** | P1 — conceptual | El paper define `ε_token-RAG = ε_token/2`, `ε_token-Lap = ε_token/2`; tanto local como referente usan `sparse_eps` separado (deviación intencional **del paper**, no del referente) | 🟢 Match referente, ⚠️ paper |
| **B6** | P1 — privacidad | `exclude_question_words` tokeniza con `add_special_tokens=False`; el referente incluye `<bos>` (`tokenizer(question)['input_ids']`) → exclusiones distintas | 🟡 Divergencia activa |
| **B7** | P2 — accounting | `LDGumbelAccountant.simple_bound` (cota genérica BBGHS) sobreestima ε severamente; `em_peeling_autodp` re-implementa `autodp` en vez de usar la lib instalada | 🟡 Replica formula, no la lib |
| **B8** | P2 — accounting | `LDGumbelAccountant` recibe `subsampling_rate` pero **nunca lo usa** (solo `AutodpLDAccountant` lo usa) | 🟡 Parámetro fantasma |
| **B9** | P2 — observación | El claim "Issue 4 — `max(tmax_seq, tmax_adv)` viola DP" del análisis previo es **incorrecto**: el paper mismo usa `max` (sec. T_max) y ambas cotas son válidas, así que tomar el máximo es legítimo | ⚪ Aclaración |
| **B10** | P2 — paridad | `_generate_next_tokens_vllm` está stubbed (`raise RuntimeError`) — el path "vLLM" anunciado en pasos 11/12 nunca se ejecuta | 🟡 Documentación engañosa |
| **B11** | P2 — defaults | `dpsparsevoterag.yaml` usa `epsilon_token: 2`; el paper usa `ε_token=1` para Sparse | ⚪ Aclaración |
| **B12** | P2 — código muerto | `SVT` declara `sparse_account_svt: True` pero el código nunca contabiliza ε del SVT en el accountant | 🟡 No-op |
| **B13** | **P0 — privacidad** | `AutodpLDAccountant` pasaba `k=self.k_bar` (≡30) cuando el referente usa `k=1`; sobrestima ε ~25× y agota budget prematuramente | ✅ Corregido (parámetro `n_peels` configurable, default 1) |
| **B14** | P2 — observabilidad | SparseVote llamaba `_limited_domain` (que silencia `NotFoundLDTop1` con eos), perdiendo `exit_status=1` cuando LD falla | ✅ Corregido (try/except con `_limited_domain_or_fail` + `_handle_ld_failure`) |

🔴 = bloqueante / 🟡 = activo, no bloqueante / 🟢 = OK / ⚪ = aclaración

> Los pasos 1–12 reportados en `REVIEW_DPVOTERAG_VS_REFERENCE.md` (LD-Gumbel, fail-modes, RDP, SVT con `sparse_eps`, batching HF, KV-cache, F1, etc.) **sí están implementados**. Lo que se rompe es la integración: los presets/runners no son invocables sin error, y dos flags claves (`prompt_construction`, `sparse_account_svt`) no llegan a su efecto.

---

## 1. Bugs bloqueantes (P0)

### B1. `run_dpsparsevoterag.py` no es ejecutable con ningún preset

**Síntoma.** Al lanzar `python experiments/run_dpsparsevoterag.py --config experiments/configs/dpsparsevoterag.yaml` (preset paper) o `..._local.yaml` (preset local) Python lanza:

```
TypeError: SparseVoteRAGGenerator.__init__() got multiple values for keyword argument 'tmax_absolute'
```

**Causa.** [experiments/run_dpsparsevoterag.py:256-274](experiments/run_dpsparsevoterag.py#L256-L274) pasa explícitamente `tmax_absolute=tmax_absolute` y `stopwords_path=stopwords_path`, y simultáneamente expande `**dp_flags`. El dict `dp_flags` (resultado de `resolve_dp_config`) **incluye ambas llaves** porque están en `PRESETS["paper"]` y `PRESETS["local"]` (ver `dp_methods/dpvoterag/presets.py:36, 42, 57, 61`). Verificado en runtime:

```python
>>> dp_flags = resolve_dp_config({"preset": "paper"})
>>> "tmax_absolute" in dp_flags, "stopwords_path" in dp_flags
(True, True)
>>> SparseVoteRAGGenerator(model=…, tmax_absolute=200, stopwords_path="data/stopword_words.txt", **dp_flags)
TypeError: got multiple values for keyword argument 'tmax_absolute'
```

**Efecto.** El runner Sparse **no se puede correr en absoluto**. No existe test de integración que lo ejercite (ningún archivo en `tests/` invoca `run_dpsparsevoterag.py` ni unpacks de presets contra el constructor).

**Por qué `run_dpvoterag.py` no falla.** Usa `resolve_dp_config_for_voterag` que strippea `_SPARSE_ONLY_KEYS = {"sparse_use_online_budget", "sparse_account_svt", "stopwords_format", "stopwords_path", "stopwords_token_index"}`. Eso elimina `stopwords_path` pero **no `tmax_absolute`**. VoteRAG sobrevive solo porque su runner **no pasa** `tmax_absolute` explícitamente (lo deja venir vía `**dp_flags`).

**Fix sugerido.** Una de tres:
1. **Stripear** en `resolve_dp_config(...)` cualquier llave que el runner ya esté pasando explícito.
2. **Mover** `tmax_absolute` y `stopwords_path` fuera de `PRESETS` (ya están en el bloque `dpsparsevoterag:` del YAML).
3. **Construir** el dict explícito y luego `dp_flags`-overrideable: `kwargs = {"tmax_absolute": tmax_absolute, **dp_flags}` en lugar del unpack mixto.

Recomiendo la (3): el unpack final permite que el preset `paper` imponga `tmax_absolute=50` aunque el bloque `dpsparsevoterag:` diga `200`, lo cual es la semántica esperada de un preset.

---

### B2. `prompt_construction='token_ids_append'` se ignora silenciosamente en SparseVote

**Síntoma.** Con `preset: paper` (que activa `prompt_construction: token_ids_append`), `SparseVoteRAGGenerator` debería reusar la tokenización inicial y solo extender la lista de IDs por voter (Issue 5 del análisis previo). En la práctica, no lo hace.

**Causa.** [sparse_vote_generator.py:278-288](dp_methods/dpvoterag/sparse_vote_generator.py#L278-L288):

```python
for _ in range(max(1, self.tmax_absolute)):
    decoded_so_far = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    batched_prompts = [
        self._build_prompt(query=query, docs=voter_docs, generated_so_far=decoded_so_far)
        for voter_docs in partitions
    ]
    batched_prompts.append(
        self._build_prompt(query=query, docs=[], generated_so_far=decoded_so_far)
    )
    next_tokens = self._generate_next_tokens(batched_prompts)
```

**No hay rama** que respete `self.prompt_construction == "token_ids_append"`. Cada token se decodifica a string + se reconstruye el prompt + se re-tokeniza, exactamente lo que el paper-faithful pretendía evitar (re-tokenización BPE rompe alineamiento; ver detalle en el análisis Issue 5 anterior).

`VoteRAGGenerator.generate_vote` sí tiene la bifurcación correcta ([generator.py:546-590](dp_methods/dpvoterag/generator.py#L546-L590)). Sparse hereda de VoteRAG pero **sobreescribe `generate_vote` completo** sin replicar la bifurcación.

**Efecto.** En modo `paper`, Sparse opera con re-tokenización aunque YAML diga lo contrario. Resultados de Sparse no son comparables al paper en este eje aunque el preset esté activo.

**Fix sugerido.** Reescribir el loop de Sparse usando dos paths simétricos a los de VoteRAG (mantener listas `prompt_token_ids_per_voter` + `prompt_token_ids_public`, extenderlas con `winning_token` cada step). El público también necesita su propia lista de IDs porque su prompt difiere (`docs=[]`).

---

## 2. Divergencias funcionales (P1)

### B3. `fail_mode='ld_pate'` no replica al referente

**Referente** ([utils/ensemble.py:124-132](https://github.com/tacchan7412/DPRAG/blob/main/utils/ensemble.py#L124-L132)):

```python
if dp_engine.fail_mode == 'ld_pate':
    try:
        majority_token, _ = majority_vote(next_tokens, token_dim,
                                          dp_engine=dp_engine, k_bar=token_dim-1)
    except NotFoundLDTop1:
        return None  # stop
```

Un retry con `k_bar=dim-1`. Si falla → stop. Esto coincide con el paper.

**Local** ([generator.py:323-333](dp_methods/dpvoterag/generator.py#L323-L333)):

```python
if self.fail_mode == "ld_pate":
    for attempt_k_bar in range(k_bar - 1, 0, -1):
        try:
            return int(self._limited_domain_or_fail(...,
                       k_bar=attempt_k_bar, sens=sens)), 0
        except NotFoundLDTop1:
            continue
    return int(self._np_rng.integers(0, self.vocab_size)), 0  # random fallback
```

Retry progresivo desde `k_bar-1` hasta `1`, y si todos fallan, devuelve token aleatorio. Esto **infla utility** vs el referente porque (i) hay más oportunidades de éxito, (ii) el random fallback nunca devuelve None.

**Efecto.** Con `fail_mode='ld_pate'` los benchmarks locales mostrarán mayor `match_accuracy` y menos `exit_status=1` que el referente / paper. Para replicar, hay que usar `fail_mode='stop'` (que es lo que ya hace el preset `paper`). Pero el modo `ld_pate` queda como deviación silenciosa.

**Fix sugerido.** O bien (a) reducir el local a un solo retry + stop (replicar al pie de la letra), o (b) introducir un nuevo nombre `fail_mode='ld_pate_aggressive'` para distinguir del referente. La opción (a) es más segura para el preset `paper`.

---

### B4. `data/stopword_words.pt` falta del repo

**Síntoma.** Con `preset: paper` el código intenta `torch.load("data/stopword_words.pt")`. El archivo no existe ([data/](data/) solo tiene `stopword_words.txt`), por lo que [sparse_vote_generator.py:140-149](dp_methods/dpvoterag/sparse_vote_generator.py#L140-L149) cae al fallback `local_txt` con `warnings.warn(...)`.

**Efecto.** El conjunto de stopwords difiere del paper aunque el flag diga `paper_pt`. El warning puede pasar desapercibido en logs largos.

**Fix sugerido.** O bien (a) descargar `stopword_words.pt` desde el repo de referencia (`/tmp/dprag-ref/data/stopword_words.pt` si existe ahí) y comprometerlo; (b) generar el `.pt` a partir del `.txt` local con `torch.save([line.strip() for line in open(...)], "data/stopword_words.pt")` y documentar el procedimiento; (c) **fallar duro** en lugar de fallback silencioso cuando el preset es `paper` y el archivo no aparece. Recomiendo (a)+(c).

---

### B5. `sparse_eps` independiente diverge del paper (replica al referente)

**Paper** (Algorithm 2, sec. 4.2): `(ε_token-RAG, δ_token-RAG) = (ε_token/2, δ_token)` y `ε_token-Lap = ε_token/2`. **Un único** ε_token se parte 50/50.

**Referente** y **local**: dos parámetros independientes (`epsilon_token` para LD, `sparse_eps` para SVT). Tanto el README como la API de `tacchan7412/DPRAG` lo confirman (`--dp_eps 1 --sparse_eps 1`). Local lo replica.

**Efecto.** No es un bug de implementación si el objetivo es replicar al referente. Es importante documentarlo: **el referente diverge del paper** y el local copia esa divergencia. El budget total reportado por el accountant local es comparable al referente, no al paper si uno se atiene literalmente al texto de la Algorithm 2.

**Acción.** Añadir al README local una nota explícita: *"`sparse_eps` se trata como presupuesto independiente, siguiendo `tacchan7412/DPRAG`, no la división `ε_token/2` descrita en el paper. Para replicar el paper estrictamente, fijar `sparse_eps = epsilon_token/2`."*

---

### B6. Tokenización de `question_words` para exclusión

**Referente** ([eval_rag_generator.py:90](https://github.com/tacchan7412/DPRAG/blob/main/eval_rag_generator.py#L90)):

```python
question_token_ids = tokenizer(question, return_tensors="pt", padding=True)["input_ids"][0].numpy().tolist()
```

`tokenizer(question)` agrega `<bos>` (id 2 en OPT). El set `exclude_ids` incluye `<bos>`.

**Local** ([sparse_vote_generator.py:117-130](dp_methods/dpvoterag/sparse_vote_generator.py#L117-L130) y [274](dp_methods/dpvoterag/sparse_vote_generator.py#L274)):

```python
def _encode_text_to_ids(self, text):
    encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ...

# In generate_vote:
exclude_ids.update(self._encode_text_to_ids(query))
```

`add_special_tokens=False` ⇒ **no incluye `<bos>`** en `exclude_ids`. Si el LLM público predice `<bos>` (caso patológico raro pero posible al inicio), local lo trata como token DP-vote candidate, referente lo excluye.

**Efecto.** Diferencia menor en outputs, pero traza distinta de `dp_flags_per_token`.

**Fix.** Para `question_words` usar `tokenizer(query)["input_ids"]` sin `add_special_tokens=False`. Mantener `_encode_text_to_ids` como helper para stopwords (que sí necesita ese modo cuando `stopwords_token_index=None`).

---

## 3. Observaciones de accounting (P2)

### B7. `em_peeling_autodp` re-implementa `autodp`, no la usa

**Estado.** [metrics/privacy_accounting_autodp.py:1-114](metrics/privacy_accounting_autodp.py#L1-L114) **no importa `autodp`**. Reimplementa `RDP_EM` (Mironov 2017 Remark 3.4), `EM_peeling`, y subsampling Poisson como funciones puras. La biblioteca `autodp>=0.2.1` está en `requirements.txt` pero **no se usa**.

**Diferencias detectables.**
- El subsampling local usa la fórmula directa `log(1 - q + q·exp((α-1)·ρ))/(α-1)` (Balle et al. 2018). El referente usa `transformer_zoo.AmplificationBySampling(improved_bound_flag=False)` con la corrección `q · (1 - δ)/(1 - q·δ)` ([dp_accountant.py:122-124](https://github.com/tacchan7412/DPRAG/blob/main/utils/dp_accountant.py#L122-L124)). Para `q=1.0` ambos colapsan a "no subsampling" — sin divergencia con el preset por defecto.
- Para `subsampling_rate < 1` los resultados pueden diferir 5-15% según α óptimo.

**Acción.** Si `subsampling_rate=1.0` siempre (preset paper), **no hay impacto numérico**. Si el usuario activa subsampling, la cota local puede ser ligeramente diferente del referente. Documentar; o reemplazar el port por la lib real.

---

### B8. `subsampling_rate` en `LDGumbelAccountant` es parámetro fantasma

[metrics/privacy_accounting_ld.py:401](metrics/privacy_accounting_ld.py#L401) lo guarda como atributo pero **no lo usa** en `account_token`/`check_budget` (ambos ignoran subsampling). Solo `AutodpLDAccountant` lo aplica.

**Efecto.** Con `accounting_method='simple_bound'` y `subsampling_rate=0.5`, el `dp_spent` reportado **es idéntico** al de `subsampling_rate=1.0`. El parámetro miente.

**Fix.** O bien aplicar la amplificación en `simple_bound` (la cota genérica también admite multiplicar por `q`), o bien eliminar el atributo y documentar que `subsampling_rate` solo aplica a `em_peeling_autodp`.

---

### B12. `sparse_account_svt: true` no contabiliza nada

[sparse_vote_generator.py:101](dp_methods/dpvoterag/sparse_vote_generator.py#L101) lee el flag y lo guarda; nunca se usa. El código tiene un placeholder en el comentario "Issue 3: account for the private token" pero no hay correspondiente `account_svt_step()` que sume el costo Laplace del SVT al budget.

El paper de Dwork & Roth §3.6 (SVT estándar) prueba que el SVT en sí mismo es `(2·ε_lap)`-DP independiente del número de queries (mientras el contador "cumulative-fail" no exceda el límite). El referente tampoco lo contabiliza explícitamente al accountant, asumiendo que ε_svt está fuera del LD-budget.

**Efecto.** El flag es no-op. `dp_spent_epsilon` reportado por Sparse cuenta solo el LD, no el SVT. Para el budget total real hay que sumar `T_total · ε_svt` aparte (donde `T_total` es el total de iteraciones del while-loop, no solo las DP).

**Fix.** O implementar `account_svt_step` (sumar `2·sparse_eps`-DP por iter por composición clásica), o renombrar el flag a `sparse_account_svt: false  # not implemented` y dejar claro en docs.

---

## 4. Aclaraciones a análisis previos

### B9. "Issue 4: `max(tmax_seq, tmax_adv)` viola DP" es incorrecto

El análisis anterior afirmaba que `max(tmax_seq, tmax_adv)` viola la garantía DP y debería ser `min`. Esto es **falso**:

- Cada teorema de composición da una cota válida sobre el costo total tras T queries.
- Sea `f_seq(T) = T·ε` (sequential) y `f_adv(T) ≈ √(2T·log(1/δ))·ε + T·ε(e^ε - 1)` (advanced).
- Ambas cotas son **simultáneamente válidas**: si `f_seq(T) ≤ ε_total` o `f_adv(T) ≤ ε_total`, entonces el mecanismo a T queries es `(ε_total, δ)`-DP.
- Por tanto el T máximo permitido es `max{T : f_seq(T) ≤ ε_total OR f_adv(T) ≤ ε_total}`, que es **exactamente `max(T_seq*, T_adv*)`**, con `T_seq* = floor(ε_total/ε)` y `T_adv*` raíz numérica.
- El paper mismo ([sec. T_max](https://arxiv.org/html/2412.04697v1)) dice literal: *"we take the maximum of two numbers of possible composition"*.

`tmax_strategy='accountant_only'` (preset paper) hace algo distinto y mejor: ignora la cota a priori y usa el accountant online para gating real. Eso replica el comportamiento del referente (`check_dp_budget()`).

`tmax_strategy='legacy_max'` (preset local) coincide con la fórmula del paper. **No es un bug**. El comentario "viola garantía" en el README anterior debería removerse.

---

### B11. Defaults de YAML difieren del paper

| Param | Paper | YAML local | Referente CLI |
|---|---|---|---|
| `m` (n_split) | 30/40/50 | `30` | `--n_split 50` |
| `epsilon_token` (VoteRAG) | 2 | `2` | `--dp_eps 2` |
| `epsilon_token` (Sparse) | 1 | `2` | `--dp_eps 1` |
| `sparse_eps` | (no aplica, paper splits) | `1` | `--sparse_eps 1` |
| `delta_token` | 1e-5 | `1e-5` | `--dp_delta 1e-5` |
| `tau` (threshold Sparse) | `m/2` (≡15 con m=30) | `15` (con m=30) | `--threshold 25` (con m=50) |
| `delta_total` | 1e-4 | `1e-4` | `--target_delta 1.0` |
| `target_eps` | (composición clásica) | (online accountant) | `--target_eps 1000` |
| `fail_mode` | implicit `stop` | `stop` (via preset) | `fail_mode='stop'` |

**Acción.** Cambiar `epsilon_token` a `1` en `dpsparsevoterag.yaml` (preset paper) para alinear con el paper exactamente; o documentar que el local usa 2 por compatibilidad histórica.

---

### B10. `_generate_next_tokens_vllm` es un stub

[generator.py:411-412](dp_methods/dpvoterag/generator.py#L411-L412):

```python
def _generate_next_tokens_vllm(self, prompts):
    raise RuntimeError("vLLM integration requires model name preservation (not yet implemented).")
```

El "Paso 11: vLLM optional" se anuncia como completado en el review previo, pero el path siempre cae al fallback HF. No es un bug funcional (HF batching funciona), pero la documentación es engañosa.

---

## 5. Estado real de los pasos 1–12 reportados

Re-evaluación del checklist anterior:

| Paso | Reclamado | Realidad |
|---|---|---|
| 1 — LD-Gumbel formal (DR'19) | ✅ | ✅ Correcto. `_limited_domain_or_fail` implementa DR'19 con `h_perp` y comparación `v_max ≥ v_perp`. |
| 2 — Fail modes (4 estrategias) | ✅ | ⚠️ Implementado, pero `ld_pate` **no replica el referente** (B3). |
| 3 — Privacy accounting RDP | ✅ | ⚠️ Implementado y disponible en dos métodos, pero `simple_bound` ignora `subsampling_rate` (B8) y `em_peeling_autodp` re-implementa autodp (B7). |
| 4 — Auto-stop por overflow | ✅ | ✅ Correcto en VoteRAG y Sparse (con `sparse_use_online_budget`). |
| 5 — `sparse_eps` separado | ✅ | ✅ Correcto vs referente; documentar B5. |
| 6 — Exclude stopwords / question_words | ✅ | ⚠️ Implementado, pero question_words tokeniza distinto al referente (B6). |
| 7 — Batching HF | ✅ | ✅ Funciona. |
| 8 — Persistencia `exit_status` | ✅ | ✅ Correcto. |
| 9 — Documentación README LD-RDP | ✅ | ✅ README actualizado. |
| 10 — F1 score | ✅ | ✅ En `metrics/utility.py`. |
| 11 — vLLM optional | ✅ | ❌ Stub que siempre cae a HF (B10). |
| 12 — KV cache | ✅ | ✅ Con fallback graceful. |
| **(extra) Replicabilidad Issues 1-12** | "12/12 parcheados" | ⚠️ Parcheados pero **el runner Sparse no es ejecutable** (B1) y `prompt_construction='token_ids_append'` no llega a Sparse (B2). |

---

## 6. Métricas — qué falta

| Métrica | Paper | Local | Comentario |
|---|---|---|---|
| `match_accuracy` | sí (Fig. 6) | ✅ `metrics/utility.py:match_accuracy` | exact match normalizado. |
| `f1_score` (token-based) | sí | ✅ `metrics/utility.py:f1_score` | OK. |
| `exact_match_score` (SQuAD-style) | sí (`utils_rag.exact_match_score`) | ⚠️ aproximado por `match_accuracy` | El paper usa la versión de `utils_rag` con normalización SQuAD (lower, strip articles, strip punct). Local solo hace lower+strip. |
| `dp_spent_epsilon` por query | sí | ✅ persistido en payload | OK. |
| `exit_status` por query | sí | ✅ persistido | OK. |
| `dp_flags_per_token` (Sparse) | sí (`*_dp-flags-list.txt`) | ✅ en `last_run_stats` y payload | OK. |
| `mia_auc_bleu` | (no en paper) | local-only | Adicional, OK. |

**Acción.** Considerar reemplazar `match_accuracy` por una variante con normalización SQuAD (`re.sub(r"\b(a|an|the)\b", " ", ...)`, strip puntuación, collapse whitespace) si se quiere igualar lectura del paper.

---

## 7. Plan de remediación sugerido (orden de prioridad)

1. **B1 + B2** (P0). Sin esto el preset `paper` para Sparse es teórico.
   - B1: en `run_dpsparsevoterag.py`, eliminar el unpack-mixto. Reemplazar por:
     ```python
     all_kwargs = dict(
         model=model, tokenizer=tokenizer, m=m, k=k,
         epsilon_token=epsilon_token, sparse_eps=sparse_eps,
         delta_token=delta_token, epsilon_total=float(epsilon_total),
         delta_total=delta_total, k_bar=k_bar, random_seed=seed,
         tau=tau, exclude_stopwords=exclude_stopwords,
         exclude_question_words=exclude_question_words,
     )
     # explicit YAML values come first; preset overrides last
     all_kwargs["tmax_absolute"] = tmax_absolute
     all_kwargs["stopwords_path"] = stopwords_path
     all_kwargs.update(dp_flags)  # preset takes precedence
     generator = SparseVoteRAGGenerator(**all_kwargs)
     ```
   - B2: refactorizar el loop de Sparse para reusar el path `token_ids_append` (mantener `prompt_token_ids_per_voter` + `prompt_token_ids_public`).

2. **B3 + B6** (P1, replicabilidad). Acercar comportamiento al referente.

3. **B4** (P1, datos). Comprometer `data/stopword_words.pt` o cambiar el fallback a error duro cuando preset=paper.

4. **B12 + B8** (P2, accounting). Decidir: implementar de verdad o eliminar el flag/atributo.

5. **B10 + B11** (P2, doc/cosmético).

6. **Tests** (P0 también, indirectamente). Añadir `tests/test_runner_smoke.py` que invoque ambos runners con `max_queries: 1` por preset; eso habría detectado B1 inmediatamente.

---

## 8. Comandos de reproducción

### Clonar referente
```bash
git clone https://github.com/tacchan7412/DPRAG.git /tmp/dprag-ref
diff -r /tmp/dprag-ref/utils/dp.py dp_methods/dpvoterag/  # las firmas no coincidirán; revisar manualmente
```

### Reproducir B1
```bash
python -c "
import sys; sys.path.insert(0, '.')
from dp_methods.dpvoterag.presets import resolve_dp_config
from dp_methods.dpvoterag.sparse_vote_generator import SparseVoteRAGGenerator
dp_flags = resolve_dp_config({'preset': 'paper'})
print('dp_flags has tmax_absolute:', 'tmax_absolute' in dp_flags)
print('dp_flags has stopwords_path:', 'stopwords_path' in dp_flags)
# call would be: SparseVoteRAGGenerator(..., tmax_absolute=200, stopwords_path='x', **dp_flags)
# → TypeError
"
```

### Smoke test post-fix
```bash
pytest -q tests/
python experiments/run_dpvoterag.py --config experiments/configs/dpvoterag.yaml  # debe correr (paper preset)
python experiments/run_dpsparsevoterag.py --config experiments/configs/dpsparsevoterag.yaml  # actualmente crashea con TypeError
```

---

**Fin del informe.**
