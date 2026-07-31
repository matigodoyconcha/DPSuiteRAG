# Cómo ejecutar el protocolo homogéneo

Orden de ejecución. Los pasos 1-3 son offline y baratos; el 4 mide y el 5 es el grid
real. Nada de esto se lanza automáticamente: los comandos son para ejecutarlos tú.

---

## 1. Datos (offline, ~minutos / ~horas con `--corpus`)

```bash
scripts/fetch_dpr_corpus.sh            # solo ficheros de relevancia (cientos de MB)
scripts/fetch_dpr_corpus.sh --corpus   # además el corpus de pasajes (~13 GB, ~21M)
```

Sin `data/dpr/`, Natural Questions y TriviaQA caen a un corpus degenerado construido
con sus propias preguntas, y el filtro de relevancia que declaran los métodos de
votación se salta con un warning que no llega a los ficheros de resultados. Las dos
cosas dejan esos datasets inservibles con aspecto de corrida normal.

Con el corpus de 21M, revisa el δ: 1/n ≈ 5e-8, y el δ común de 1e-2 queda muy por
encima. Es defendible pero hay que decirlo en el texto.

## 2. Caracterización del corpus (offline, sin GPU)

```bash
.venv/bin/python analysis/document_reuse.py \
    --datasets chatdoctor medical_dirichlet_phi3 movielens natural_questions triviaqa \
    --max-index-docs 20000 --max-test-queries 100 --top-k 50
```

Clasifica cada dataset como **dependiente** (pocos documentos sirven a muchas
preguntas) o **independiente**. Predice qué métodos van a sufrir antes de gastar GPU:
donde la reutilización se concentra, el filtro por documento de MuRAG agota
documentos a mitad del stream, y ahí es donde la composición por consulta duele más.

## 3. Grillas de ε (offline, instantáneo)

```bash
.venv/bin/python analysis/derive_epsilon_grids.py --targets 1 3 10 30 100 --queries 100
```

Invierte el eje común y devuelve, para cada método, el valor de **su** knob que
corresponde a cada ε_total. Sale en `results/epsilon_grids.json`.

> **Pendiente y bloqueante (tarea #16):** estas grillas todavía no están cableadas en
> los YAML. Los runners de votación recorren `epsilon_values` (el tope por consulta)
> mientras que la inversión devuelve `epsilon_token` (el presupuesto por token, que es
> lo que domina el coste del workload). Hay que decidir si el sweep pasa a recorrer
> `epsilon_token`, y verificarlo con el paso 4 antes del grid real.

## 4. Calibración medida (GPU ligera)

```bash
.venv/bin/python scripts/calibrate_tau.py \
    --datasets chatdoctor medical_dirichlet_phi3 movielens --target-charged 160
```

Re-mide el umbral τ de Private-RAG sobre **documentos enteros**. El valor actual (0.6)
se calibró con trozos de 200 palabras y el benchmark ya no parte documentos, así que la
distribución de coseno cambió. Los dos modos de fallo son silenciosos: demasiado alto
deja al retriever sin documentos (conjuntos de votantes vacíos, utilidad degradada, sin
error); demasiado bajo cobra medio corpus por consulta y agota todos los presupuestos
con M=1. Fija `private_rag.threshold` por dataset con el `recommended_tau` que imprime.

### Smoke run (tarea #19)

```bash
BENCHDP_DEFER_METRICS=1 .venv/bin/python experiments/runner.py \
    --methods vanilla dp_rag dpvoterag dpsparsevoterag dpksa private_rag private_rag_ada dp_synrag
```

Con un dataset y un ε por método, para verificar que `metrics_summary.csv` sale con las
8 filas y medir segundos por consulta. **El grid principal se dimensiona con estos
números medidos, no con estimaciones.** Comprueba también `answer_rate`: un método que
sale con respuestas vacías está apagado, no es que rinda peor.

## 5. Grid principal

```bash
.venv/bin/python experiments/runner.py --seeds 1 2 3
```

Repite cada celda variando solo el ruido del mecanismo: el corpus y las preguntas siguen
`data_seed`, así que el índice se construye una vez y la dispersión entre semillas mide
el mecanismo y no la muestra. Los resultados quedan en `<método>/seed_<n>/`.

---

## Qué mirar en los resultados

Cada JSON trae ahora, además de las métricas de utilidad:

| campo | para qué |
|---|---|
| `epsilon_workload` | el ε sobre el workload de Q consultas al δ común — **el único eje comparable entre métodos** |
| `epsilon_workload_basic` | el mismo bajo composición básica; si el ranking no cambia entre los dos, no es un artefacto contable |
| `workload_error` | por qué un método no pudo componer (su δ aditivo excede el objetivo) |
| `answer_rate` | separa "no respondió" de "respondió mal" |
| `query_id` | permite verificar que dos métodos respondieron el mismo conjunto |
| `retrieved_ids` / `nonprivate_retrieved_ids` | alimentan `retrieval_precision_at_k`, lo único que hace visible el agotamiento de presupuesto |
| `prefix_curve` (MuRAG) | utilidad y presupuesto tras cada consulta del stream: la figura de escalamiento con Q sale de aquí |
