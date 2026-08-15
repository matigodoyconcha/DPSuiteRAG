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
.venv/bin/python analysis/derive_epsilon_grids.py --targets 1 5 10 50 --queries 100
```

Invierte el eje común y devuelve, para cada método, el valor de **su** knob que
corresponde a cada ε_total. Sale en `results/epsilon_grids.json`.

> Este paso es de inspección, no de configuración: cada YAML de `full/` declara
> `epsilon_targets: [1, 5, 10, 50]` y `workload_queries: [1, 5, 25, 100]`, y el runner
> hace la misma inversión (`budget_for_target`) celda a celda, de modo que un objetivo
> y el valor que lo implementa no pueden separarse. `tests/test_shared_grid.py`
> recompone cada celda contra su objetivo.

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
    --methods no_rag vanilla dp_rag dpvoterag dpsparsevoterag dpksa private_rag private_rag_ada dp_synrag
```

Con un dataset y un ε por método, para verificar que `metrics_summary.csv` sale con las
9 filas y medir segundos por consulta. **El grid principal se dimensiona con estos
números medidos, no con estimaciones.** Comprueba también `answer_rate`: un método que
sale con respuestas vacías está apagado, no es que rinda peor. Para DPSparseVoteRAG la
columna es `private_answer_rate`: su camino público responde aunque el mecanismo privado
no dispare nunca.

### A/B de la caché KV (GPU ligera, obligatorio antes del paso 5)

Los bucles de generación token a token van a pasar a cachear las claves y valores de
atención en vez de re-codificar el prompt en cada paso. Sobre el papel es la misma
cuenta; en coma flotante no lo es: cambia el orden de reducción, los logits se mueven
~1e-6, y estos métodos hacen argmax sobre un histograma de votos ruidoso. Dos candidatos
separados por menos de 1e-6 pueden intercambiarse y la respuesta cambia. Así que la
salida **no** va a ser idéntica bit a bit, y preguntar si lo es no sirve. La pregunta es
si lo que se mueve son empates y se mueve sin sesgo.

```bash
.venv/bin/python scripts/ab_kv_cache.py --dry-run      # imprime el plan, no ejecuta nada

nohup .venv/bin/python scripts/ab_kv_cache.py \
    --queries 50 --seed 1 > logs/ab_kv_cache.log 2>&1 &
```

Corre las mismas celdas dos veces —mismo config, mismo `data_seed`, mismo `run_seed`,
mismo índice— una con la caché y otra con `BENCHDP_DISABLE_KV_CACHE=1` forzando el camino
viejo, y compara los dos árboles de resultados. Deja `ab_kv_cache_summary.json` (para
leer con código) y `ab_kv_cache_report.txt` (para leer a ojo) en
`results/ab_kv_cache_<stamp>/`. Sale con 0 solo si el veredicto es `PASS`.

**El criterio, en tres partes:**

1. **ε exacto o es un bug.** El presupuesto que se cobra a un método sale de su config y
   de su contabilidad, no de nada que calcule un generador. `epsilon`,
   `epsilon_workload`, `epsilon_workload_basic`, `epsilon_workload_by_q`,
   `workload_queries` y `workload_regime` tienen que ser iguales hasta el último bit.
   `epsilon_workload_spent` y `budget_utilisation` quedan fuera a propósito: cuentan
   liberaciones realmente hechas, así que un argmax volteado que corta la generación un
   token antes los mueve con razón. Se reportan, no deciden.
2. **Lo que se mueva tiene que ser insesgado.** Es la comprobación que sostiene el
   argumento, y es pareada: ambos brazos responden las mismas preguntas en el mismo
   orden. Entre las respuestas que cambiaron, la corrección por consulta se compara con
   un McNemar exacto bilateral. Bajo la hipótesis de que la caché solo resuelve empates,
   ganar y perder son intercambiables y el test pasa; una caché que ayuda o perjudica
   sistemáticamente lo falla aunque las dos medias caigan cerca. Además la tasa de
   cambio está topada en 25 % (`--max-flip-rate`): el ruido DP de estos histogramas es
   órdenes de magnitud mayor que 1e-6, así que los empates deberían ser raros, y si
   cambia un cuarto de las respuestas la explicación ya no es "coma flotante".
3. **Las métricas agregadas dentro del ruido que el repo ya tolera.** Es una red de
   seguridad. Conviene decir de dónde sale el umbral: no hay corridas con semillas
   repetidas todavía, así que no existe un número de ruido semilla-a-semilla medido. Lo
   que sí hay son 23 configuraciones ejecutadas dos veces en corridas distintas (`dpksa`
   entre `run_20260702_114726` y `run_20260712_094517`, n=40; `dp_synrag` entre
   `run_20260710_100921` y `run_20260712_224716_maquinatoro`, n=102-200). Esas parejas
   mezclan ruido con deriva de código entre fechas, así que **acotan** el ruido por
   arriba en vez de medirlo. Las más limpias son las de n=40: |Δ| ≤ 0.05 en
   `match_accuracy` y ≤ 0.03 en `rouge_l`, `f1_score` y `bert_score_f1`. Esos son los
   valores por defecto. Son holgados a propósito: con n consultas toda métrica de
   proporción está cuantizada en 1/n, y una banda más estrecha fallaría por aritmética y
   no por ciencia. La parte 2 es la que afila el veredicto.

`INCONCLUSIVE` se reserva para cuando no se midió nada: sobre todo si el brazo con caché
no fue más rápido (< 1.05×), que casi siempre significa que la variable de entorno no se
está honrando y los dos brazos corrieron el mismo código.

**Lo que tiene que existir del lado del generador.** El script pasa exactamente una
variable, siguiendo la convención `BENCHDP_*` que ya usa el orquestador:

    BENCHDP_DISABLE_KV_CACHE=1   ->  camino viejo, re-codificar en cada token

Sin ella (o con cualquier otro valor) se usa la caché. Mientras ningún fichero del árbol
lea ese nombre el script se niega a correr —dos brazos por el mismo camino coinciden
perfectamente y no prueban nada—; `--allow-unwired` salta la negativa para ensayar.

Por defecto entran los cinco métodos con bucle propio token a token (`dp_rag`,
`dpvoterag`, `dpsparsevoterag`, `private_rag`, `private_rag_ada`) más `vanilla` como
control: `vanilla` llama a `generate()`, no cambia, y si deja de reproducirse entre
brazos el que está midiendo mal es el arnés. Con `--queries 50` sobre el corpus smoke
son ~2-6 h de GPU pequeña; los 4 queries del set smoke no sirven aquí, porque con n=4
una sola respuesta volteada mueve cada métrica 0.25.

Si hay que re-analizar sin volver a generar:

```bash
.venv/bin/python scripts/ab_kv_cache.py \
    --compare-only results/ab_kv_cache_<stamp>/cache_on results/ab_kv_cache_<stamp>/cache_off
```

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
| `private_answer_rate` / `public_only` / `tokens_private` | si el mecanismo llegó a disparar. En DPSparseVoteRAG el SVT enruta tokens por un camino público sin presupuesto, así que una celda que el mecanismo no puede servir igual responde y igual puntúa: `answer_rate` no distingue esos casos y estas tres sí |
| `query_id` | permite verificar que dos métodos respondieron el mismo conjunto |
| `retrieved_ids` / `nonprivate_retrieved_ids` | alimentan `retrieval_precision_at_k`, lo único que hace visible el agotamiento de presupuesto |
| `prefix_curve` (MuRAG) | utilidad y presupuesto tras cada consulta del stream: la figura de escalamiento con Q sale de aquí |
