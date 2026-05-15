# DP-Benchmark-RAG

Framework modular y reproducible para evaluar metodos de Privacidad Diferencial (DP)
en sistemas Retrieval-Augmented Generation (RAG): curvas del trade-off
privacidad-utilidad para distintos valores de ε.

> La especificacion completa de implementacion es
> [`benchdp_rag_agent_guide.md`](benchdp_rag_agent_guide.md) — **base oficial** del
> proyecto. Lo no implementado se construye como sale ahi; lo implementado se
> documenta ahi tal como quedo en el codigo. Este README es solo el quick-start.

## Estado actual

| Componente | Estado |
|---|---|
| Interfaces abstractas `core/` + `RAGPipeline` | ✅ |
| Pipeline vanilla end-to-end | ✅ |
| Datasets (NQ, TriviaQA, HotpotQA, SQuAD, ChatDoctor + SearchQA, MovieLens, medical_synth, medical_dirichlet_phi3) | ✅ |
| Metricas de utilidad/privacidad + privacy accounting (RDP / zCDP / autodp) | ✅ |
| **DP-RAG** (DP-ICL) | ✅ |
| **DPVoteRAG / DPSparseVoteRAG** (Limited Domain Gumbel + SVT) | ✅ |
| **DP-SynRAG** (corpus sintetico DP, zCDP) | ✅ |
| **LPRAG** | 🚧 pendiente |
| **RP-RAG** | 🚧 pendiente |
| **PRIVATE-RAG** | 🚧 pendiente |
| Runner central `experiments/runner.py` | 🚧 pendiente |

## Requisitos

- Python 3.10+
- Entorno virtual activo

## Instalacion

```bash
pip install -r requirements.txt
pip install -e .
```

## Ejecutar experimentos

Cada metodo tiene su script `experiments/run_<metodo>.py` y su YAML en
`experiments/configs/`. Los runners DP **solo** aceptan `--config`, `--verbose`
y `--verbose-max-examples`; todos los parametros del experimento (dataset, LLM,
ε, etc.) viven en el YAML. Usa las variantes `*_quick.yaml` para smoke runs.

```bash
# Baseline vanilla (sin DP)
python experiments/run_vanilla.py --config experiments/configs/vanilla.yaml

# DP-RAG  (barre la grilla de ε del YAML)
python experiments/run_dp_rag.py --config experiments/configs/dp_rag.yaml

# DPVoteRAG / DPSparseVoteRAG  (scripts independientes)
python experiments/run_dpvoterag.py       --config experiments/configs/dpvoterag.yaml
python experiments/run_dpsparsevoterag.py --config experiments/configs/dpsparsevoterag.yaml

# DP-SynRAG  (ε es SALIDA del accountant zCDP, no se barre)
python experiments/run_dp_synrag.py --config experiments/configs/dp_synrag.yaml
```

Resultados en `results/<metodo>/...` (no se commitea). Analisis y figuras:
`analysis/summary_table.py` y `analysis/plot_tradeoff.py`.

## Configurar dataset, LLM y splits

Todo se controla en el YAML del metodo (bloques `dataset`, `llm`, `models`,
`pipeline`, `generation`/`inference`):

- `dataset.name`: `squad | chatdoctor | natural_questions | triviaqa | hotpotqa |
  searchqa | movielens | medical_synth | medical_dirichlet_phi3`.
- `dataset.index_split` / `test_split`: que parte se indexa vs se evalua.
- `dataset.max_index_docs` / `max_test_queries` (o `max_samples`): acotan tamano
  para pruebas rapidas.
- `dataset.docs_path` / `queries_path`: documentos/queries propios
  (`.txt/.json/.jsonl`; queries como `{"query": ..., "answer": ...}`).
- `llm.name` (proveedor, hoy `huggingface`) y `llm.model_name`; `models.embedder`.

### Overrides por CLI — solo `run_vanilla.py`

```bash
python experiments/run_vanilla.py \
	--config experiments/configs/vanilla.yaml \
	--dataset squad \
	--llm-name huggingface \
	--llm-model sshleifer/tiny-gpt2 \
	--embedder-model sentence-transformers/all-MiniLM-L6-v2 \
	--index-split train --test-split validation \
	--max-index-docs 300 --max-test-queries 50 \
	--k 3 --max-new-tokens 64 \
	--output results/vanilla/vanilla_custom_run.json
```

## Tests

```bash
pytest tests/ -v
```

## Siguiente objetivo

Implementar los metodos pendientes siguiendo
[`benchdp_rag_agent_guide.md`](benchdp_rag_agent_guide.md) §4: LPRAG (5.1) →
RP-RAG (5.4) → PRIVATE-RAG (5.5) → runner central (6.1) → experimentos
completos y figuras.
