# DP-Benchmark-RAG

Framework para evaluar metodos de Privacidad Diferencial en sistemas Retrieval-Augmented Generation (RAG).

## Estado actual

- Estructura base del proyecto creada.
- Interfaces abstractas en core implementadas.
- Pipeline vanilla base implementado.
- Piloto de ejecucion end-to-end disponible en experiments/run_vanilla.py.

## Requisitos

- Python 3.10+
- Entorno virtual activo

## Instalacion

```bash
pip install -r requirements.txt
pip install -e .
```

## Ejecutar piloto vanilla

```bash
python experiments/run_vanilla.py \
	--config experiments/configs/vanilla.yaml
```

## Donde elegir dataset, LLM y split index/test

1. Dataset y particion:
	 - En experiments/configs/vanilla.yaml, bloque dataset.
	 - name: demo o squad.
	 - index_split y test_split definen que parte se indexa y que parte se evalua.
	 - max_index_docs y max_test_queries acotan tamano para pruebas rapidas.

2. Modelo LLM y embedder:
	 - En experiments/configs/vanilla.yaml, bloque models.
	 - llm define el generador.
	 - embedder define el modelo de embeddings.

3. Elegir exactamente documentos de index y test propios:
	 - Usa dataset.docs_path y dataset.queries_path en el YAML.
	 - docs_path acepta .txt, .json o .jsonl.
	 - queries_path acepta .json o .jsonl con objetos {"query": ..., "answer": ...}.

4. Overrides por CLI (sin tocar YAML):

```bash
python experiments/run_vanilla.py \
	--config experiments/configs/vanilla.yaml \
	--dataset squad \
	--llm-model sshleifer/tiny-gpt2 \
	--embedder-model sentence-transformers/all-MiniLM-L6-v2 \
	--index-split train \
	--test-split validation \
	--max-index-docs 300 \
	--max-test-queries 50 \
	--k 3 \
	--max-new-tokens 64 \
	--output results/vanilla/vanilla_custom_run.json
```

## Siguiente objetivo

Conectar el runner vanilla a un dataset real (NaturalQuestions subset) y guardar resultados con formato de experimento reproducible.
