import argparse
import json
import random
import sys
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.index_cache import index_cache_dir, load_or_build_index
from bench_llms import load_llm
from core.pipeline import RAGPipeline
from bench_datasets.squad import SQuADDataset
from bench_datasets.chatdoctor import ChatDoctorDataset
from bench_datasets.natural_questions import NaturalQuestionsDataset
from bench_datasets.triviaqa import TriviaQADataset
from bench_datasets.hotpotqa import HotpotQADataset
from vanilla.chunker import FixedSizeChunker
from vanilla.embedder import HFEmbedder
from vanilla.indexer import FAISSIndexer
from vanilla.postprocessor import IdentityPostprocessor
from vanilla.prompt_builder import SimplePromptBuilder
from vanilla.reranker import IdentityReranker
from vanilla.retriever import CosineSimilarityRetriever


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def build_demo_documents() -> list[str]:
	return [
		(
			"Retrieval-Augmented Generation (RAG) combina un recuperador de documentos y "
			"un modelo generativo para responder preguntas con base en contexto externo. "
			"Un pipeline clasico incluye chunking, embeddings, indexacion y recuperacion top-k."
		),
		(
			"FAISS es una libreria eficiente para busqueda de similitud vectorial. "
			"Con IndexFlatIP y embeddings normalizados, el producto interno aproxima similitud coseno."
		),
		(
			"La Privacidad Diferencial introduce mecanismos de ruido para limitar la exposicion "
			"de informacion sensible de individuos en los datos."
		),
	]


def build_demo_queries() -> list[str]:
	return [
		"Que componentes tiene un pipeline RAG?",
		"Por que se usa FAISS en sistemas RAG?",
		"Que busca proteger la privacidad diferencial?",
	]


def default_config() -> dict[str, Any]:
	return {
		"method": "vanilla",
		"dataset": {
			"name": "demo",
			"index_split": "train",
			"test_split": "validation",
			"max_index_docs": 200,
			"max_test_queries": 25,
			"docs_path": None,
			"queries_path": None,
		},
		"llm": {
			"name": "huggingface",
			"model_name": "sshleifer/tiny-gpt2",
		},
		"models": {
			"embedder": "sentence-transformers/all-MiniLM-L6-v2",
		},
		"pipeline": {
			"chunk_size": 120,
			"chunk_overlap": 20,
			"k_retrieved": 2,
		},
		"generation": {
			"max_new_tokens": 48,
		},
		"random_seed": 42,
		"output_path": "results/vanilla/pilot_results.json",
	}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
	for key, value in updates.items():
		if isinstance(value, dict) and isinstance(base.get(key), dict):
			deep_update(base[key], value)
		else:
			base[key] = value
	return base


def load_yaml_config(path: str | None) -> dict[str, Any]:
	config = default_config()
	if not path:
		return normalize_llm_config(config)

	cfg_path = Path(path)
	if not cfg_path.exists():
		return normalize_llm_config(config)

	with cfg_path.open("r", encoding="utf-8") as f:
		file_config = yaml.safe_load(f) or {}
	if not isinstance(file_config, dict):
		raise ValueError("El archivo de configuracion YAML debe ser un objeto/dict.")
	deep_update(config, file_config)
	return normalize_llm_config(config)


def normalize_llm_config(config: dict[str, Any]) -> dict[str, Any]:
	models_cfg = config.get("models")
	legacy_llm_model = None
	if isinstance(models_cfg, dict):
		legacy_llm_model = models_cfg.pop("llm", None)

	llm_cfg = config.get("llm")
	if not isinstance(llm_cfg, dict):
		config["llm"] = {
			"name": "huggingface",
			"model_name": str(legacy_llm_model or "sshleifer/tiny-gpt2"),
		}
		return config

	if not llm_cfg.get("name"):
		llm_cfg["name"] = "huggingface"
	if not llm_cfg.get("model_name"):
		llm_cfg["model_name"] = str(legacy_llm_model or "sshleifer/tiny-gpt2")

	return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
	if args.dataset is not None:
		config["dataset"]["name"] = args.dataset
	if args.llm_name is not None:
		config["llm"]["name"] = args.llm_name
	if args.llm_model is not None:
		config["llm"]["model_name"] = args.llm_model
	if args.embedder_model is not None:
		config["models"]["embedder"] = args.embedder_model
	if args.k is not None:
		config["pipeline"]["k_retrieved"] = args.k
	if args.max_new_tokens is not None:
		config["generation"]["max_new_tokens"] = args.max_new_tokens
	if args.seed is not None:
		config["random_seed"] = args.seed
	if args.output is not None:
		config["output_path"] = args.output

	if args.index_split is not None:
		config["dataset"]["index_split"] = args.index_split
	if args.test_split is not None:
		config["dataset"]["test_split"] = args.test_split
	if args.max_index_docs is not None:
		config["dataset"]["max_index_docs"] = args.max_index_docs
	if args.max_test_queries is not None:
		config["dataset"]["max_test_queries"] = args.max_test_queries
	if args.chunk_size is not None:
		config["pipeline"]["chunk_size"] = args.chunk_size
	if args.chunk_overlap is not None:
		config["pipeline"]["chunk_overlap"] = args.chunk_overlap
	if args.docs_path is not None:
		config["dataset"]["docs_path"] = args.docs_path
	if args.queries_path is not None:
		config["dataset"]["queries_path"] = args.queries_path

	return config


def _read_json_or_jsonl(path: Path) -> list[Any]:
	if path.suffix.lower() == ".jsonl":
		rows: list[Any] = []
		with path.open("r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if line:
					rows.append(json.loads(line))
		return rows

	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError(f"El archivo {path} debe contener una lista JSON.")
	return data


def load_custom_documents(path: str) -> list[str]:
	file_path = Path(path)
	if not file_path.exists():
		raise FileNotFoundError(f"No existe el archivo de documentos: {file_path}")

	if file_path.suffix.lower() == ".txt":
		return [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]

	rows = _read_json_or_jsonl(file_path)
	docs: list[str] = []
	for row in rows:
		if isinstance(row, str):
			text = row.strip()
		elif isinstance(row, dict):
			text = str(row.get("text", "")).strip()
		else:
			text = ""
		if text:
			docs.append(text)
	return docs


def load_custom_queries(path: str) -> list[dict]:
	file_path = Path(path)
	if not file_path.exists():
		raise FileNotFoundError(f"No existe el archivo de queries: {file_path}")

	rows = _read_json_or_jsonl(file_path)
	queries: list[dict] = []
	for row in rows:
		if not isinstance(row, dict):
			continue
		query = str(row.get("query", "")).strip()
		answer = str(row.get("answer", "")).strip()
		if query:
			queries.append({"query": query, "answer": answer})
	return queries


def load_data_sources(config: dict[str, Any]) -> tuple[list[str], list[dict]]:
	dataset_cfg = config["dataset"]
	dataset_name = str(dataset_cfg["name"]).lower()

	docs_path = dataset_cfg.get("docs_path")
	queries_path = dataset_cfg.get("queries_path")
	if docs_path and queries_path:
		documents = load_custom_documents(str(docs_path))
		queries = load_custom_queries(str(queries_path))
		return documents, queries

	if dataset_name == "demo":
		docs = build_demo_documents()
		raw_queries = build_demo_queries()
		return docs, [{"query": q, "answer": ""} for q in raw_queries]

	if dataset_name == "squad":
		dataset = SQuADDataset(
			index_split=str(dataset_cfg["index_split"]),
			test_split=str(dataset_cfg["test_split"]),
			max_index_docs=int(dataset_cfg["max_index_docs"]) if dataset_cfg.get("max_index_docs") is not None else None,
			max_test_queries=int(dataset_cfg["max_test_queries"]) if dataset_cfg.get("max_test_queries") is not None else None,
			random_seed=int(config["random_seed"]),
		)
		dataset.load()
		return dataset.get_documents(), dataset.get_queries()
	if dataset_name == "chatdoctor":
		dataset = ChatDoctorDataset(
			index_split=str(dataset_cfg["index_split"]),
			test_split=str(dataset_cfg["test_split"]),
			max_index_docs=int(dataset_cfg["max_index_docs"]) if dataset_cfg.get("max_index_docs") is not None else None,
			max_test_queries=int(dataset_cfg["max_test_queries"]) if dataset_cfg.get("max_test_queries") is not None else None,
			random_seed=int(config["random_seed"]),
		)
		dataset.load()
		return dataset.get_documents(), dataset.get_queries()

	if dataset_name == "natural_questions":
		dataset = NaturalQuestionsDataset(
			index_split=str(dataset_cfg["index_split"]),
			test_split=str(dataset_cfg["test_split"]),
			max_index_docs=int(dataset_cfg["max_index_docs"]) if dataset_cfg.get("max_index_docs") is not None else None,
			max_test_queries=int(dataset_cfg["max_test_queries"]) if dataset_cfg.get("max_test_queries") is not None else None,
			random_seed=int(config["random_seed"]),
		)
		dataset.load()
		return dataset.get_documents(), dataset.get_queries()

	if dataset_name == "triviaqa":
		dataset = TriviaQADataset(
			index_split=str(dataset_cfg["index_split"]),
			test_split=str(dataset_cfg["test_split"]),
			max_index_docs=int(dataset_cfg["max_index_docs"]) if dataset_cfg.get("max_index_docs") is not None else None,
			max_test_queries=int(dataset_cfg["max_test_queries"]) if dataset_cfg.get("max_test_queries") is not None else None,
			random_seed=int(config["random_seed"]),
		)
		dataset.load()
		return dataset.get_documents(), dataset.get_queries()

	if dataset_name == "hotpotqa":
		dataset = HotpotQADataset(
			index_split=str(dataset_cfg["index_split"]),
			test_split=str(dataset_cfg["test_split"]),
			max_index_docs=int(dataset_cfg["max_index_docs"]) if dataset_cfg.get("max_index_docs") is not None else None,
			max_test_queries=int(dataset_cfg["max_test_queries"]) if dataset_cfg.get("max_test_queries") is not None else None,
			random_seed=int(config["random_seed"]),
		)
		dataset.load()
		return dataset.get_documents(), dataset.get_queries()

	raise ValueError(
		f"Dataset '{dataset_name}' no soportado en run_vanilla.py. "
		"Usa 'demo', 'squad', 'chatdoctor', 'natural_questions', 'triviaqa', 'hotpotqa' o provee docs_path/queries_path."
	)


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
	seed = int(config["random_seed"])
	set_seed(seed)

	llm = load_llm(config["llm"])
	llm_config = llm.get_config()
	llm_model = str(llm_config["model_name"])
	embedder_model = str(config["models"]["embedder"])
	chunk_size = int(config["pipeline"]["chunk_size"])
	chunk_overlap = int(config["pipeline"]["chunk_overlap"])
	k = int(config["pipeline"]["k_retrieved"])
	max_new_tokens = int(config["generation"]["max_new_tokens"])

	documents, queries = load_data_sources(config)
	if not documents:
		raise ValueError("No hay documentos para indexar.")
	if not queries:
		raise ValueError("No hay queries para evaluar.")

	chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_overlap)
	embedder = HFEmbedder(model_name=embedder_model)
	indexer = FAISSIndexer()
	retriever = CosineSimilarityRetriever(indexer=indexer)
	reranker = IdentityReranker()
	prompt_builder = SimplePromptBuilder()
	generator = llm.build_generator(max_new_tokens=max_new_tokens)
	postprocessor = IdentityPostprocessor()

	pipeline = RAGPipeline(
		chunker=chunker,
		embedder=embedder,
		indexer=indexer,
		retriever=retriever,
		reranker=reranker,
		prompt_builder=prompt_builder,
		generator=generator,
		postprocessor=postprocessor,
	)

	print(
		"Configuracion activa:\n"
		f"- dataset: {config['dataset']['name']}\n"
		f"- llm: {llm_config['name']} ({llm_model})\n"
		f"- embedder: {embedder_model}\n"
		f"- chunk_size: {chunk_size}\n"
		f"- chunk_overlap: {chunk_overlap}\n"
		f"- index_docs: {len(documents)}\n"
		f"- test_queries: {len(queries)}\n"
		f"- k: {k}\n"
		f"- max_new_tokens: {max_new_tokens}"
	)

	cache_dir = index_cache_dir(
		dataset_name=str(config["dataset"]["name"]),
		max_docs=config["dataset"].get("max_index_docs"),
		embedder_model=embedder_model,
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		seed=seed,
	)
	load_or_build_index(
		indexer=indexer,
		build_fn=lambda: pipeline.index(documents),
		cache_dir=cache_dir,
	)
	print()

	results = []
	for row in queries:
		query = str(row.get("query", "")).strip()
		if not query:
			continue

		reference = str(row.get("answer", "")).strip()

		# Single pass: inline the pipeline to reuse retrieved_chunks for logging
		query_vector = embedder.embed_query(query)
		retrieved_chunks = retriever.retrieve(query_vector, k=k)
		reranked_chunks = reranker.rerank(query, retrieved_chunks)
		prompt = prompt_builder.build(query, reranked_chunks)
		raw_answer = generator.generate(prompt)
		answer = postprocessor.process(raw_answer)

		result = {
			"query": query,
			"answer": answer,
			"reference": reference,
			"retrieved_chunks": retrieved_chunks,
		}
		results.append(result)

		print("=" * 30)
		print(f"Pregunta: {query}")
		print("-" * 30)
		print(f"CONTEXTO RECUPERADO (k={k}):")
		for chunk in retrieved_chunks:
			print(chunk["text"][:200])
		print("-" * 30)
		print(f"Respuesta: {answer}")
		if reference:
			print(f"Referencia: {reference}")
		print()


	return {
		"method": "vanilla",
		"timestamp": datetime.now(UTC).isoformat(),
		"dataset": config["dataset"],
		"llm": llm_config,
		"models": config["models"],
		"pipeline": config["pipeline"],
		"generation": config["generation"],
		"seed": seed,
		"num_index_documents": len(documents),
		"num_test_queries": len(results),
		"results": results,
	}


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Runner configurable de RAG Vanilla (dataset, modelos y splits)"
	)
	parser.add_argument(
		"--config",
		type=str,
		default="experiments/configs/vanilla.yaml",
		help="Ruta a YAML de configuracion",
	)
	parser.add_argument(
		"--dataset",
		type=str,
		default=None,
		help="Override de dataset (demo|squad|chatdoctor|natural_questions|triviaqa|hotpotqa)",
	)
	parser.add_argument("--llm-name", type=str, default=None, help="Override del tipo de LLM (ej: huggingface)")
	parser.add_argument("--llm-model", type=str, default=None, help="Override modelo LLM")
	parser.add_argument("--embedder-model", type=str, default=None, help="Override modelo de embeddings")
	parser.add_argument("--k", type=int, default=None, help="Override top-k retrieval")
	parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max new tokens")
	parser.add_argument("--seed", type=int, default=None, help="Override semilla")
	parser.add_argument("--output", type=str, default=None, help="Override ruta JSON de salida")

	parser.add_argument("--index-split", type=str, default=None, help="Split para indexar")
	parser.add_argument("--test-split", type=str, default=None, help="Split para evaluar")
	parser.add_argument("--max-index-docs", type=int, default=None, help="Maximo de docs indexados")
	parser.add_argument("--max-test-queries", type=int, default=None, help="Maximo de queries de test")
	parser.add_argument("--chunk-size", type=int, default=None, help="Override tamano de chunk en palabras")
	parser.add_argument("--chunk-overlap", type=int, default=None, help="Override overlap de chunks en palabras")
	parser.add_argument(
		"--docs-path",
		type=str,
		default=None,
		help="Path opcional de documentos custom (.txt/.json/.jsonl)",
	)
	parser.add_argument(
		"--queries-path",
		type=str,
		default=None,
		help="Path opcional de queries custom (.json/.jsonl con query y answer)",
	)
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	config = load_yaml_config(args.config)
	config = apply_cli_overrides(config, args)

	output_path = Path(str(config["output_path"]))
	output_path.parent.mkdir(parents=True, exist_ok=True)

	payload = run_experiment(config)
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, ensure_ascii=False)

	print(f"Resultados guardados en: {output_path}")


if __name__ == "__main__":
	main()
