from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from bench_datasets.chatdoctor import ChatDoctorDataset
from bench_datasets.hotpotqa import HotpotQADataset
from bench_datasets.natural_questions import NaturalQuestionsDataset
from bench_datasets.squad import SQuADDataset
from bench_datasets.triviaqa import TriviaQADataset
from core.pipeline import RAGPipeline
from dp_methods.dp_rag import DPAggregatedGenerator, DPRetriever, PerDocPromptBuilder
from metrics.privacy import compute_privacy_metrics
from metrics.utility import compute_utility_metrics
from vanilla.chunker import FixedSizeChunker
from vanilla.embedder import HFEmbedder
from vanilla.indexer import FAISSIndexer
from vanilla.postprocessor import IdentityPostprocessor
from vanilla.reranker import IdentityReranker


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def load_config(config_path: str | None) -> dict[str, Any]:
	default_path = PROJECT_ROOT / "experiments" / "configs" / "dp_rag.yaml"
	cfg_path = Path(config_path) if config_path else default_path
	with cfg_path.open("r", encoding="utf-8") as f:
		config = yaml.safe_load(f) or {}
	if not isinstance(config, dict):
		raise ValueError("DP-RAG config YAML must define a dictionary.")
	config.setdefault("verbose", False)
	return config


def _resolve_dataset(config: dict[str, Any]):
	dataset_cfg = config.get("dataset", {})
	name = str(dataset_cfg.get("name", "natural_questions")).lower()
	max_samples = dataset_cfg.get("max_samples")
	max_samples = int(max_samples) if max_samples is not None else None
	seed = int(config.get("random_seed", 42))

	if name == "natural_questions":
		dataset = NaturalQuestionsDataset(
			index_split="train",
			test_split=str(dataset_cfg.get("split", "validation")),
			max_index_docs=max_samples,
			max_test_queries=max_samples,
			random_seed=seed,
		)
	elif name == "squad":
		dataset = SQuADDataset(
			index_split="train",
			test_split=str(dataset_cfg.get("split", "validation")),
			max_index_docs=max_samples,
			max_test_queries=max_samples,
			random_seed=seed,
		)
	elif name == "chatdoctor":
		dataset = ChatDoctorDataset(
			index_split="train",
			test_split=str(dataset_cfg.get("split", "validation")),
			max_index_docs=max_samples,
			max_test_queries=max_samples,
			random_seed=seed,
		)
	elif name == "triviaqa":
		dataset = TriviaQADataset(
			index_split="train",
			test_split=str(dataset_cfg.get("split", "validation")),
			max_index_docs=max_samples,
			max_test_queries=max_samples,
			random_seed=seed,
		)
	elif name == "hotpotqa":
		dataset = HotpotQADataset(
			index_split="train",
			test_split=str(dataset_cfg.get("split", "validation")),
			max_index_docs=max_samples,
			max_test_queries=max_samples,
			random_seed=seed,
		)
	else:
		raise ValueError(f"Unsupported dataset for DP-RAG: '{name}'.")

	dataset.load()
	return dataset


def _index_once(chunker, embedder, indexer, documents: list[str]) -> None:
	all_chunks: list[str] = []
	all_metadata: list[dict[str, Any]] = []

	for doc_id, doc in enumerate(documents):
		chunks = chunker.chunk(doc)
		for chunk_id, chunk_text in enumerate(chunks):
			all_chunks.append(chunk_text)
			all_metadata.append({"doc_id": doc_id, "chunk_id": chunk_id, "text": chunk_text})

	vectors = embedder.embed(all_chunks)
	indexer.build(vectors, all_metadata)


def _save_results(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, ensure_ascii=False)


def _run_verbose_query(
	query: str,
	reference: str,
	k_retrieved: int,
	embedder: HFEmbedder,
	retriever: DPRetriever,
	reranker: IdentityReranker,
	prompt_builder: PerDocPromptBuilder,
	generator: DPAggregatedGenerator,
	postproc: IdentityPostprocessor,
) -> str:
	query_vector = embedder.embed_query(query)
	retrieved_chunks = retriever.retrieve(query_vector, k=k_retrieved)
	reranked_chunks = reranker.rerank(query, retrieved_chunks)
	prompt = prompt_builder.build(query, reranked_chunks)
	raw_response = generator.generate(prompt)
	final_response = postproc.process(raw_response)

	print("=" * 30)
	print(f"Pregunta: {query}")
	print("-" * 30)
	print(f"CONTEXTO RECUPERADO (k={k_retrieved}):")
	for idx, chunk in enumerate(retrieved_chunks, start=1):
		text = str(chunk.get("text", ""))
		score = chunk.get("score", None)
		score_str = f"score={float(score):.4f}" if score is not None else "score=N/A"
		print(f"[{idx}] {score_str} | {text}")

	print("-" * 30)
	print("INPUT AL GENERADOR:")
	try:
		parsed = json.loads(prompt)
		if isinstance(parsed, list):
			for i, p in enumerate(parsed, start=1):
				print(f"[prompt_doc_{i}] {p}")
		else:
			print(prompt)
	except (json.JSONDecodeError, TypeError):
		print(prompt)

	print("-" * 30)
	print(f"OUTPUT MODELO (raw): {raw_response}")
	print(f"OUTPUT FINAL: {final_response}")
	if reference:
		print(f"REFERENCIA: {reference}")
	print()

	return final_response


def main(config_path: str = None, verbose_override: bool | None = None) -> None:
	config = load_config(config_path)
	if verbose_override is not None:
		config["verbose"] = bool(verbose_override)
	seed = int(config.get("random_seed", 42))
	set_seed(seed)

	dataset = _resolve_dataset(config)
	documents = dataset.get_documents()
	queries = dataset.get_queries()

	chunker = FixedSizeChunker(chunk_size=256, overlap=32)
	embedder = HFEmbedder(model_name=str(config["models"]["embedder"]))
	indexer = FAISSIndexer()
	reranker = IdentityReranker()
	postproc = IdentityPostprocessor()

	print("Indexing dataset once for all epsilon combinations...")
	_index_once(chunker, embedder, indexer, documents)

	eps_r_values = [float(x) for x in config.get("epsilon_retrieval_values", [1.0])]
	eps_g_values = [float(x) for x in config.get("epsilon_generation_values", [0.5])]
	delta = float(config.get("delta", 1e-5))
	k_retrieved = int(config.get("k_retrieved", 5))
	max_docs = int(config.get("max_docs", 20))
	max_new_tokens = int(config.get("max_new_tokens", 64))
	use_batch = bool(config.get("use_batch_forward", True))

	utility_metric_names = config.get("metrics", {}).get("utility", [])
	privacy_metric_names = config.get("metrics", {}).get("privacy", [])
	private_entities = dataset.get_private_entities()
	verbose = bool(config.get("verbose", False))

	dataset_name = str(config.get("dataset", {}).get("name", "unknown"))
	output_base = Path(str(config.get("output_dir", "results/dp_rag/"))) / dataset_name

	for eps_r, eps_g in product(eps_r_values, eps_g_values):
		print(f"Running DP-RAG for epsilon_retrieval={eps_r}, epsilon_generation={eps_g}")

		retriever = DPRetriever(
			index=indexer,
			epsilon=eps_r,
			k=k_retrieved,
			max_docs=max_docs,
			seed=seed,
		)
		prompt_builder = PerDocPromptBuilder()
		generator = DPAggregatedGenerator(
			model_name=str(config["llm"]["model_name"]),
			epsilon=eps_g,
			delta=delta,
			max_new_tokens=max_new_tokens,
			seed=seed,
			use_batch=use_batch,
		)

		pipeline = RAGPipeline(
			chunker=chunker,
			embedder=embedder,
			indexer=indexer,
			retriever=retriever,
			reranker=reranker,
			prompt_builder=prompt_builder,
			generator=generator,
			postprocessor=postproc,
		)

		predictions: list[str] = []
		references: list[str] = []
		results: list[dict[str, str]] = []
		for item in tqdm(queries, desc=f"eps_r={eps_r}, eps_g={eps_g}"):
			query = str(item.get("query", "")).strip()
			if not query:
				continue
			reference = str(item.get("answer", "")).strip()
			if verbose:
				response = _run_verbose_query(
					query=query,
					reference=reference,
					k_retrieved=k_retrieved,
					embedder=embedder,
					retriever=retriever,
					reranker=reranker,
					prompt_builder=prompt_builder,
					generator=generator,
					postproc=postproc,
				)
			else:
				response = pipeline.query(query, k=k_retrieved)
			predictions.append(str(response))
			references.append(reference)
			results.append({
				"query": query,
				"prediction": str(response),
				"reference": reference,
			})

		utility_metrics = compute_utility_metrics(predictions, references, utility_metric_names)
		privacy_metrics = compute_privacy_metrics(
			predictions,
			references,
			private_entities,
			privacy_metric_names,
			seed=seed,
		)

		metrics_payload = {**utility_metrics, **privacy_metrics}
		payload = {
			"method": "dp_rag",
			"dataset": dataset_name,
			"epsilon_retrieval": eps_r,
			"epsilon_generation": eps_g,
			"epsilon_total_upperbound": eps_r + (max_new_tokens * eps_g),
			"delta": delta,
			"metrics": metrics_payload,
			"n_queries": len(predictions),
			"timestamp": datetime.now(UTC).isoformat(),
			"results": results,
		}

		out_name = f"eps_r{eps_r}_eps_g{eps_g}.json"
		out_path = output_base / out_name
		_save_results(out_path, payload)
		print(f"Saved: {out_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run DP-RAG experiment grid.")
	parser.add_argument(
		"--config",
		type=str,
		default=str(PROJECT_ROOT / "experiments" / "configs" / "dp_rag.yaml"),
		help="Path to DP-RAG YAML config.",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Print retrieval, generator input prompts and model outputs for each query.",
	)
	args = parser.parse_args()
	main(args.config, verbose_override=args.verbose)
