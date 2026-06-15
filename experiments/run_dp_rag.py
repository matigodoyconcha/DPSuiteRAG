"""Benchmark script for DP-RAG with exponential mechanism and DP-ICL.

Runs the DP-RAG pipeline (from Grislain et al.) with differentially private
retrieval via exponential mechanism and DP-ICL generation on multiple datasets.

Reference: Grislain et al., "RAG with Differential Privacy", arXiv:2412.19291v2

Usage:
	python experiments/run_dp_rag.py --config config/dp_rag.yaml --dataset squad --epsilon 5.0
"""

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

from experiments.index_cache import index_cache_dir, load_or_build_index
from bench_datasets.chatdoctor import ChatDoctorDataset
from bench_datasets.hotpotqa import HotpotQADataset
from bench_datasets.natural_questions import NaturalQuestionsDataset
from bench_datasets.medical_dirichlet_phi3 import MedicalDirichletPhi3Dataset
from bench_datasets.squad import SQuADDataset
from bench_datasets.triviaqa import TriviaQADataset
from core.device import free_memory
from core.pipeline import RAGPipeline
from dp_methods.per_query.dp_rag import DPAggregatedGenerator, DPRetriever, DPTopPRetriever, PerDocPromptBuilder
from metrics.privacy import compute_privacy_metrics
from metrics.privacy_accounting import compute_privacy_accounting
from metrics.utility import compute_utility_metrics
from vanilla.chunker import FixedSizeChunker, NoChunker
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


def _build_chunker(config: dict[str, Any]):
    """Return the chunker specified by the ``chunking`` section of the config.

    When ``chunking.enabled`` is ``false`` (or the section is absent and the
    default is disabled), a :class:`NoChunker` is returned.  This is the
    correct setting for DP-RAG: the exponential mechanism assumes one document
    per privacy unit; chunking a document into N pieces inflates the true
    sensitivity by up to N.

    When ``chunking.enabled`` is ``true``, a :class:`FixedSizeChunker` is
    returned with ``chunk_size`` and ``overlap`` taken from the config.
    """
    chunk_cfg = config.get("chunking", {})
    enabled = chunk_cfg.get("enabled", False)
    if not enabled:
        return NoChunker()
    chunk_size = int(chunk_cfg.get("chunk_size", 256))
    overlap = int(chunk_cfg.get("overlap", 32))
    return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)


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
	elif name == "medical_dirichlet_phi3":
		dataset = MedicalDirichletPhi3Dataset(
			index_split="train",
			test_split=str(dataset_cfg.get("split", "train")),
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
	retriever: DPRetriever | DPTopPRetriever,
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
	print(f"Question: {query}")
	print("-" * 30)
	print(f"RETRIEVED CONTEXT (k={k_retrieved}):")
	for idx, chunk in enumerate(retrieved_chunks, start=1):
		text = str(chunk.get("text", ""))
		score = chunk.get("score", None)
		score_str = f"score={float(score):.4f}" if score is not None else "score=N/A"
		print(f"[{idx}] {score_str} | {text}")

	print("-" * 30)
	print("GENERATOR INPUT (k private + 1 public):")
	try:
		parsed = json.loads(prompt)
		if isinstance(parsed, list):
			n_total = len(parsed)
			for i, p in enumerate(parsed, start=1):
				label = "[PUBLIC]" if i == n_total else f"[private {i}/{n_total - 1}]"
				# Print first 200 chars to keep output readable
				preview = p[:200].replace("\n", " ")
				print(f"  {label} {preview}{'...' if len(p) > 200 else ''}")
		else:
			print(prompt)
	except (json.JSONDecodeError, TypeError):
		print(prompt)

	print("-" * 30)
	print(f"MODEL OUTPUT (raw): {raw_response}")
	print(f"OUTPUT FINAL:        {final_response}")
	if reference:
		print(f"REFERENCIA:          {reference}")
	print()

	return final_response


def main(
	config_path: str = None,
	verbose_override: bool | None = None,
	verbose_max_examples_override: int | None = None,
) -> None:
	config = load_config(config_path)
	if verbose_override is not None:
		config["verbose"] = bool(verbose_override)
	if verbose_max_examples_override is not None:
		config["verbose_max_examples"] = int(verbose_max_examples_override)
	seed = int(config.get("random_seed", 42))
	set_seed(seed)

	dataset = _resolve_dataset(config)
	documents = dataset.get_documents()
	queries = dataset.get_queries()

	chunker = _build_chunker(config)
	embedder = HFEmbedder(model_name=str(config["models"]["embedder"]))
	indexer = FAISSIndexer()
	reranker = IdentityReranker()
	postproc = IdentityPostprocessor()

	dataset_cfg = config.get("dataset", {})
	chunk_cfg = config.get("chunking", {})
	no_chunk = not chunk_cfg.get("enabled", False)
	cache_dir = index_cache_dir(
		dataset_name=str(dataset_cfg.get("name", "unknown")),
		max_docs=dataset_cfg.get("max_samples"),
		embedder_model=str(config["models"]["embedder"]),
		chunk_size=int(chunk_cfg.get("chunk_size", 256)),
		chunk_overlap=int(chunk_cfg.get("overlap", 32)),
		seed=int(config.get("random_seed", 42)),
		no_chunk=no_chunk,
	)
	load_or_build_index(
		indexer=indexer,
		build_fn=lambda: _index_once(chunker, embedder, indexer, documents),
		cache_dir=cache_dir,
	)

	eps_r_values = [float(x) for x in config.get("epsilon_retrieval_values", [1.0])]
	eps_g_values = [float(x) for x in config.get("epsilon_generation_values", [0.5])]
	delta = float(config.get("delta", 1e-3))
	k_retrieved = int(config.get("k_retrieved", 5))
	max_docs = int(config.get("max_docs", 20))
	max_new_tokens = int(config.get("max_new_tokens", 64))
	use_batch = bool(config.get("use_batch_forward", True))
	retriever_type = str(config.get("retriever_type", "top_k")).lower()
	top_p = float(config.get("top_p", 0.3))
	top_p_alpha = float(config.get("top_p_alpha", 1.0))
	# DP-ICL hyperparameters (see configs/dp_rag.yaml for full documentation)
	theta = float(config.get("theta", 1.0))
	C = float(config.get("C", 1.0))
	alpha = float(config.get("alpha", 1.0))

	utility_metric_names = config.get("metrics", {}).get("utility", [])
	privacy_metric_names = config.get("metrics", {}).get("privacy", [])
	private_entities = dataset.get_private_entities()
	verbose = bool(config.get("verbose", False))
	verbose_max_examples = config.get("verbose_max_examples")
	verbose_max_examples = int(verbose_max_examples) if verbose_max_examples is not None else None

	dataset_name = str(config.get("dataset", {}).get("name", "unknown"))
	output_base = Path(str(config.get("output_dir", "results/dp_rag/"))) / dataset_name

	for eps_r, eps_g in product(eps_r_values, eps_g_values):
		print(f"Running DP-RAG for epsilon_retrieval={eps_r}, epsilon_generation={eps_g}, retriever={retriever_type}")

		if retriever_type == "top_p":
			retriever = DPTopPRetriever(
				index=indexer,
				epsilon=eps_r,
				p=top_p,
				alpha=top_p_alpha,
				max_docs=max_docs,
				seed=seed,
				verbose=verbose,
			)
		else:
			retriever = DPRetriever(
				index=indexer,
				epsilon=eps_r,
				k=k_retrieved,
				max_docs=max_docs,
				seed=seed,
				verbose=verbose,
			)
		prompt_builder = PerDocPromptBuilder()
		generator = DPAggregatedGenerator(
			model_name=str(config["llm"]["model_name"]),
			epsilon=eps_g,
			delta=delta,
			max_new_tokens=max_new_tokens,
			theta=theta,
			C=C,
			alpha=alpha,
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
		verbose_count = 0
		for item in tqdm(queries, desc=f"eps_r={eps_r}, eps_g={eps_g}"):
			query = str(item.get("query", "")).strip()
			if not query:
				continue
			reference = str(item.get("answer", "")).strip()
			can_print_verbose = verbose and (verbose_max_examples is None or verbose_count < verbose_max_examples)
			if can_print_verbose:
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
				verbose_count += 1
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
		# eps_g is the total generation budget; generator internally calibrated
		# token_epsilon via PLD binary search so that T compositions = (eps_g, delta)-DP.
		accounting_results = compute_privacy_accounting(
			eps_retrieval=eps_r,
			eps_generation_per_token=generator.token_epsilon,
			num_tokens=max_new_tokens,
			delta=delta,
			methods=("basic", "rdp", "advanced"),
		)

		accounting_payload = {
			method_name: {
				"epsilon": result.epsilon,
				"delta": result.delta,
				"eps_retrieval": result.eps_retrieval,
				"eps_generation_total": eps_g,
				"eps_generation_per_token": result.eps_generation,
				"eps_total_generation": result.eps_total_generation,
				"num_tokens": result.T,
			}
			for method_name, result in accounting_results.items()
		}

		payload = {
			"method": "dp_rag",
			"dataset": dataset_name,
			"retriever_type": retriever_type,
			"epsilon_retrieval": eps_r,
			"epsilon_generation": eps_g,
			"epsilon_total": eps_r + eps_g,
			"delta": delta,
			"privacy_accounting": accounting_payload,
			"metrics": metrics_payload,
			"n_queries": len(predictions),
			"timestamp": datetime.now(UTC).isoformat(),
			"results": results,
		}

		out_name = f"eps_r{eps_r}_eps_g{eps_g}.json"
		out_path = output_base / out_name
		_save_results(out_path, payload)
		print(f"Saved: {out_path}")

		del generator, retriever, pipeline  # pipeline holds refs to both; drop it too
		free_memory()  # free the per-iteration model before the next (eps_r, eps_g)


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
		action=argparse.BooleanOptionalAction,
		default=None,
		help="Print retrieval, generator input prompts and model outputs for each query.",
	)
	parser.add_argument(
		"--verbose-max-examples",
		type=int,
		default=None,
		help="Maximum number of queries to print in verbose mode. By default, prints all.",
	)
	args = parser.parse_args()
	main(
		args.config,
		verbose_override=args.verbose,
		verbose_max_examples_override=args.verbose_max_examples,
	)
