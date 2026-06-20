"""Benchmark script for DPVoteRAG (Koga et al., arXiv:2412.04697).

Runs the DPVoteRAG pipeline (m voters, LD-Gumbel aggregation) over a sweep of
epsilon_total values. All parameters are taken from the YAML config.

Usage:
	python experiments/run_dpvoterag.py --config experiments/configs/dpvoterag.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from core.device import free_memory, model_device_kwargs, resolve_device
from experiments.index_cache import index_cache_dir, load_or_build_index
from bench_datasets.chatdoctor import ChatDoctorDataset
from bench_datasets.hotpotqa import HotpotQADataset
from bench_datasets.natural_questions import NaturalQuestionsDataset
from bench_datasets.medical_dirichlet_phi3 import MedicalDirichletPhi3Dataset
from bench_datasets.squad import SQuADDataset
from bench_datasets.triviaqa import TriviaQADataset
from dp_methods.per_query.dpvoterag.generator import VoteRAGGenerator
from dp_methods.per_query.dpvoterag.pipeline import DPVoteRAGPipeline
from metrics.privacy import compute_privacy_metrics
from metrics.utility import compute_utility_metrics
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


def load_config(path: str) -> dict[str, Any]:
	cfg_path = Path(path)
	if not cfg_path.exists():
		raise FileNotFoundError(f"Config file does not exist: {cfg_path}")
	with cfg_path.open("r", encoding="utf-8") as f:
		config = yaml.safe_load(f) or {}
	if not isinstance(config, dict):
		raise ValueError("The config YAML must be a dict.")
	# The general runner (experiments/runner.py) sets BENCHDP_OUTPUT_DIR to a
	# per-run timestamped directory so each run's results land in their own folder.
	_out_override = os.environ.get("BENCHDP_OUTPUT_DIR")
	if _out_override:
		config["output_dir"] = _out_override
	return config


def load_dataset(
	name: str,
	seed: int,
	max_queries: int | None,
	max_index_docs: int | None = 200,
) -> Any:
	key = str(name).lower()

	common_kwargs = {
		"max_index_docs": max_index_docs,
		"max_test_queries": max_queries,
		"random_seed": seed,
	}

	if key == "natural_questions":
		dataset = NaturalQuestionsDataset(**common_kwargs)
	elif key == "triviaqa":
		dataset = TriviaQADataset(**common_kwargs)
	elif key == "hotpotqa":
		dataset = HotpotQADataset(**common_kwargs)
	elif key == "squad":
		dataset = SQuADDataset(**common_kwargs)
	elif key == "chatdoctor":
		dataset = ChatDoctorDataset(**common_kwargs)
	elif key == "medical_dirichlet_phi3":
		dataset = MedicalDirichletPhi3Dataset(**common_kwargs)
	else:
		raise ValueError(
			f"Dataset '{name}' not supported for DPVoteRAG. "
			"Use natural_questions, triviaqa, hotpotqa, squad, chatdoctor or medical_dirichlet_phi3."
		)

	dataset.load()
	return dataset


def _iter_dataset_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
	"""Return one config dict per dataset to evaluate.

	Honors a top-level ``datasets:`` list (each entry is merged over the singular
	``dataset:`` block, which supplies defaults), and falls back to the singular
	``dataset:`` block when no list is present. This lets a single YAML sweep
	several datasets, each with its own corpus size (``max_index_docs``) and
	query count (``max_test_queries``).
	"""
	base = config.get("dataset", {}) or {}
	datasets = config.get("datasets")
	if isinstance(datasets, list) and datasets:
		return [{**base, **(entry or {})} for entry in datasets]
	return [dict(base)]


def _iter_llm_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
	"""Return one llm-config dict per LLM to evaluate.

	Honors a top-level ``llms:`` list (each entry is merged over the singular
	``llm:`` block, which supplies defaults), and falls back to the singular
	``llm:`` block when no list is present. This lets a single YAML sweep several
	LLMs, exactly the way ``datasets:`` sweeps several datasets.
	"""
	base = config.get("llm", {}) or {}
	llms = config.get("llms")
	if isinstance(llms, list) and llms:
		return [{**base, **(entry or {})} for entry in llms]
	return [dict(base)]


def _llm_label(llm_cfg: dict[str, Any]) -> str:
	"""Filesystem-safe label for an LLM, derived from its ``model_name``."""
	return str(llm_cfg.get("model_name", "model")).replace("/", "__")


def _safe_float(value: Any, default: float = 0.0) -> float:
	try:
		out = float(value)
	except (TypeError, ValueError):
		return default
	if np.isnan(out) or np.isinf(out):
		return default
	return out


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run DPVoteRAG experiment")
	parser.add_argument(
		"--config",
		type=str,
		default="experiments/configs/dpvoterag.yaml",
		help="Path to DPVoteRAG configuration YAML.",
	)
	parser.add_argument(
		"--verbose",
		action=argparse.BooleanOptionalAction,
		default=None,
		help="Print outputs per query (question, answer and reference).",
	)
	parser.add_argument(
		"--verbose-max-examples",
		type=int,
		default=None,
		help="Maximum number of queries to print in verbose mode. Prints all by default.",
	)
	return parser


def _run_verbose_query(
	query: str,
	reference: str,
	k_retrieved: int,
	embedder: HFEmbedder,
	retriever: CosineSimilarityRetriever,
	reranker: IdentityReranker,
	generator: VoteRAGGenerator,
	postprocessor: IdentityPostprocessor,
) -> str:
	query_vector = embedder.embed_query(query)
	retrieved_chunks = retriever.retrieve(query_vector, k=k_retrieved)
	reranked_chunks = reranker.rerank(query, retrieved_chunks)
	retrieved_docs = [
		str(chunk.get("text", "")).strip()
		for chunk in reranked_chunks
		if str(chunk.get("text", "")).strip()
	]
	# Turn on the generator's per-token mechanistic logging for this example only.
	generator.verbose = True
	generator.verbose_label = query
	try:
		raw_response = generator.generate_vote(query, retrieved_docs)
	finally:
		generator.verbose = False
	final_response = postprocessor.process(raw_response)

	print("=" * 30)
	print(f"Question: {query}")
	print("-" * 30)
	print(f"RETRIEVED DOCS (k={k_retrieved}):")
	for idx, chunk in enumerate(retrieved_chunks, start=1):
		text = str(chunk.get("text", ""))
		score = chunk.get("score", None)
		score_str = f"score={float(score):.4f}" if score is not None else "score=N/A"
		print(f"[{idx}] {score_str} | {text}")
	print("-" * 30)
	print(f"MODEL OUTPUT (raw): {raw_response}")
	print(f"OUTPUT FINAL: {final_response}")
	if reference:
		print(f"REFERENCIA: {reference}")
	print()

	return final_response


def _run_for_llm(
	config: dict[str, Any],
	args: argparse.Namespace,
	llm_model_name: str,
	llm_label: str,
	multi_llm: bool,
) -> list[dict[str, Any]]:
	embedder_model_name = str(config.get("models", {}).get("embedder", "sentence-transformers/all-MiniLM-L6-v2"))

	dp_cfg = config.get("dpvoterag", {})
	seed = int(dp_cfg.get("random_seed", 42))
	m = int(dp_cfg.get("m", 30))
	k = int(dp_cfg.get("k", 1))
	epsilon_token = float(dp_cfg.get("epsilon_token", 2.0))
	delta_token = float(dp_cfg.get("delta_token", 1e-5))
	k_bar = dp_cfg.get("k_bar", None)
	k_bar = int(k_bar) if k_bar is not None else None
	_tmax_cfg = dp_cfg.get("tmax_absolute", None)
	tmax_absolute = int(_tmax_cfg) if _tmax_cfg is not None else None  # None => Tmax from composition
	fail_mode = str(dp_cfg.get("fail_mode", "stop"))
	exclude_stopwords = bool(dp_cfg.get("exclude_stopwords", False))
	exclude_question_words = bool(dp_cfg.get("exclude_question_words", False))
	stopwords_path = dp_cfg.get("stopwords_path", None)

	epsilon_values = [float(v) for v in config.get("epsilon_values", [5, 10, 20, 40])]
	delta_total = float(config.get("delta_total", 1e-4))
	k_retrieved = int(config.get("k_retrieved", m * k))
	max_queries = config.get("max_queries", 20)
	max_queries = int(max_queries) if max_queries is not None else None
	verbose = bool(config.get("verbose", False)) if args.verbose is None else bool(args.verbose)
	verbose_max_examples = (
		int(args.verbose_max_examples)
		if args.verbose_max_examples is not None
		else config.get("verbose_max_examples")
	)
	verbose_max_examples = int(verbose_max_examples) if verbose_max_examples is not None else None

	utility_metrics = list(config.get("metrics", {}).get("utility", ["match_accuracy"]))
	privacy_metrics = list(config.get("metrics", {}).get("privacy", ["mia_auc_bleu"]))
	output_dir = Path(str(config.get("output_dir", "results/dpvoterag/")))
	output_dir.mkdir(parents=True, exist_ok=True)

	set_seed(seed)

	_kwargs = model_device_kwargs()
	_device = resolve_device()
	_ngpu = torch.cuda.device_count() if _device == "cuda" else 1
	print(f"Loading model/tokenizer... (device: {_device}, gpus: {_ngpu})")
	tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
	model = AutoModelForCausalLM.from_pretrained(llm_model_name, **_kwargs)
	if not _kwargs:
		model = model.to(_device)
	model.eval()

	chunker = FixedSizeChunker(chunk_size=200, overlap=20)
	embedder = HFEmbedder(model_name=embedder_model_name)
	reranker = IdentityReranker()
	prompt_builder = SimplePromptBuilder()
	postprocessor = IdentityPostprocessor()

	summary_rows: list[dict[str, Any]] = []

	for ds_cfg in _iter_dataset_configs(config):
		dataset_name = str(ds_cfg.get("name", "natural_questions"))
		max_index_docs = ds_cfg.get("max_index_docs", 200)
		max_index_docs = int(max_index_docs) if max_index_docs is not None else None
		ds_max_queries = ds_cfg.get("max_test_queries", max_queries)
		ds_max_queries = int(ds_max_queries) if ds_max_queries is not None else None

		print(
			f"\n=== Dataset: {dataset_name} "
			f"(max_index_docs={max_index_docs}, max_test_queries={ds_max_queries}) ==="
		)

		indexer = FAISSIndexer()
		retriever = CosineSimilarityRetriever(indexer=indexer)

		dataset = load_dataset(
			dataset_name, seed=seed, max_queries=ds_max_queries, max_index_docs=max_index_docs
		)
		documents = dataset.get_documents()
		queries = dataset.get_queries()
		private_entities = dataset.get_private_entities()

		if not documents:
			raise ValueError(f"Dataset '{dataset_name}' returned no documents to index.")
		if not queries:
			raise ValueError(f"Dataset '{dataset_name}' returned no queries to evaluate.")

		cache_dir = index_cache_dir(
			dataset_name=dataset_name,
			max_docs=max_index_docs,
			embedder_model=embedder_model_name,
			chunk_size=200,
			chunk_overlap=20,
			seed=seed,
		)
		index_built = False

		for epsilon_total in epsilon_values:
			generator = VoteRAGGenerator(
				model=model,
				tokenizer=tokenizer,
				m=m,
				k=k,
				epsilon_token=epsilon_token,
				delta_token=delta_token,
				epsilon_total=float(epsilon_total),
				delta_total=delta_total,
				k_bar=k_bar,
				random_seed=seed,
				tmax_absolute=tmax_absolute,
				fail_mode=fail_mode,
				exclude_stopwords=exclude_stopwords,
				exclude_question_words=exclude_question_words,
				stopwords_path=stopwords_path,
				task_instruction=dataset.get_task_instruction(),
			)

			pipeline = DPVoteRAGPipeline(
				chunker=chunker,
				embedder=embedder,
				indexer=indexer,
				retriever=retriever,
				reranker=reranker,
				prompt_builder=prompt_builder,
				generator=generator,
				postprocessor=postprocessor,
			)

			if not index_built:
				load_or_build_index(
					indexer=indexer,
					build_fn=lambda: pipeline.index(documents),
					cache_dir=cache_dir,
				)
				index_built = True

			predictions: list[str] = []
			references: list[str] = []
			results: list[dict[str, str]] = []
			dp_spent_per_query: list[dict[str, float]] = []
			exit_statuses: list[int] = []

			query_times: list[float] = []
			selected_queries = queries[:ds_max_queries] if ds_max_queries is not None else queries
			verbose_count = 0

			for sample in selected_queries:
				question = str(sample.get("query", "")).strip()
				reference = str(sample.get("answer", "")).strip()
				if not question:
					continue

				t0 = time.time()
				can_print_verbose = verbose and (verbose_max_examples is None or verbose_count < verbose_max_examples)
				if can_print_verbose:
					answer = _run_verbose_query(
						query=question,
						reference=reference,
						k_retrieved=k_retrieved,
						embedder=embedder,
						retriever=retriever,
						reranker=reranker,
						generator=generator,
						postprocessor=postprocessor,
					)
					verbose_count += 1
				else:
					answer = pipeline.query(question, k=k_retrieved)
				elapsed = time.time() - t0

				predictions.append(str(answer))
				references.append(reference)
				run_stats = dict(generator.last_run_stats)
				dp_spent = {
					"epsilon": _safe_float(run_stats.get("dp_spent_epsilon", 0.0)),
					"delta": _safe_float(run_stats.get("dp_spent_delta", 0.0)),
				}
				exit_status = int(run_stats.get("exit_status", 0))
				results.append(
					{
						"query": question,
						"prediction": str(answer),
						"reference": reference,
						"exit_status": exit_status,
						"dp_spent": dp_spent,
					}
				)
				dp_spent_per_query.append(dp_spent)
				exit_statuses.append(exit_status)
				query_times.append(float(elapsed))

			# When deferred (orchestrated run), skip in-process metrics; score_results.py
			# computes them later in a clean process. Avoids loading BERTScore/perplexity
			# models on top of the generator (OOM).
			if os.environ.get("BENCHDP_DEFER_METRICS") == "1":
				util, priv = {}, {}
			else:
				util = compute_utility_metrics(predictions, references, utility_metrics)
				priv = compute_privacy_metrics(predictions, references, private_entities, privacy_metrics, seed=seed)

			time_per_query = _safe_float(np.mean(query_times) if query_times else 0.0)
			match_acc = _safe_float(util.get("match_accuracy", 0.0))
			mia_auc = _safe_float(priv.get("mia_auc_bleu", 0.5))
			total_eps_spent = _safe_float(sum(item.get("epsilon", 0.0) for item in dp_spent_per_query))
			total_delta_spent = _safe_float(sum(item.get("delta", 0.0) for item in dp_spent_per_query))

			payload: dict[str, Any] = {
				"method": "dpvoterag",
				"timestamp": datetime.now(UTC).isoformat(),
				"dataset": dataset_name,
				"llm": llm_model_name,
				"max_index_docs": max_index_docs,
				"epsilon_total": float(epsilon_total),
				"delta_total": delta_total,
				"m": m,
				"k": k,
				"epsilon_token": epsilon_token,
				"dp_spent": {
					"epsilon_total_spent": total_eps_spent,
					"delta_total_spent": total_delta_spent,
					"per_query": dp_spent_per_query,
				},
				"exit_statuses": exit_statuses,
				"predictions": predictions,
				"references": references,
				"private_entities": private_entities,
				"results": results,
				"metrics": {**util, **priv},
				"match_accuracy": match_acc,
				"mia_auc": mia_auc,
				"time_per_query_sec": time_per_query,
			}

			output_path = output_dir / dataset_name
			if multi_llm:
				output_path = output_path / llm_label
			output_path = output_path / f"epsilon_{epsilon_total}.json"
			output_path.parent.mkdir(parents=True, exist_ok=True)
			with output_path.open("w", encoding="utf-8") as f:
				json.dump(payload, f, indent=2, ensure_ascii=False)

			summary_rows.append(
				{
					"dataset": dataset_name,
					"llm": llm_model_name,
					"epsilon_total": float(epsilon_total),
					"match_accuracy": match_acc,
					"mia_auc": mia_auc,
					"eps_spent_total": total_eps_spent,
					"time_per_query_sec": time_per_query,
					"output": str(output_path),
				}
			)

			del generator, pipeline
			free_memory()  # release memory between epsilon runs

		del dataset, indexer, retriever
		free_memory()  # release the dataset index before the next dataset

	del model, tokenizer
	free_memory()

	return summary_rows


def main() -> None:
	args = build_parser().parse_args()
	config = load_config(args.config)

	llm_configs = _iter_llm_configs(config)
	multi_llm = len(llm_configs) > 1
	summary_rows: list[dict[str, Any]] = []
	for llm_cfg in llm_configs:
		llm_model_name = str(llm_cfg.get("model_name", "facebook/opt-1.3b"))
		llm_label = _llm_label(llm_cfg)
		if multi_llm:
			print(f"\n##### LLM: {llm_model_name} #####")
		try:
			summary_rows.extend(
				_run_for_llm(config, args, llm_model_name, llm_label, multi_llm)
			)
		except OSError as exc:
			# A model we cannot load (e.g. gated HF repo without access, or an
			# offline/missing checkpoint) should not abort the whole multi-LLM
			# run; skip it and keep going with the remaining generators.
			print(f"\n[WARN] skipping LLM {llm_model_name}: {exc}")
			free_memory()
			continue

	print("\nResumen DPVoteRAG:")
	for row in summary_rows:
		print(
			f"- [{row['dataset']} | {row['llm']}] eps={row['epsilon_total']}: "
			f"match_accuracy={row['match_accuracy']:.4f}, "
			f"mia_auc={row['mia_auc']:.4f}, "
			f"eps_spent_total={row['eps_spent_total']:.4f}, "
			f"time_per_query_sec={row['time_per_query_sec']:.4f}, "
			f"out={row['output']}"
		)


if __name__ == "__main__":
	main()
