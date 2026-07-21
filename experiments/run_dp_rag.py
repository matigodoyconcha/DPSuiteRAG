"""Benchmark script for DP-RAG with exponential mechanism and DP-ICL.

Runs the DP-RAG pipeline (from Grislain et al.) with differentially private
retrieval via exponential mechanism and DP-ICL generation on multiple datasets.

Reference: Grislain et al., "RAG with Differential Privacy", arXiv:2412.19291v2

Usage:
	python experiments/run_dp_rag.py --config experiments/configs/full/dp_rag.yaml

Datasets and epsilon values are not CLI flags — they are set in the YAML config
(the `datasets:` list and `epsilon_retrieval_values` / `epsilon_generation_values`).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
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
from experiments.common import select_datasets
from experiments.mia_probe import mia_only_enabled, resolve_mia_params, run_mia_probes
from experiments.run_logging import log, setup_method_logging
from bench_datasets.chatdoctor import ChatDoctorDataset
from bench_datasets.hotpotqa import HotpotQADataset
from bench_datasets.natural_questions import NaturalQuestionsDataset
from bench_datasets.medical_dirichlet_phi3 import MedicalDirichletPhi3Dataset
from bench_datasets.squad import SQuADDataset
from bench_datasets.triviaqa import TriviaQADataset
from bench_datasets.searchqa import SearchQADataset
from bench_datasets.movielens import MovielensDataset
from bench_datasets.dpr_corpus import corpus_available
from core.device import free_memory, is_oom_error
from core.pipeline import RAGPipeline, build_index_streaming
from dp_methods.per_query.dp_rag import DPAggregatedGenerator, DPRetriever, DPTopPRetriever, PerDocPromptBuilder
from metrics.privacy import compute_privacy_metrics
from metrics.privacy_accounting import pld_dp_rag_total
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
	default_path = PROJECT_ROOT / "experiments" / "configs" / "full" / "dp_rag.yaml"
	cfg_path = Path(config_path) if config_path else default_path
	with cfg_path.open("r", encoding="utf-8") as f:
		config = yaml.safe_load(f) or {}
	if not isinstance(config, dict):
		raise ValueError("DP-RAG config YAML must define a dictionary.")
	config.setdefault("verbose", False)
	# The general runner (experiments/runner.py) sets BENCHDP_OUTPUT_DIR to a
	# per-run timestamped directory so each run's results land in their own folder.
	_out_override = os.environ.get("BENCHDP_OUTPUT_DIR")
	if _out_override:
		config["output_dir"] = _out_override
	return config


def _iter_dataset_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
	"""Return one config dict per dataset to evaluate.

	Honors a top-level ``datasets:`` list (each entry is merged over the singular
	``dataset:`` block, which supplies defaults), and falls back to the singular
	``dataset:`` block when no list is present. This lets a single YAML sweep
	several datasets, each with its own ``max_samples`` (corpus + query cap).
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


def _resolve_dataset(dataset_cfg: dict[str, Any], seed: int):
	name = str(dataset_cfg.get("name", "natural_questions")).lower()
	max_samples = dataset_cfg.get("max_samples")
	max_samples = int(max_samples) if max_samples is not None else None
	# Optional separate query cap; falls back to max_samples (corpus == queries).
	max_queries = dataset_cfg.get("max_test_queries")
	max_queries = int(max_queries) if max_queries is not None else max_samples

	if name == "natural_questions":
		dataset = NaturalQuestionsDataset(
			index_split="train",
			test_split=str(dataset_cfg.get("split", "validation")),
			max_index_docs=max_samples,
			max_test_queries=max_queries,
			random_seed=seed,
		)
	elif name == "squad":
		dataset = SQuADDataset(
			index_split="train",
			test_split=str(dataset_cfg.get("split", "validation")),
			max_index_docs=max_samples,
			max_test_queries=max_queries,
			random_seed=seed,
		)
	elif name == "chatdoctor":
		dataset = ChatDoctorDataset(
			index_split="train",
			test_split=str(dataset_cfg.get("split", "validation")),
			max_index_docs=max_samples,
			max_test_queries=max_queries,
			random_seed=seed,
		)
	elif name == "triviaqa":
		dataset = TriviaQADataset(
			index_split="train",
			test_split=str(dataset_cfg.get("split", "validation")),
			max_index_docs=max_samples,
			max_test_queries=max_queries,
			random_seed=seed,
		)
	elif name == "hotpotqa":
		dataset = HotpotQADataset(
			index_split="train",
			test_split=str(dataset_cfg.get("split", "validation")),
			max_index_docs=max_samples,
			max_test_queries=max_queries,
			random_seed=seed,
		)
	elif name == "medical_dirichlet_phi3":
		dataset = MedicalDirichletPhi3Dataset(
			index_split="train",
			test_split=str(dataset_cfg.get("split", "train")),
			max_index_docs=max_samples,
			max_test_queries=max_queries,
			random_seed=seed,
		)
	elif name == "searchqa":
		dataset = SearchQADataset(
			index_split="train",
			test_split=str(dataset_cfg.get("split", "validation")),
			max_index_docs=max_samples,
			max_test_queries=max_queries,
			random_seed=seed,
		)
	elif name == "movielens":
		dataset = MovielensDataset(
			max_index_docs=max_samples,
			max_test_queries=max_queries,
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

	build_index_streaming(embedder, indexer, all_chunks, all_metadata)



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
	# Turn on the generator's per-token mechanistic logging for this example only.
	generator.verbose = True
	generator.verbose_label = query
	try:
		raw_response = generator.generate(prompt)
	finally:
		generator.verbose = False
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


def _run_for_llm(
	config: dict[str, Any],
	llm_model_name: str,
	llm_label: str,
	multi_llm: bool,
) -> None:
	seed = int(config.get("random_seed", 42))
	set_seed(seed)

	chunker = _build_chunker(config)
	embedder = HFEmbedder(model_name=str(config["models"]["embedder"]))
	reranker = IdentityReranker()
	postproc = IdentityPostprocessor()

	# Dataset-independent hyperparameters (parsed once for every dataset).
	eps_r_values = [float(x) for x in config.get("epsilon_retrieval_values", [1.0])]
	eps_g_values = [float(x) for x in config.get("epsilon_generation_values", [0.5])]
	delta = float(config.get("delta", 1e-3))
	k_retrieved = int(config.get("k_retrieved", 5))
	max_docs = int(config.get("max_docs", 20))
	max_new_tokens = int(config.get("max_new_tokens", 64))
	use_batch = bool(config.get("use_batch_forward", True))
	retriever_type = str(config.get("retriever_type", "top_k")).lower()
	top_p = float(config.get("top_p", 0.3))
	top_p_alpha = float(config.get("top_p_alpha", 5.0))  # mirrors DPTopPRetriever.alpha
	# DP-ICL hyperparameters (see configs/full/dp_rag.yaml for full documentation)
	theta = float(config.get("theta", 0.01))  # mirrors DPICLGenerator.theta
	temperature = float(config.get("temperature", 1.0))
	alpha = float(config.get("alpha", 1.0))
	# C2 (chat template, auto by default) + C3 (optional stop strings, OFF by default).
	use_chat_template = config.get("use_chat_template", "auto")
	stop_strings = config.get("stop_strings")

	utility_metric_names = config.get("metrics", {}).get("utility", [])
	privacy_metric_names = config.get("metrics", {}).get("privacy", [])
	verbose = bool(config.get("verbose", False))
	verbose_max_examples = config.get("verbose_max_examples")
	verbose_max_examples = int(verbose_max_examples) if verbose_max_examples is not None else None

	chunk_cfg = config.get("chunking", {})
	no_chunk = not chunk_cfg.get("enabled", False)

	# One loaded copy of the generation model, reused across every dataset and
	# (eps_r, eps_g) point for this LLM; re-reading it from disk each time is pure
	# overhead. Populated on the first generator build below.
	shared_model = None
	shared_tokenizer = None

	for ds_cfg in select_datasets(_iter_dataset_configs(config)):
		dataset_name = str(ds_cfg.get("name", "unknown"))
		log(f"Dataset START: {dataset_name} (max_samples={ds_cfg.get('max_samples')})")

		try:
			dataset = _resolve_dataset(ds_cfg, seed)
			documents = dataset.get_documents()
			queries = dataset.get_queries()
		except Exception as exc:
			log(f"Dataset SKIPPED: {dataset_name} — {type(exc).__name__}: {exc}")
			free_memory()
			continue

		# Standalone MIA mode: index only the member subset and probe the pipeline
		# instead of running the utility QA loop (score_results.py scores it offline).
		mia_split = None
		mia_params = None
		if mia_only_enabled():
			mia_params = resolve_mia_params(config)
			mia_split = dataset.get_mia_split(
				n_per_class=mia_params["n_per_class"],
				index_size=mia_params["index_size"],
				seed=mia_params["seed"],
			)
			if mia_split is None:
				log(f"[{dataset_name}] MIA: dataset has no member/non-member split; skipping.")
				free_memory()
				continue
			documents = mia_split["index_documents"]

		# DP-RAG scores every document through embeddings_matrix and never calls
		# index.search, so skip the redundant IndexFlatIP and keep only the matrix.
		indexer = FAISSIndexer(build_faiss_index=False)
		# NQ and TriviaQA retrieve from the same DPR corpus, so when it is present
		# they share one index directory instead of building it twice.
		corpus_id = (
			"dpr_psgs_w100"
			if str(dataset_name).lower() in {"natural_questions", "triviaqa"} and corpus_available()
			else None
		)
		cache_dir = index_cache_dir(
			dataset_name=dataset_name,
			corpus_id=corpus_id,
			max_docs=ds_cfg.get("max_samples"),
			embedder_model=str(config["models"]["embedder"]),
			chunk_size=int(chunk_cfg.get("chunk_size", 256)),
			chunk_overlap=int(chunk_cfg.get("overlap", 32)),
			seed=seed,
			no_chunk=no_chunk,
			build_faiss_index=False,
		)
		load_or_build_index(
			indexer=indexer,
			build_fn=lambda idx=indexer: _index_once(chunker, embedder, idx, documents),
			cache_dir=cache_dir,
		)

		output_base = Path(str(config.get("output_dir", "results/dp_rag/"))) / dataset_name
		if multi_llm:
			output_base = output_base / llm_label

		for eps_r, eps_g in product(eps_r_values, eps_g_values):
			log(f"[{dataset_name}] RUN START: eps_r={eps_r}, eps_g={eps_g}, retriever={retriever_type}")

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
			prompt_builder = PerDocPromptBuilder(
				private_template=dataset.get_prompt_template(),
				public_template=dataset.get_task_instruction() + "\nQuestion: {query}\nAnswer:",
			)
			generator = DPAggregatedGenerator(
				model_name=llm_model_name,
				model=shared_model,
				tokenizer=shared_tokenizer,
				epsilon=eps_g,
				delta=delta,
				max_new_tokens=max_new_tokens,
				theta=theta,
				temperature=temperature,
				alpha=alpha,
				seed=seed,
				use_batch=use_batch,
				use_chat_template=use_chat_template,
				stop_strings=stop_strings,
			)
			# Keep this LLM's weights loaded for the remaining mechanism settings.
			shared_model = generator.model
			shared_tokenizer = generator.tokenizer

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

			if mia_only_enabled():
				answer_fn = lambda q: pipeline.query(q, k=k_retrieved)
				mia_records = run_mia_probes(
					answer_fn,
					mia_split,
					anderson_doc_maxlen=mia_params["anderson_doc_maxlen"],
				)
				payload = {
					"method": "dp_rag",
					"dataset": dataset_name,
					"llm": llm_model_name,
					"max_samples": ds_cfg.get("max_samples"),
					"retriever_type": retriever_type,
					"epsilon_retrieval": eps_r,
					"epsilon_generation": eps_g,
					"delta": delta,
					"seed": seed,
					"timestamp": datetime.now(timezone.utc).isoformat(),
					"mia_probes": mia_records,
				}
				out_name = f"eps_r{eps_r}_eps_g{eps_g}.json"
				out_path = output_base / out_name
				_save_results(out_path, payload)
				print(f"Saved: {out_path}")
				del generator, retriever, pipeline
				free_memory()
				continue

			predictions: list[str] = []
			references: list[str] = []
			results: list[dict[str, str]] = []
			verbose_count = 0
			# Logging-only noise-impact aggregate over the run (generator.last_run_stats).
			n_tokens_sum = 0
			n_flipped_sum = 0
			t_run_start = time.time()
			total_q = len(queries)
			for q_idx, item in enumerate(
				tqdm(queries, desc=f"{dataset_name} eps_r={eps_r}, eps_g={eps_g}"), start=1
			):
				query = str(item.get("query", "")).strip()
				if not query:
					continue
				reference = str(item.get("answer", "")).strip()
				answers = [str(a).strip() for a in (item.get("answers") or [reference]) if str(a).strip()] or [reference]
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
				stats = generator.last_run_stats
				n_tokens_sum += int(stats.get("n_tokens", 0))
				n_flipped_sum += int(stats.get("n_noise_flipped", 0))
				if q_idx % 10 == 0:
					log(f"[{dataset_name} eps_r={eps_r},eps_g={eps_g}] heartbeat: query {q_idx}/{total_q}")
				predictions.append(str(response))
				references.append(reference)
				results.append({
					"query": query,
					"prediction": str(response),
					"reference": reference,
					"answers": answers,
				})
				free_memory()  # DP-RAG runs k forwards per token: free memory each query

			_flip_pct = (100.0 * n_flipped_sum / n_tokens_sum) if n_tokens_sum else 0.0
			log(
				f"[{dataset_name}] RUN END: eps_r={eps_r}, eps_g={eps_g} — "
				f"{len(predictions)} queries in {time.time() - t_run_start:.1f}s | "
				f"DP noise flipped {n_flipped_sum}/{n_tokens_sum} sampled tokens "
				f"({_flip_pct:.1f}%); token_eps={generator.token_epsilon:.4f}"
			)

			# Deferred (orchestrated run): skip in-process metrics; score_results.py
			# computes them later in a clean process (avoids BERTScore/perplexity OOM).
			if os.environ.get("BENCHDP_DEFER_METRICS") == "1":
				utility_metrics, privacy_metrics = {}, {}
			else:
				utility_metrics = compute_utility_metrics(predictions, references, utility_metric_names)
				privacy_metrics = compute_privacy_metrics(
					predictions,
					references,
					privacy_metric_names,
					seed=seed,
				)

			metrics_payload = {**utility_metrics, **privacy_metrics}
			# Headline epsilon = a SINGLE PLD composition of the whole mechanism:
			# retrieval(eps_r) (+) T x generation(token_eps). eps_g is already the total
			# generation budget (the generator calibrated token_epsilon via PLD so that the
			# T-token generation alone equals (eps_g, delta)-DP), so composing the two PLDs
			# once gives the true total (~5.3 at eps_r=0.5/eps_g=5.0/T=70). We deliberately
			# do NOT re-compose token_epsilon over T with basic/RDP/advanced bounds: that
			# double-counts the calibration and inflates the budget (DPRAG-04/05).
			eps_total_pld = pld_dp_rag_total(
				eps_retrieval=eps_r,
				token_epsilon=generator.token_epsilon,
				num_tokens=max_new_tokens,
				delta=delta,
			)

			accounting_payload = {
				"pld": {
					"epsilon": eps_total_pld,
					"delta": delta,
					"eps_retrieval": eps_r,
					"eps_generation_total": eps_g,
					"eps_generation_per_token": generator.token_epsilon,
					"num_tokens": max_new_tokens,
					"note": "Single PLD composition retrieval(eps_r) (+) T x generation(token_eps); headline epsilon.",
				},
				"naive_sum": {
					"epsilon": eps_r + eps_g,
					"note": "Loose union bound eps_r + eps_g; kept for comparison only.",
				},
			}

			payload = {
				"method": "dp_rag",
				"dataset": dataset_name,
				"llm": llm_model_name,
				"max_samples": ds_cfg.get("max_samples"),
				"retriever_type": retriever_type,
				"epsilon_retrieval": eps_r,
				"epsilon_generation": eps_g,
				"epsilon_total": eps_total_pld,
				"delta": delta,
				"privacy_accounting": accounting_payload,
				"metrics": metrics_payload,
				"n_queries": len(predictions),
				"timestamp": datetime.now(timezone.utc).isoformat(),
				"results": results,
			}

			out_name = f"eps_r{eps_r}_eps_g{eps_g}.json"
			out_path = output_base / out_name
			_save_results(out_path, payload)
			print(f"Saved: {out_path}")

			del generator, retriever, pipeline  # pipeline holds refs to both; drop it too
			free_memory()  # the shared model stays resident; release transient buffers only

		del dataset, indexer
		free_memory()  # release the dataset index before the next dataset

	# Release the shared generation model now that this LLM's grid is complete.
	del shared_model, shared_tokenizer
	free_memory()


def main(
	config_path: str = None,
	verbose_override: bool | None = None,
	verbose_max_examples_override: int | None = None,
) -> None:
	setup_method_logging("dp_rag")
	config = load_config(config_path)
	if verbose_override is not None:
		config["verbose"] = bool(verbose_override)
	if verbose_max_examples_override is not None:
		config["verbose_max_examples"] = int(verbose_max_examples_override)

	llm_configs = _iter_llm_configs(config)
	multi_llm = len(llm_configs) > 1
	for llm_cfg in llm_configs:
		llm_model_name = str(llm_cfg.get("model_name", ""))
		llm_label = _llm_label(llm_cfg)
		if multi_llm:
			print(f"\n##### LLM: {llm_model_name} #####")
		try:
			_run_for_llm(config, llm_model_name, llm_label, multi_llm)
		except OSError as exc:
			# A model we cannot load (e.g. gated HF repo without access, or an
			# offline/missing checkpoint) should not abort the whole multi-LLM
			# run; skip it and keep going with the remaining generators.
			print(f"\n[WARN] skipping LLM {llm_model_name}: {exc}")
		except (RuntimeError, MemoryError) as exc:
			# An OOM (CUDA or host) should not abort the whole multi-LLM run; free
			# memory (below) and skip this model. Non-OOM RuntimeErrors are re-raised.
			if not is_oom_error(exc):
				raise
			print(f"\n[WARN] OOM — skipping LLM {llm_model_name}: {exc}")
		free_memory()  # release the model before loading the next LLM


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run DP-RAG experiment grid.")
	parser.add_argument(
		"--config",
		type=str,
		default=str(PROJECT_ROOT / "experiments" / "configs" / "full" / "dp_rag.yaml"),
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
