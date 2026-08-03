"""Benchmark script for DPVoteRAG (Koga et al., arXiv:2412.04697).

Runs the DPVoteRAG pipeline (m voters, LD-Gumbel aggregation) over the shared
privacy axis crossed with the workload size: each cell fixes a target total epsilon
over Q questions and derives the per-query budget that reaches it. DPVoteRAG pays
per query, so the budget shrinks as Q grows. All parameters come from the YAML config.

Usage:
	python experiments/run_dpvoterag.py --config experiments/configs/full/dpvoterag.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import timezone
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

from core.device import (
	cuda_oom_warning,
	free_memory,
	is_oom_error,
	model_device_kwargs,
	resolve_device,
)
from experiments.index_cache import index_cache_dir, load_or_build_index
from experiments.llm_isolation import resolve_llm_execution
from experiments.common import (
    RecordingRetriever,
    workload_accounting_block,
    nonprivate_ranking,
    load_dataset_from_config,
    resolve_seeds,
    select_queries,
    build_chunker,
    chunking_fingerprint,
    corpus_fingerprint,
    select_datasets,
)
from experiments.mia_probe import mia_only_enabled, resolve_mia_params, run_mia_probes
from experiments.run_logging import log, setup_method_logging
from bench_datasets.dpr_corpus import corpus_available
from dp_methods.per_query.dpvoterag.generator import VoteRAGGenerator
from dp_methods.per_query.dpvoterag.pipeline import DPVoteRAGPipeline
from metrics.privacy import compute_privacy_metrics
from metrics.privacy_accounting_autodp import ld_gumbel_feasible, release_ceiling
from metrics.workload_accounting import DEFAULT_DELTA, budget_for_target
from metrics.utility import compute_utility_metrics
from vanilla.embedder import HFEmbedder, build_embedder, embedder_identity
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


def _cell_output_path(
	output_dir: Path,
	dataset_name: str,
	llm_label: str,
	multi_llm: bool,
	target: float,
	workload_q: int,
) -> Path:
	"""Where one (target, workload) cell writes its result.

	The workload size is part of the location because the same target is run at
	several stream lengths, each with its own budget and its own answers: with only
	the target in the name the last workload would keep overwriting the others.
	"""
	path = output_dir / dataset_name
	if multi_llm:
		path = path / llm_label
	return path / f"q{workload_q}" / f"target_{target:g}.json"


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
		default="experiments/configs/full/dpvoterag.yaml",
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
	embedder_cache_name = embedder_identity(embedder_model_name)

	dp_cfg = config.get("dpvoterag", {})
	# data_seed picks the corpus and the questions; run_seed drives the DP noise, so
	# a repetition varies the mechanism without resampling or rebuilding the index.
	data_seed, seed = resolve_seeds(config)
	m = int(dp_cfg.get("m", 30))
	k = int(dp_cfg.get("k", 1))
	epsilon_token_cfg = float(dp_cfg.get("epsilon_token", 2.0))
	# The shared protocol derives it from the swept budget; the paper pins it and
	# sweeps it separately. Both stay available, one per config set.
	epsilon_token_from_budget = bool(dp_cfg.get("epsilon_token_from_budget", False))
	delta_token = float(dp_cfg.get("delta_token", 1e-5))
	k_bar = dp_cfg.get("k_bar", None)
	k_bar = int(k_bar) if k_bar is not None else None
	_tmax_cfg = dp_cfg.get("tmax_absolute", None)
	tmax_absolute = int(_tmax_cfg) if _tmax_cfg is not None else None  # optional safety ceiling
	max_length = int(dp_cfg.get("max_length", 50))  # DPVOTE-09: fixed generation length
	# How many votes one answer may ask for. It is the count the budget is divided
	# across and the count the accounting charges, so it is fixed for the whole grid:
	# every cell releases answers of the same length and only the noise per vote moves.
	loop_cap = max_length if _tmax_cfg is None else min(max_length, int(_tmax_cfg))
	fail_mode = str(dp_cfg.get("fail_mode", "stop"))
	exclude_stopwords = bool(dp_cfg.get("exclude_stopwords", False))
	exclude_question_words = bool(dp_cfg.get("exclude_question_words", False))
	stopwords_path = dp_cfg.get("stopwords_path", None)
	# C2 (chat template, auto by default) + C3 (optional stop strings, OFF by default).
	use_chat_template = dp_cfg.get("use_chat_template", "auto")
	stop_strings = dp_cfg.get("stop_strings")
	# DPVOTE-01 — accountant composition: 'simple' (paper's max advanced/sequential)
	# or 'rdp' (reference runtime default). Default 'rdp' to preserve old behaviour.
	compose_method = str(dp_cfg.get("compose_method", "rdp"))

	# The sweep is the shared axis crossed with the workload size: `epsilon_targets`
	# are total privacy losses over a stream of `workload_queries` questions, the same
	# targets every method is run at. What each cell varies is how much of that total a
	# single question may spend, which is what the method's own budget knob controls.
	epsilon_targets = [float(x) for x in config.get("epsilon_targets", [1, 5, 10, 50])]
	workload_queries = [int(x) for x in config.get("workload_queries", [100])]
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
	privacy_metrics = list(config.get("metrics", {}).get("privacy", []))
	output_dir = Path(str(config.get("output_dir", "results/dpvoterag/")))
	output_dir.mkdir(parents=True, exist_ok=True)

	set_seed(seed)

	_kwargs = model_device_kwargs()
	_device = resolve_device()
	_ngpu = torch.cuda.device_count() if _device == "cuda" else 1
	print(f"Loading model/tokenizer... (device: {_device}, gpus: {_ngpu})")
	tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
	with cuda_oom_warning(llm_model_name, _device):
		model = AutoModelForCausalLM.from_pretrained(llm_model_name, **_kwargs)
		# Move to device unless accelerate already placed it (multi-GPU device_map).
		# The single-GPU path has no device_map, so .to() runs and either fits or
		# raises a clean CUDA OOM — never a silent CPU offload.
		if "device_map" not in _kwargs:
			model = model.to(_device)
	model.eval()

	chunker = build_chunker(config)
	embedder = build_embedder(embedder_model_name)
	reranker = IdentityReranker()
	prompt_builder = SimplePromptBuilder()
	postprocessor = IdentityPostprocessor()

	summary_rows: list[dict[str, Any]] = []

	for ds_cfg in select_datasets(_iter_dataset_configs(config)):
		dataset_name = str(ds_cfg.get("name", "natural_questions"))
		max_index_docs = ds_cfg.get("max_index_docs", 200)
		max_index_docs = int(max_index_docs) if max_index_docs is not None else None
		ds_max_queries = ds_cfg.get("max_test_queries", max_queries)
		ds_max_queries = int(ds_max_queries) if ds_max_queries is not None else None

		log(
			f"Dataset START: {dataset_name} "
			f"(max_index_docs={max_index_docs}, max_test_queries={ds_max_queries})"
		)

		indexer = FAISSIndexer(store_embeddings_matrix=False)
		retriever = RecordingRetriever(CosineSimilarityRetriever(indexer=indexer))

		try:
			dataset = load_dataset_from_config(
				ds_cfg, data_seed, default_max_queries=ds_max_queries
			)
			documents = dataset.get_documents()
			queries = dataset.get_queries()

			if not documents:
				raise ValueError(f"Dataset '{dataset_name}' returned no documents to index.")
			if not queries:
				raise ValueError(f"Dataset '{dataset_name}' returned no queries to evaluate.")
		except Exception as exc:
			log(f"Dataset SKIPPED: {dataset_name} — {type(exc).__name__}: {exc}")
			free_memory()
			continue

		# Standalone MIA mode: index only the member subset and probe it instead of
		# running the utility QA loop. A pure add-on — untouched when not MIA.
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
			max_docs=max_index_docs,
			embedder_model=embedder_cache_name,
			seed=data_seed,
			store_embeddings_matrix=False,
			loader_kwargs=corpus_fingerprint(ds_cfg, default_max_queries=ds_max_queries),
			**chunking_fingerprint(config),
		)
		index_built = False

		for workload_q in workload_queries:
			for target in epsilon_targets:
				# The per-query budget that composes to `target` over a stream of
				# `workload_q` questions. A per-query method has to spend less on each
				# question as the stream grows, so this is what the workload axis moves;
				# the questions actually answered are the utility sample and stay whatever
				# `max_test_queries` asks for.
				try:
					epsilon_total = budget_for_target(
						"dpvoterag",
						target_epsilon=target,
						queries=workload_q,
						delta=DEFAULT_DELTA,
						max_tokens=loop_cap,
						delta_token=delta_token,
					)
				except ValueError as exc:
					# A target this method cannot reach — its per-token delta alone may
					# already overrun it — is reported as absent rather than clamped to the
					# nearest budget, which would put a point on the axis the mechanism
					# never honoured.
					log(f"[{dataset_name}] SKIP target={target} Q={workload_q}: {exc}")
					continue
				# The cell's budget is the query's whole allowance. Pinning `epsilon_token`
				# independently makes it, not the budget, decide how many tokens the guard
				# allows: at a fixed 2.0 a query worth 2 releases one token and a query
				# worth 40 releases twenty, so the answers get longer along the sweep and
				# length is confounded with privacy. Splitting the budget across a fixed
				# answer length keeps every point releasing the same number of tokens, and
				# leaves the noise per vote as the only thing the sweep changes.
				epsilon_token = (
					epsilon_total / loop_cap if epsilon_token_from_budget else epsilon_token_cfg
				)
				# Whether the Limited-Domain Gumbel can release anything at all at this
				# budget. Below the threshold no histogram the m voters can produce clears
				# the bottom-out symbol, so every answer comes back empty — a fact about the
				# mechanism at this cell, which the record states instead of leaving a
				# reader to infer it from a zero score.
				ld_feasible, ld_threshold = ld_gumbel_feasible(
					epsilon_token=epsilon_token, delta_token=delta_token, m=m
				)
				log(
					f"[{dataset_name}] RUN START: target={target}, Q={workload_q}, "
					f"epsilon_total={epsilon_total:.4g}, epsilon_token={epsilon_token:.4g}, "
					f"m={m}, k={k}, ld_feasible={ld_feasible} (h_perp={ld_threshold:.3g})"
				)
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
					max_length=max_length,
					fail_mode=fail_mode,
					exclude_stopwords=exclude_stopwords,
					exclude_question_words=exclude_question_words,
					stopwords_path=stopwords_path,
					task_instruction=dataset.get_task_instruction(),
					use_chat_template=use_chat_template,
					stop_strings=stop_strings,
					compose_method=compose_method,
				)
				# Paper-faithful by default; the shared protocol reads whole documents so this
				# method sees the same text as the six that never truncated at "//".
				generator.truncate_at_double_slash = bool(
					dp_cfg.get("truncate_at_double_slash", True)
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
						build_fn=lambda p=pipeline: p.index(documents),
						cache_dir=cache_dir,
					)
					index_built = True

				# Standalone MIA: probe the live member index for this DP cell instead
				# of the utility loop; score_results computes the AUCs offline.
				if mia_only_enabled():
					k_ret = k_retrieved
					answer_fn = lambda q: pipeline.query(q, k=k_ret)
					mia_records = run_mia_probes(
						answer_fn, mia_split,
						anderson_doc_maxlen=mia_params["anderson_doc_maxlen"],
					)
					payload = {
						"method": "dpvoterag",
						"timestamp": datetime.now(timezone.utc).isoformat(),
						"dataset": dataset_name,
						"llm": llm_model_name,
						"epsilon_target": float(target),
						"workload_queries": int(workload_q),
						"epsilon_total": float(epsilon_total),
						"epsilon_token": float(epsilon_token),
						"ld_feasible": bool(ld_feasible),
						"ld_threshold": float(ld_threshold),
						"seed": seed,
						"mia_probes": mia_records,
					}
					output_path = _cell_output_path(
						output_dir, dataset_name, llm_label, multi_llm, target, workload_q
					)
					output_path.parent.mkdir(parents=True, exist_ok=True)
					with output_path.open("w", encoding="utf-8") as f:
						json.dump(payload, f, indent=2, ensure_ascii=False)
					log(
						f"[{dataset_name} target={target} Q={workload_q}] MIA: wrote "
						f"{len(mia_records)} probe records to {output_path}"
					)
					del generator, pipeline
					free_memory()
					continue

				predictions: list[str] = []
				references: list[str] = []
				results: list[dict[str, str]] = []
				dp_spent_per_query: list[dict[str, float]] = []
				# Releases the run actually made, for the spent-vs-budget diagnostic.
				tokens_released = 0
				exit_statuses: list[int] = []

				query_times: list[float] = []
				selected_queries = select_queries(queries, max_queries=ds_max_queries)
				verbose_count = 0
				# Logging-only noise-impact aggregate over the run (generator.last_run_stats).
				n_tokens_sum = 0
				n_flipped_sum = 0
				t_run_start = time.time()
				total_q = len(selected_queries)

				for q_idx, sample in enumerate(selected_queries, start=1):
					question = str(sample.get("query", "")).strip()
					reference = str(sample.get("answer", "")).strip()
					answers = [str(a).strip() for a in (sample.get("answers") or [reference]) if str(a).strip()] or [reference]
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
							"query_id": sample.get("query_id", ""),
							# Set-valued gold (Movielens held-out favorites) drives hit_rate;
							# without it the metric scores against one title and reads as zero.
							"favorites": [str(f) for f in (sample.get("favorites") or [])],
							"retrieved_ids": list(retriever.last_retrieved_ids),
							"nonprivate_retrieved_ids": nonprivate_ranking(
								indexer, embedder.embed_query(question), k_retrieved
							),
							"prediction": str(answer),
							"reference": reference,
							"answers": answers,
							"exit_status": exit_status,
							"dp_spent": dp_spent,
						}
					)
					dp_spent_per_query.append(dp_spent)
					tokens_released += int(run_stats.get("n_private_tokens", 0))
					exit_statuses.append(exit_status)
					query_times.append(float(elapsed))
					n_tokens_sum += int(run_stats.get("tokens_generated", 0))
					n_flipped_sum += int(run_stats.get("n_noise_flipped", 0))
					if q_idx % 10 == 0:
						log(f"[{dataset_name} target={target} Q={workload_q}] heartbeat: query {q_idx}/{total_q}")

					free_memory()  # DPVoteRAG runs m voters per token: free memory each query

				_flip_pct = (100.0 * n_flipped_sum / n_tokens_sum) if n_tokens_sum else 0.0
				_n_ok = sum(1 for s in exit_statuses if s == 0)
				_n_halt = sum(1 for s in exit_statuses if s == 1)
				_n_budget = sum(1 for s in exit_statuses if s == 2)
				log(
					f"[{dataset_name}] RUN END: target={target}, Q={workload_q}, "
					f"epsilon_total={epsilon_total:.4g} — "
					f"{len(predictions)} queries in {time.time() - t_run_start:.1f}s | "
					f"LD-Gumbel flipped {n_flipped_sum}/{n_tokens_sum} tokens ({_flip_pct:.1f}%); "
					f"exit_status ok={_n_ok}/halt={_n_halt}/budget={_n_budget}"
				)

				# When deferred (orchestrated run), skip in-process metrics; score_results.py
				# computes them later in a clean process. Avoids loading BERTScore/perplexity
				# models on top of the generator (OOM).
				if os.environ.get("BENCHDP_DEFER_METRICS") == "1":
					util, priv = {}, {}
				else:
					util = compute_utility_metrics(predictions, references, utility_metrics)
					priv = compute_privacy_metrics(predictions, references, privacy_metrics, seed=seed)

				time_per_query = _safe_float(np.mean(query_times) if query_times else 0.0)
				match_acc = _safe_float(util.get("match_accuracy", 0.0))
				contains_acc = _safe_float(util.get("contains_accuracy", 0.0))
				total_eps_spent = _safe_float(sum(item.get("epsilon", 0.0) for item in dp_spent_per_query))
				total_delta_spent = _safe_float(sum(item.get("delta", 0.0) for item in dp_spent_per_query))

				payload: dict[str, Any] = {
					"method": "dpvoterag",
					"timestamp": datetime.now(timezone.utc).isoformat(),
					"dataset": dataset_name,
					"llm": llm_model_name,
					"max_index_docs": max_index_docs,
					# The cell's coordinates on the shared grid: the total privacy loss this
					# point targets, the stream length it is priced for, and the per-query
					# budget that reaches the one over the other.
					"epsilon_target": float(target),
					"workload_queries": int(workload_q),
					# The utility sample: how many questions this cell actually answered,
					# set by `max_test_queries` and independent of Q. Both are recorded
					# because they answer different questions about the same run.
					"queries_answered": len(predictions),
					"epsilon_total": float(epsilon_total),
					# Whether a vote can clear the bottom-out symbol at all here, and the
					# vote count it has to clear. Where it cannot, every prediction is empty
					# by construction, and a zero score is the mechanism declining to speak
					# rather than the generator answering badly.
					"ld_feasible": bool(ld_feasible),
					"ld_threshold": float(ld_threshold),
					"delta_total": delta_total,
					"m": m,
					"k_retrieved": k_retrieved,
					**workload_accounting_block(
						# Q is the declared workload the guarantee is priced for, not the number
						# of questions this cell happened to answer: those are the utility
						# sample, and letting them set the axis would tie the privacy figure to
						# how many questions were sampled.
						"dpvoterag", queries=workload_q, run_seed=seed,
						epsilon_token=epsilon_token,
						# The decoding length is not the ceiling: the generator's budget
						# guard refuses the next vote long before it, and how far it lets
						# the loop run is exactly what `epsilon_total` controls. Charging
						# `max_length` reported the same epsilon for every point of the
						# sweep, flattening the method against its own privacy knob.
						max_tokens=release_ceiling(
							loop_cap=loop_cap,
							target_epsilon=float(epsilon_total),
							target_delta=delta_total,
							epsilon_token=epsilon_token,
							delta_token=delta_token,
							m=m, k=k, k_bar=k_bar,
							compose_method=compose_method,
						),
						delta_token=delta_token,
						realised={"max_tokens": tokens_released},
					),
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
					"results": results,
					"metrics": {**util, **priv},
					"match_accuracy": match_acc,
					"contains_accuracy": contains_acc,
					"time_per_query_sec": time_per_query,
				}

				output_path = _cell_output_path(
					output_dir, dataset_name, llm_label, multi_llm, target, workload_q
				)
				output_path.parent.mkdir(parents=True, exist_ok=True)
				with output_path.open("w", encoding="utf-8") as f:
					json.dump(payload, f, indent=2, ensure_ascii=False)

				summary_rows.append(
					{
						"dataset": dataset_name,
						"llm": llm_model_name,
						"epsilon_target": float(target),
						"workload_queries": int(workload_q),
						"epsilon_total": float(epsilon_total),
						"ld_feasible": bool(ld_feasible),
						"match_accuracy": match_acc,
						"contains_accuracy": contains_acc,
						"eps_spent_total": total_eps_spent,
						"time_per_query_sec": time_per_query,
						"output": str(output_path),
					}
				)

				del generator, pipeline
				free_memory()  # release memory between grid cells

		del dataset, indexer, retriever
		free_memory()  # release the dataset index before the next dataset

	del model, tokenizer, embedder
	free_memory()

	return summary_rows


def main() -> None:
	setup_method_logging("dpvoterag")
	args = build_parser().parse_args()
	config = load_config(args.config)

	llm_configs = _iter_llm_configs(config)
	multi_llm = len(llm_configs) > 1
	# Run each LLM in its own subprocess so GPU memory is fully reclaimed between
	# models; returns None in the orchestrator once all LLMs have run as children.
	todo = resolve_llm_execution(llm_configs, method_name="dpvoterag")
	if todo is None:
		return
	summary_rows: list[dict[str, Any]] = []
	for _idx, llm_cfg in todo:
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
		except (RuntimeError, MemoryError) as exc:
			# An OOM (CUDA or host) should not abort the whole multi-LLM run; free
			# memory (below) and skip this model. Non-OOM RuntimeErrors are re-raised.
			if not is_oom_error(exc):
				raise
			print(f"\n[WARN] OOM — skipping LLM {llm_model_name}: {exc}")
		# free_memory() runs OUTSIDE the except on purpose: while `exc` is alive
		# its traceback pins _run_for_llm's frame and locals (model, embedder,
		# generator, ...), so a gc.collect() inside the handler frees nothing.
		# Reached on the success path too — an extra empty_cache is harmless.
		free_memory()  # release the (possibly failed) model before the next LLM

	print("\nResumen DPVoteRAG:")
	for row in summary_rows:
		print(
			f"- [{row['dataset']} | {row['llm']}] target={row['epsilon_target']} "
			f"Q={row['workload_queries']} (eps_query={row['epsilon_total']:.4g}, "
			f"ld_feasible={row['ld_feasible']}): "
			f"match_accuracy={row['match_accuracy']:.4f}, "
			f"contains_accuracy={row['contains_accuracy']:.4f}, "
			f"eps_spent_total={row['eps_spent_total']:.4f}, "
			f"time_per_query_sec={row['time_per_query_sec']:.4f}, "
			f"out={row['output']}"
		)


if __name__ == "__main__":
	main()
