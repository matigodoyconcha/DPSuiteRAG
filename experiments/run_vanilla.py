"""Benchmark script for vanilla RAG (non-private baseline).

Runs the vanilla RAG pipeline on multiple datasets and collects utility and
privacy leakage metrics for comparison with differentially private methods.

Usage:
    python experiments/run_vanilla.py --config experiments/configs/full/vanilla.yaml --dataset squad
"""

import argparse
import json
import os
import random
import sys
from datetime import timezone
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.common import (
	dataset_loader_kwargs,
	iter_dataset_configs,
	iter_llm_configs,
	llm_label,
	select_datasets,
)
from experiments.index_cache import index_cache_dir, load_or_build_index
from experiments.mia_probe import mia_only_enabled, resolve_mia_params, run_mia_probes
from bench_llms import load_llm
from core.pipeline import RAGPipeline
from bench_datasets.base_dataset import BaseDataset
from bench_datasets.squad import SQuADDataset
from bench_datasets.chatdoctor import ChatDoctorDataset
from bench_datasets.natural_questions import NaturalQuestionsDataset
from bench_datasets.triviaqa import TriviaQADataset
from bench_datasets.hotpotqa import HotpotQADataset
from bench_datasets.medical_dirichlet_phi3 import MedicalDirichletPhi3Dataset
from bench_datasets.movielens import MovielensDataset
from bench_datasets.searchqa import SearchQADataset
from bench_datasets.dpr_corpus import corpus_available
from vanilla.chunker import FixedSizeChunker
from vanilla.embedder import HFEmbedder
from vanilla.indexer import FAISSIndexer
from vanilla.postprocessor import IdentityPostprocessor
from vanilla.prompt_builder import SimplePromptBuilder
from vanilla.reranker import IdentityReranker
from vanilla.retriever import CosineSimilarityRetriever


def set_seed(seed: int) -> None:
	"""Set random seeds for reproducibility.

	Args:
		seed: Random seed value.
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def build_demo_documents() -> list[str]:
	return [
		(
			"Retrieval-Augmented Generation (RAG) combines a document retriever and "
			"a generative model to answer questions based on external context. "
			"A classic pipeline includes chunking, embeddings, indexing and top-k retrieval."
		),
		(
			"FAISS is an efficient library for vector similarity search. "
			"With IndexFlatIP and normalized embeddings, the inner product approximates cosine similarity."
		),
		(
			"Differential Privacy introduces noise mechanisms to limit the exposure "
			"of sensitive information about individuals in the data."
		),
	]


def build_demo_queries() -> list[str]:
	return [
		"What components does a RAG pipeline have?",
		"Why is FAISS used in RAG systems?",
		"What does differential privacy seek to protect?",
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
		raise ValueError("The config YAML file must be an object/dict.")
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
		raise ValueError(f"File {path} must contain a JSON list.")
	return data


def load_custom_documents(path: str) -> list[str]:
	file_path = Path(path)
	if not file_path.exists():
		raise FileNotFoundError(f"Documents file does not exist: {file_path}")

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
		raise FileNotFoundError(f"Queries file does not exist: {file_path}")

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


# Fallback prompt for sources without a dataset class (demo / custom docs).
_DEFAULT_PROMPT_TEMPLATE = "Context: {context}\n\nQuestion: {query}\nAnswer:"


# Dataset classes selectable by ``dataset.name``. demo / custom-file sources are
# handled separately (they have no dataset object and no MIA split).
_DATASET_CLASSES = {
	"squad": SQuADDataset,
	"chatdoctor": ChatDoctorDataset,
	"natural_questions": NaturalQuestionsDataset,
	"triviaqa": TriviaQADataset,
	"hotpotqa": HotpotQADataset,
	"medical_dirichlet_phi3": MedicalDirichletPhi3Dataset,
	"movielens": MovielensDataset,
	"searchqa": SearchQADataset,
}


def make_dataset(dataset_cfg: dict[str, Any], seed: int) -> BaseDataset | None:
	"""Construct the dataset object for ``dataset_cfg["name"]``.

	Returns ``None`` for the demo / custom-file sources, which have no dataset class
	and no MIA split. Loader arguments come from the config entry itself, so the
	baseline reads the same corpus, the same query set and the same filters as the DP
	methods it is the reference condition for.
	"""
	cls = _DATASET_CLASSES.get(str(dataset_cfg.get("name", "")).lower())
	if cls is None:
		return None
	kwargs = dataset_loader_kwargs(cls, dataset_cfg)
	kwargs.setdefault("random_seed", int(seed))
	return cls(**kwargs)


def load_data_sources(
	dataset_cfg: dict[str, Any], seed: int
) -> tuple[list[str], list[dict], str]:
	dataset_name = str(dataset_cfg.get("name", "")).lower()

	docs_path = dataset_cfg.get("docs_path")
	queries_path = dataset_cfg.get("queries_path")
	if docs_path and queries_path:
		documents = load_custom_documents(str(docs_path))
		queries = load_custom_queries(str(queries_path))
		return documents, queries, _DEFAULT_PROMPT_TEMPLATE

	if dataset_name == "demo":
		docs = build_demo_documents()
		raw_queries = build_demo_queries()
		return docs, [{"query": q, "answer": ""} for q in raw_queries], _DEFAULT_PROMPT_TEMPLATE

	dataset = make_dataset(dataset_cfg, seed)
	if dataset is None:
		raise ValueError(
			f"Dataset '{dataset_name}' not supported in run_vanilla.py. Use one of "
			f"'demo', {', '.join(sorted(_DATASET_CLASSES))} or provide docs_path/queries_path."
		)
	dataset.load()
	return dataset.get_documents(), dataset.get_queries(), dataset.get_prompt_template()


def run_experiment(
	config: dict[str, Any],
	dataset_cfg: dict[str, Any],
	llm_cfg: dict[str, Any],
) -> dict[str, Any]:
	seed = int(config["random_seed"])
	set_seed(seed)

	dataset_name = str(dataset_cfg.get("name", "demo"))
	llm = load_llm(llm_cfg)
	llm_config = llm.get_config()
	llm_model = str(llm_config["model_name"])
	embedder_model = str(config["models"]["embedder"])
	chunk_size = int(config["pipeline"]["chunk_size"])
	chunk_overlap = int(config["pipeline"]["chunk_overlap"])
	k = int(config["pipeline"]["k_retrieved"])
	max_new_tokens = int(config["generation"]["max_new_tokens"])
	# C2 (chat template, auto by default) + C3 (optional stop strings, OFF by default).
	gen_cfg = config.get("generation", {})
	use_chat_template = gen_cfg.get("use_chat_template", "auto")
	stop_strings = gen_cfg.get("stop_strings")

	# Standalone MIA run: index only the member corpus and probe members vs
	# held-out non-members (see experiments/mia_probe.py). The utility QA loop is
	# skipped entirely, so no test queries are needed.
	mia_split = None
	if mia_only_enabled():
		dataset_obj = make_dataset(dataset_cfg, seed)
		if dataset_obj is None:
			raise ValueError("MIA mode requires a dataset class (not the demo / custom-file sources).")
		dataset_obj.load()
		mia_params = resolve_mia_params(config)
		mia_split = dataset_obj.get_mia_split(
			n_per_class=mia_params["n_per_class"],
			index_size=mia_params["index_size"],
			seed=mia_params["seed"],
		)
		if mia_split is None:
			raise ValueError(f"Dataset '{dataset_name}' has no MIA member/non-member split.")
		documents = mia_split["index_documents"]
		queries = []
		prompt_template = dataset_obj.get_prompt_template()
	else:
		documents, queries, prompt_template = load_data_sources(dataset_cfg, seed)
	if not documents:
		raise ValueError("No documents to index.")
	if not mia_only_enabled() and not queries:
		raise ValueError("No queries to evaluate.")

	chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_overlap)
	embedder = HFEmbedder(model_name=embedder_model)
	indexer = FAISSIndexer()
	retriever = CosineSimilarityRetriever(indexer=indexer)
	reranker = IdentityReranker()
	prompt_builder = SimplePromptBuilder(template=prompt_template)
	generator = llm.build_generator(
		max_new_tokens=max_new_tokens,
		use_chat_template=use_chat_template,
		stop_strings=stop_strings,
	)
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
		f"- dataset: {dataset_name}\n"
		f"- llm: {llm_config['name']} ({llm_model})\n"
		f"- embedder: {embedder_model}\n"
		f"- chunk_size: {chunk_size}\n"
		f"- chunk_overlap: {chunk_overlap}\n"
		f"- index_docs: {len(documents)}\n"
		f"- test_queries: {len(queries)}\n"
		f"- k: {k}\n"
		f"- max_new_tokens: {max_new_tokens}"
	)

	# NQ and TriviaQA retrieve from the same DPR corpus, so when it is present
	# they share one index directory instead of building it twice.
	corpus_id = (
		"dpr_psgs_w100"
		if dataset_name.lower() in {"natural_questions", "triviaqa"} and corpus_available()
		else None
	)
	cache_dir = index_cache_dir(
		dataset_name=dataset_name,
		corpus_id=corpus_id,
		max_docs=dataset_cfg.get("max_index_docs"),
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

	if mia_only_enabled():
		mia_params = resolve_mia_params(config)
		answer_fn = lambda q: pipeline.query(q, k=k)
		mia_records = run_mia_probes(
			answer_fn, mia_split, anderson_doc_maxlen=mia_params["anderson_doc_maxlen"]
		)
		return {
			"method": "vanilla",
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"dataset": dataset_name,
			"dataset_config": dataset_cfg,
			"llm": llm_config,
			"models": config["models"],
			"pipeline": config["pipeline"],
			"generation": config["generation"],
			"random_seed": seed,
			"seed": seed,
			"num_index_documents": len(documents),
			"mia_probes": mia_records,
		}

	results = []
	for row in queries:
		query = str(row.get("query", "")).strip()
		if not query:
			continue

		reference = str(row.get("answer", "")).strip()
		answers = [str(a).strip() for a in (row.get("answers") or [reference]) if str(a).strip()] or [reference]

		# Single pass: inline the pipeline to reuse retrieved_chunks for logging
		query_vector = embedder.embed_query(query)
		retrieved_chunks = retriever.retrieve(query_vector, k=k)
		reranked_chunks = reranker.rerank(query, retrieved_chunks)
		prompt = prompt_builder.build(query, reranked_chunks)
		raw_answer = generator.generate(prompt)
		answer = postprocessor.process(raw_answer)

		# ``prediction`` is the key the scorer reads; every method writes it under
		# that name so a single scoring path covers the baseline and the DP methods.
		result = {
			"query": query,
			"prediction": answer,
			"reference": reference,
			"answers": answers,
			"retrieved_chunks": retrieved_chunks,
			"retrieved_ids": [chunk.get("id") for chunk in retrieved_chunks],
		}
		# Set-valued gold (Movielens held-out favorites) drives hit_rate.
		if row.get("favorites"):
			result["favorites"] = [str(f) for f in row["favorites"]]
		results.append(result)

		print("=" * 30)
		print(f"Question: {query}")
		print("-" * 30)
		print(f"RETRIEVED CONTEXT (k={k}):")
		for chunk in retrieved_chunks:
			print(chunk["text"][:200])
		print("-" * 30)
		print(f"Answer: {answer}")
		if reference:
			print(f"Reference: {reference}")
		print()


	return {
		"method": "vanilla",
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"dataset": dataset_name,
		"dataset_config": dataset_cfg,
		"llm": llm_config,
		"models": config["models"],
		"pipeline": config["pipeline"],
		"generation": config["generation"],
		"random_seed": seed,
		"seed": seed,
		"num_index_documents": len(documents),
		"num_test_queries": len(results),
		"n_queries": len(results),
		"predictions": [r["prediction"] for r in results],
		"references": [r["reference"] for r in results],
		"results": results,
	}


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Runner configurable de RAG Vanilla (dataset, modelos y splits)"
	)
	parser.add_argument(
		"--config",
		type=str,
		default="experiments/configs/full/vanilla.yaml",
		help="Path to configuration YAML",
	)
	parser.add_argument(
		"--dataset",
		type=str,
		default=None,
		help="Override dataset (demo|squad|chatdoctor|natural_questions|triviaqa|hotpotqa)",
	)
	parser.add_argument("--llm-name", type=str, default=None, help="Override LLM type (e.g. huggingface)")
	parser.add_argument("--llm-model", type=str, default=None, help="Override LLM model")
	parser.add_argument("--embedder-model", type=str, default=None, help="Override embedding model")
	parser.add_argument("--k", type=int, default=None, help="Override top-k retrieval")
	parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max new tokens")
	parser.add_argument("--seed", type=int, default=None, help="Override random seed")
	parser.add_argument("--output", type=str, default=None, help="Override JSON output path")

	parser.add_argument("--index-split", type=str, default=None, help="Split used for indexing")
	parser.add_argument("--test-split", type=str, default=None, help="Split used for evaluation")
	parser.add_argument("--max-index-docs", type=int, default=None, help="Maximum number of indexed docs")
	parser.add_argument("--max-test-queries", type=int, default=None, help="Maximum number of test queries")
	parser.add_argument("--chunk-size", type=int, default=None, help="Override chunk size in words")
	parser.add_argument("--chunk-overlap", type=int, default=None, help="Override chunk overlap in words")
	parser.add_argument(
		"--docs-path",
		type=str,
		default=None,
		help="Optional path to custom documents (.txt/.json/.jsonl)",
	)
	parser.add_argument(
		"--queries-path",
		type=str,
		default=None,
		help="Optional path to custom queries (.json/.jsonl with query and answer fields)",
	)
	parser.add_argument(
		"--mia-only",
		action="store_true",
		help="Run the standalone membership inference attack (S2MIA + Anderson Yes/No) "
		"instead of the utility loop: index only a member subset and probe members vs "
		"held-out non-members.",
	)
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	if args.mia_only:
		os.environ["BENCHDP_MIA_ONLY"] = "1"

	config = load_yaml_config(args.config)
	config = apply_cli_overrides(config, args)

	dataset_cfgs = select_datasets(iter_dataset_configs(config))
	if not dataset_cfgs:
		print("No datasets selected; nothing to run.")
		return
	llm_cfgs = iter_llm_configs(config)
	fname = "mia.json" if mia_only_enabled() else "results.json"

	# The orchestrator (experiments/runner.py) redirects output under the run
	# directory; a standalone invocation keeps the config's own output_path.
	out_dir = os.environ.get("BENCHDP_OUTPUT_DIR") or config.get("output_dir")
	single_cell = len(dataset_cfgs) == 1 and len(llm_cfgs) == 1

	for ds_cfg in dataset_cfgs:
		dataset_name = str(ds_cfg.get("name", "demo"))
		for llm_cfg in llm_cfgs:
			if out_dir:
				parts = [Path(str(out_dir)), dataset_name]
				if len(llm_cfgs) > 1:
					parts.append(llm_label(llm_cfg))
				output_path = Path(*parts) / fname
			elif single_cell:
				output_path = Path(str(config["output_path"]))
			else:
				base = Path(str(config["output_path"]))
				output_path = base.parent / dataset_name / llm_label(llm_cfg) / base.name
			output_path.parent.mkdir(parents=True, exist_ok=True)

			payload = run_experiment(config, ds_cfg, llm_cfg)
			with output_path.open("w", encoding="utf-8") as f:
				json.dump(payload, f, indent=2, ensure_ascii=False)

			print(f"Resultados guardados en: {output_path}")


if __name__ == "__main__":
	main()
