"""Benchmark script for DP-KSA (Tang et al., arXiv:2602.14374).

Runs the DP-KSA pipeline (sample-and-aggregate over the top-N retrieved documents,
FindBestK + TopKWithPTR keyword release, keyword-conditioned final generation) over
a sweep of epsilon values. epsilon is an INPUT budget: the FindBestK Gumbel scale
and the PTR noise are calibrated to each target (epsilon, delta).

Usage:
    python experiments/run_dpksa.py --config experiments/configs/dpksa.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import UTC, datetime
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
from bench_datasets.medical_dirichlet_phi3 import MedicalDirichletPhi3Dataset
from bench_datasets.natural_questions import NaturalQuestionsDataset
from bench_datasets.squad import SQuADDataset
from bench_datasets.triviaqa import TriviaQADataset
from dp_methods.per_query.dpksa.generator import DPKSAGenerator
from dp_methods.per_query.dpksa.pipeline import DPKSAPipeline
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
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("The configuration YAML must be a dict.")
    return config


def load_dataset(name: str, seed: int, max_queries: int | None) -> Any:
    key = str(name).lower()
    common_kwargs = {"max_index_docs": 200, "max_test_queries": max_queries, "random_seed": seed}

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
            f"Dataset '{name}' not supported for DP-KSA. "
            "Use natural_questions, triviaqa, hotpotqa, squad, chatdoctor or medical_dirichlet_phi3."
        )

    dataset.load()
    return dataset


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(out) or np.isinf(out):
        return default
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DP-KSA experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/dpksa.yaml",
        help="Path to DP-KSA configuration YAML.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print outputs per query (question, keywords, answer and reference).",
    )
    parser.add_argument(
        "--verbose-max-examples",
        type=int,
        default=None,
        help="Maximum number of queries to print in verbose mode. Prints all by default.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    dataset_name = str(config.get("dataset", {}).get("name", "natural_questions"))
    llm_cfg = config.get("llm", {})
    llm_model_name = str(llm_cfg.get("model_name", "distilgpt2"))
    embedder_model_name = str(config.get("models", {}).get("embedder", "sentence-transformers/all-MiniLM-L6-v2"))

    dp_cfg = config.get("dpksa", {})
    seed = int(dp_cfg.get("random_seed", 42))
    n_docs = int(dp_cfg.get("n_docs", 80))
    max_answer_tokens = int(dp_cfg.get("max_answer_tokens", 16))
    max_final_tokens = int(dp_cfg.get("max_final_tokens", 32))
    k_min = int(dp_cfg.get("k_min", 1))
    k_max = dp_cfg.get("k_max", None)
    k_max = int(k_max) if k_max is not None else None
    ptr_delta_fraction = float(dp_cfg.get("ptr_delta_fraction", 0.5))
    ptr_noise_coupling = float(dp_cfg.get("ptr_noise_coupling", 1.0))
    lowercase = bool(dp_cfg.get("lowercase", True))
    exclude_stopwords = bool(dp_cfg.get("exclude_stopwords", False))

    epsilon_values = [float(v) for v in config.get("epsilon_values", [1, 2, 3, 5, 8])]
    delta = float(config.get("delta", 1e-4))
    k_retrieved = int(config.get("k_retrieved", n_docs))
    max_queries = config.get("max_queries", 20)
    max_queries = int(max_queries) if max_queries is not None else None
    verbose = bool(config.get("verbose", False)) if args.verbose is None else bool(args.verbose)
    verbose_max_examples = (
        int(args.verbose_max_examples)
        if args.verbose_max_examples is not None
        else config.get("verbose_max_examples")
    )
    verbose_max_examples = int(verbose_max_examples) if verbose_max_examples is not None else None

    utility_metrics = list(config.get("metrics", {}).get("utility", ["match_accuracy", "rouge_l"]))
    privacy_metrics = list(config.get("metrics", {}).get("privacy", ["mia_auc_bleu"]))
    output_dir = Path(str(config.get("output_dir", "results/dpksa/")))
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
    indexer = FAISSIndexer()
    retriever = CosineSimilarityRetriever(indexer=indexer)
    reranker = IdentityReranker()
    prompt_builder = SimplePromptBuilder()
    postprocessor = IdentityPostprocessor()

    dataset = load_dataset(dataset_name, seed=seed, max_queries=max_queries)
    documents = dataset.get_documents()
    queries = dataset.get_queries()
    private_entities = dataset.get_private_entities()

    if not documents:
        raise ValueError("Dataset returned no documents to index.")
    if not queries:
        raise ValueError("Dataset returned no queries to evaluate.")

    cache_dir = index_cache_dir(
        dataset_name=dataset_name,
        max_docs=None,
        embedder_model=embedder_model_name,
        chunk_size=200,
        chunk_overlap=20,
        seed=seed,
    )
    index_built = False
    summary_rows: list[dict[str, Any]] = []

    for epsilon in epsilon_values:
        generator = DPKSAGenerator(
            model=model,
            tokenizer=tokenizer,
            target_epsilon=float(epsilon),
            target_delta=delta,
            n_docs=n_docs,
            max_answer_tokens=max_answer_tokens,
            max_final_tokens=max_final_tokens,
            k_min=k_min,
            k_max=k_max,
            ptr_delta_fraction=ptr_delta_fraction,
            ptr_noise_coupling=ptr_noise_coupling,
            lowercase=lowercase,
            exclude_stopwords=exclude_stopwords,
            random_seed=seed,
        )

        pipeline = DPKSAPipeline(
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
        results: list[dict[str, Any]] = []
        ptr_pass_count = 0
        query_times: list[float] = []
        selected_queries = queries[:max_queries] if max_queries is not None else queries
        verbose_count = 0

        for sample in selected_queries:
            question = str(sample.get("query", "")).strip()
            reference = str(sample.get("answer", "")).strip()
            if not question:
                continue

            t0 = time.time()
            answer = pipeline.query(question, k=k_retrieved)
            elapsed = time.time() - t0

            stats = dict(generator.last_run_stats)
            ptr_pass_count += int(bool(stats.get("ptr_passed", False)))

            predictions.append(str(answer))
            references.append(reference)
            results.append(
                {
                    "query": question,
                    "prediction": str(answer),
                    "reference": reference,
                    "k_hat": int(stats.get("k_hat", 0)),
                    "ptr_passed": bool(stats.get("ptr_passed", False)),
                    "keywords": list(stats.get("keywords", [])),
                }
            )
            query_times.append(float(elapsed))

            if verbose and (verbose_max_examples is None or verbose_count < verbose_max_examples):
                print("=" * 30)
                print(f"Question: {question}")
                print(f"Keywords (k_hat={stats.get('k_hat')}, ptr_passed={stats.get('ptr_passed')}): "
                      f"{stats.get('keywords')}")
                print(f"Answer: {answer}")
                if reference:
                    print(f"Reference: {reference}")
                verbose_count += 1

            free_memory()  # DP-KSA runs N generations per query: free GPU/host memory each query

        util = compute_utility_metrics(predictions, references, utility_metrics)
        priv = compute_privacy_metrics(predictions, references, private_entities, privacy_metrics, seed=seed)

        time_per_query = _safe_float(np.mean(query_times) if query_times else 0.0)
        match_acc = _safe_float(util.get("match_accuracy", 0.0))
        rouge_l = _safe_float(util.get("rouge_l", 0.0))
        mia_auc = _safe_float(priv.get("mia_auc_bleu", 0.5))
        ptr_pass_rate = _safe_float(ptr_pass_count / len(results)) if results else 0.0

        payload: dict[str, Any] = {
            "method": "dpksa",
            "timestamp": datetime.now(UTC).isoformat(),
            "dataset": dataset_name,
            "epsilon": float(epsilon),
            "delta": delta,
            "n_docs": n_docs,
            "calibration": generator.calibration,
            "ptr_pass_rate": ptr_pass_rate,
            "predictions": predictions,
            "references": references,
            "results": results,
            "match_accuracy": match_acc,
            "rouge_l": rouge_l,
            "mia_auc": mia_auc,
            "time_per_query_sec": time_per_query,
        }

        output_path = output_dir / dataset_name / f"epsilon_{epsilon}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        summary_rows.append(
            {
                "epsilon": float(epsilon),
                "match_accuracy": match_acc,
                "rouge_l": rouge_l,
                "mia_auc": mia_auc,
                "ptr_pass_rate": ptr_pass_rate,
                "time_per_query_sec": time_per_query,
                "output": str(output_path),
            }
        )

        del generator, pipeline
        free_memory()  # release memory between epsilon runs

    del model, tokenizer
    free_memory()

    print("\nDP-KSA summary:")
    for row in summary_rows:
        print(
            f"- eps={row['epsilon']}: "
            f"match_accuracy={row['match_accuracy']:.4f}, "
            f"rouge_l={row['rouge_l']:.4f}, "
            f"mia_auc={row['mia_auc']:.4f}, "
            f"ptr_pass_rate={row['ptr_pass_rate']:.2f}, "
            f"time_per_query_sec={row['time_per_query_sec']:.4f}, "
            f"out={row['output']}"
        )


if __name__ == "__main__":
    main()
