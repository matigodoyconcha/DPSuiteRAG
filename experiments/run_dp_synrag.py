"""Benchmark script for DP-SynRAG (index-time differential privacy via synthetic corpus).

Builds a differentially private synthetic document corpus once (Stages 1a–2e from the
paper), then serves all queries using a vanilla RAG pipeline over that corpus.

Reference: Anonymous et al., "DP-SynRAG: Differentially Private RAG via Synthetic
Document Generation", arXiv:2510.06719v2 (2025).

Usage:
    python experiments/run_dp_synrag.py --config experiments/configs/dp_synrag.yaml
    python experiments/run_dp_synrag.py --config experiments/configs/dp_synrag.yaml --verbose
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import random
import shutil
import sys
import tempfile
from datetime import UTC, datetime
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
from bench_datasets.medical_dirichlet_phi3 import MedicalDirichletPhi3Dataset
from bench_datasets.movielens import MovielensDataset
from bench_datasets.natural_questions import NaturalQuestionsDataset
from bench_datasets.searchqa import SearchQADataset
from bench_datasets.squad import SQuADDataset
from bench_datasets.triviaqa import TriviaQADataset
from core.pipeline import RAGPipeline
from dp_methods.offline.dp_synrag import (
    DPClusterRetriever,
    DPHistogramClusterer,
    DPSynRAGIndexer,
    DPSyntheticGenerator,
    KeywordExtractor,
    SelfFilter,
)
from metrics.privacy import compute_privacy_metrics
from metrics.privacy_accounting_zcdp import print_zcdp_analysis, zcdp_dp_synrag
from metrics.utility import compute_utility_metrics
from vanilla.chunker import NoChunker
from vanilla.embedder import HFEmbedder
from vanilla.generator import HFGenerator
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


def load_config(config_path: str | None) -> dict[str, Any]:
    default_path = PROJECT_ROOT / "experiments" / "configs" / "dp_synrag.yaml"
    cfg_path = Path(config_path) if config_path else default_path
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("DP-SynRAG config YAML must define a dictionary.")
    config.setdefault("verbose", False)
    return config


def _resolve_dataset(config: dict[str, Any]):
    dataset_cfg = config.get("dataset", {})
    name = str(dataset_cfg.get("name", "searchqa")).lower()
    max_samples = dataset_cfg.get("max_samples")
    max_samples = int(max_samples) if max_samples is not None else None
    seed = int(config.get("random_seed", 42))
    split = str(dataset_cfg.get("split", "validation"))

    if name == "searchqa":
        dataset = SearchQADataset(
            index_split="train",
            test_split=split,
            max_index_docs=max_samples,
            max_test_queries=max_samples,
            random_seed=seed,
        )
    elif name == "movielens":
        dataset = MovielensDataset(
            max_index_docs=max_samples,
            max_test_queries=max_samples,
            random_seed=seed,
        )
    elif name == "natural_questions":
        dataset = NaturalQuestionsDataset(
            index_split="train",
            test_split=split,
            max_index_docs=max_samples,
            max_test_queries=max_samples,
            random_seed=seed,
        )
    elif name == "squad":
        dataset = SQuADDataset(
            index_split="train",
            test_split=split,
            max_index_docs=max_samples,
            max_test_queries=max_samples,
            random_seed=seed,
        )
    elif name == "triviaqa":
        dataset = TriviaQADataset(
            index_split="train",
            test_split=split,
            max_index_docs=max_samples,
            max_test_queries=max_samples,
            random_seed=seed,
        )
    elif name == "hotpotqa":
        dataset = HotpotQADataset(
            index_split="train",
            test_split=split,
            max_index_docs=max_samples,
            max_test_queries=max_samples,
            random_seed=seed,
        )
    elif name == "chatdoctor":
        dataset = ChatDoctorDataset(
            index_split="train",
            test_split=split,
            max_index_docs=max_samples,
            max_test_queries=max_samples,
            random_seed=seed,
        )
    elif name == "medical_dirichlet_phi3":
        dataset = MedicalDirichletPhi3Dataset(
            index_split="train",
            test_split=split,
            max_index_docs=max_samples,
            max_test_queries=max_samples,
            random_seed=seed,
        )
    else:
        raise ValueError(f"Unsupported dataset for DP-SynRAG: '{name}'.")

    dataset.load()
    return dataset


def _synth_cache_dir(config: dict[str, Any]) -> Path:
    """Derive a stable temp path keyed on all DP hyperparams that affect the synthetic corpus."""
    dataset_cfg = config.get("dataset", {})
    kw_cfg = config.get("keyword_extractor", {})
    cl_cfg = config.get("clustering", {})
    cr_cfg = config.get("cluster_retrieval", {})
    sg_cfg = config.get("synthetic_generation", {})
    sf_cfg = config.get("self_filter", {})

    key_dict = {
        "dataset_name": dataset_cfg.get("name", "searchqa"),
        "max_samples": dataset_cfg.get("max_samples"),
        "seed": config.get("random_seed", 42),
        "embedder": config.get("models", {}).get("embedder", ""),
        "llm": config.get("llm", {}).get("model_name", ""),
        "K": kw_cfg.get("K", 10),
        "R": cl_cfg.get("R", 50),
        "L": cl_cfg.get("L", 2),
        "sigma_h": cl_cfg.get("sigma_h", 1.0),
        "k_cluster": cr_cfg.get("k", 10),
        "sigma_mu": cr_cfg.get("sigma_mu", 0.5),
        "epsilon_theta_s": cr_cfg.get("epsilon_theta_s", 1.0),
        "T": sg_cfg.get("T", 30),
        "c": sg_cfg.get("c", 1.0),
        "tau": sg_cfg.get("tau", 0.0),
        "self_filter_enabled": sf_cfg.get("enabled", False),
    }
    digest = hashlib.md5(
        json.dumps(key_dict, sort_keys=True).encode()
    ).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"dp_synrag_idx_{digest}"


def _build_indexer(config: dict[str, Any], embedder: HFEmbedder, seed: int) -> DPSynRAGIndexer:
    kw_cfg = config.get("keyword_extractor", {})
    cl_cfg = config.get("clustering", {})
    cr_cfg = config.get("cluster_retrieval", {})
    sg_cfg = config.get("synthetic_generation", {})
    sf_cfg = config.get("self_filter", {})
    llm_name = str(config["llm"]["model_name"])

    keyword_extractor = KeywordExtractor(
        model_name=llm_name,
        K=int(kw_cfg.get("K", 10)),
        vocab=KeywordExtractor.build_default_vocab(
            min_word_len=int(kw_cfg.get("vocab_min_word_len", 3)),
            exclude_stopwords=bool(kw_cfg.get("exclude_stopwords", True)),
        ),
        max_new_tokens=int(kw_cfg.get("max_new_tokens", 50)),
        seed=seed,
    )
    clusterer = DPHistogramClusterer(
        R=int(cl_cfg.get("R", 50)),
        L=int(cl_cfg.get("L", 2)),
        sigma_h=float(cl_cfg.get("sigma_h", 1.0)),
        seed=seed,
    )
    cluster_retriever = DPClusterRetriever(
        embedder=embedder,
        k=int(cr_cfg.get("k", 10)),
        sigma_mu=float(cr_cfg.get("sigma_mu", 0.5)),
        epsilon_theta_s=float(cr_cfg.get("epsilon_theta_s", 1.0)),
        seed=seed,
    )
    prompt_template = str(sg_cfg.get(
        "prompt_template",
        "Example document:\n{document}\n\nWrite a new, similar document:\n",
    ))
    synthetic_generator = DPSyntheticGenerator(
        model_name=llm_name,
        T=int(sg_cfg.get("T", 30)),
        c=float(sg_cfg.get("c", 1.0)),
        tau=float(sg_cfg.get("tau", 0.0)),
        prompt_template=prompt_template,
        seed=seed,
    )
    self_filter: SelfFilter | None = None
    if sf_cfg.get("enabled", False):
        self_filter = SelfFilter(
            model_name=llm_name,
            prompt_template=str(sf_cfg.get(
                "prompt_template",
                "Does this document contain task-relevant information? Answer YES or NO.\n\nDocument: {text}\n\nAnswer:",
            )),
            max_new_tokens=int(sf_cfg.get("max_new_tokens", 4)),
            seed=seed,
        )

    return DPSynRAGIndexer(
        embedder=embedder,
        keyword_extractor=keyword_extractor,
        clusterer=clusterer,
        cluster_retriever=cluster_retriever,
        synthetic_generator=synthetic_generator,
        self_filter=self_filter,
        seed=seed,
    )


def _save_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main(
    config_path: str | None = None,
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
    private_entities = dataset.get_private_entities()

    embedder = HFEmbedder(model_name=str(config["models"]["embedder"]))
    indexer = _build_indexer(config, embedder, seed)

    cache_dir = _synth_cache_dir(config)
    task_prefix = str(config.get("synthetic_generation", {}).get("task_prefix", ""))

    if cache_dir.exists():
        print(f"Loading cached synthetic index from {cache_dir} ...")
        indexer.load(str(cache_dir))
    else:
        print(f"Building synthetic corpus from {len(documents)} documents ...")
        indexer.build_from_documents(documents, task_prefix=task_prefix)
        indexer.save(str(cache_dir))
        atexit.register(shutil.rmtree, str(cache_dir), ignore_errors=True)
        print(f"Synthetic index cached at {cache_dir} (will be deleted on exit).")

    stats = indexer.last_stats
    print(
        f"Synthetic corpus: {stats.get('n_original_documents', '?')} original docs → "
        f"{stats.get('n_non_empty_subsets', '?')} non-empty clusters → "
        f"{stats.get('n_synthetic_kept', '?')} synthetic docs"
    )

    # Privacy accounting (one-time cost; covers the entire corpus)
    kw_cfg = config.get("keyword_extractor", {})
    cl_cfg = config.get("clustering", {})
    cr_cfg = config.get("cluster_retrieval", {})
    sg_cfg = config.get("synthetic_generation", {})
    delta = float(config.get("delta", 1e-3))

    zcdp_result = zcdp_dp_synrag(
        K=int(kw_cfg.get("K", 10)),
        sigma_h=float(cl_cfg.get("sigma_h", 1.0)),
        L=int(cl_cfg.get("L", 2)),
        epsilon_theta_s=float(cr_cfg.get("epsilon_theta_s", 1.0)),
        sigma_mu=float(cr_cfg.get("sigma_mu", 0.5)),
        T=int(sg_cfg.get("T", 30)),
        c=float(sg_cfg.get("c", 1.0)),
        tau=float(sg_cfg.get("tau", 0.0)),
        delta=delta,
    )
    print(print_zcdp_analysis(zcdp_result))

    # Vanilla inference over the synthetic corpus
    retriever = CosineSimilarityRetriever(indexer=indexer)
    reranker = IdentityReranker()
    prompt_builder = SimplePromptBuilder()
    generator = HFGenerator(
        model_name=str(config["llm"]["model_name"]),
        max_new_tokens=int(config.get("inference", {}).get("max_new_tokens", 64)),
    )
    postproc = IdentityPostprocessor()

    pipeline = RAGPipeline(
        chunker=NoChunker(),
        embedder=embedder,
        indexer=indexer,
        retriever=retriever,
        reranker=reranker,
        prompt_builder=prompt_builder,
        generator=generator,
        postprocessor=postproc,
    )

    k_retrieved = int(config.get("inference", {}).get("k_retrieved", 5))
    verbose = bool(config.get("verbose", False))
    verbose_max_examples = config.get("verbose_max_examples")
    verbose_max_examples = int(verbose_max_examples) if verbose_max_examples is not None else None

    predictions: list[str] = []
    references: list[str] = []
    results: list[dict[str, str]] = []
    verbose_count = 0

    for item in tqdm(queries, desc="DP-SynRAG inference"):
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        reference = str(item.get("answer", "")).strip()

        can_print_verbose = verbose and (
            verbose_max_examples is None or verbose_count < verbose_max_examples
        )

        if can_print_verbose:
            query_vector = embedder.embed_query(query)
            retrieved_chunks = retriever.retrieve(query_vector, k=k_retrieved)
            reranked_chunks = reranker.rerank(query, retrieved_chunks)
            prompt = prompt_builder.build(query, reranked_chunks)
            raw_response = generator.generate(prompt)
            response = postproc.process(raw_response)

            print("=" * 30)
            print(f"Question: {query}")
            print(f"Retrieved context (k={k_retrieved}):")
            for idx, chunk in enumerate(retrieved_chunks, start=1):
                score = chunk.get("score", "N/A")
                text = str(chunk.get("text", ""))
                score_str = f"{float(score):.4f}" if isinstance(score, (int, float)) else score
                print(f"  [{idx}] score={score_str} | {text[:150]}")
            print(f"Answer: {response}")
            if reference:
                print(f"Reference: {reference}")
            print()
            verbose_count += 1
        else:
            response = pipeline.query(query, k=k_retrieved)

        predictions.append(str(response))
        references.append(reference)
        results.append({"query": query, "prediction": str(response), "reference": reference})

    utility_metric_names = config.get("metrics", {}).get("utility", [])
    privacy_metric_names = config.get("metrics", {}).get("privacy", [])

    utility_metrics = compute_utility_metrics(
        predictions,
        references,
        utility_metric_names,
        texts=indexer.synthetic_texts,  # used by perplexity metric
    )
    privacy_metrics = compute_privacy_metrics(
        predictions,
        references,
        private_entities,
        privacy_metric_names,
        seed=seed,
    )
    metrics_payload = {**utility_metrics, **privacy_metrics}

    dataset_name = str(config.get("dataset", {}).get("name", "unknown"))
    hyperparams = {
        "K": int(kw_cfg.get("K", 10)),
        "R": int(cl_cfg.get("R", 50)),
        "L": int(cl_cfg.get("L", 2)),
        "sigma_h": float(cl_cfg.get("sigma_h", 1.0)),
        "k_cluster": int(cr_cfg.get("k", 10)),
        "sigma_mu": float(cr_cfg.get("sigma_mu", 0.5)),
        "epsilon_theta_s": float(cr_cfg.get("epsilon_theta_s", 1.0)),
        "T": int(sg_cfg.get("T", 30)),
        "c": float(sg_cfg.get("c", 1.0)),
        "tau": float(sg_cfg.get("tau", 0.0)),
        "delta": delta,
        "k_retrieved": k_retrieved,
        "embedder": str(config["models"]["embedder"]),
        "llm": str(config["llm"]["model_name"]),
    }

    payload = {
        "method": "dp_synrag",
        "dataset": dataset_name,
        "hyperparams": hyperparams,
        "privacy_accounting": {
            "zcdp": {
                "epsilon": zcdp_result.epsilon,
                "delta": zcdp_result.delta,
                "rho": zcdp_result.rho,
                "components": zcdp_result.components,
            }
        },
        "indexer_stats": stats,
        "metrics": metrics_payload,
        "n_queries": len(predictions),
        "n_synthetic_docs": len(indexer.synthetic_texts),
        "timestamp": datetime.now(UTC).isoformat(),
        "results": results,
    }

    output_dir = Path(str(config.get("output_dir", "results/dp_synrag/"))) / dataset_name
    out_path = output_dir / "run.json"
    _save_results(out_path, payload)
    print(f"Saved: {out_path}")
    print(f"ε = {zcdp_result.epsilon:.4f}, δ = {zcdp_result.delta:.1e}, ρ = {zcdp_result.rho:.4f}")
    print(f"Metrics: {metrics_payload}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DP-SynRAG experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "configs" / "dp_synrag.yaml"),
        help="Path to DP-SynRAG YAML config.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print retrieval and model outputs for each query.",
    )
    parser.add_argument(
        "--verbose-max-examples",
        type=int,
        default=None,
        help="Maximum number of queries to print in verbose mode.",
    )
    args = parser.parse_args()
    main(
        args.config,
        verbose_override=args.verbose,
        verbose_max_examples_override=args.verbose_max_examples,
    )
