"""Benchmark script for DPSparseVoteRAG with sparse voting-based differential privacy.

Runs the sparse voting variant of DPVoteRAG with agreement thresholds (tau)
to reduce privacy budget consumption while maintaining utility.

Usage:
	python experiments/run_dpsparsevoterag.py --config config/dpsparsevoterag.yaml --dataset squad --m 30 --tau 15
"""

from __future__ import annotations

import argparse
import json
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

from experiments.index_cache import index_cache_dir, load_or_build_index
from bench_datasets.chatdoctor import ChatDoctorDataset
from bench_datasets.hotpotqa import HotpotQADataset
from bench_datasets.natural_questions import NaturalQuestionsDataset
from bench_datasets.medical_dirichlet_phi3 import MedicalDirichletPhi3Dataset
from bench_datasets.squad import SQuADDataset
from bench_datasets.triviaqa import TriviaQADataset
from dp_methods.dpvoterag.pipeline import DPVoteRAGPipeline
from dp_methods.dpvoterag.sparse_vote_generator import SparseVoteRAGGenerator
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
        raise FileNotFoundError(f"No existe el archivo de configuracion: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("El YAML de configuracion debe ser un dict.")
    return config


def load_dataset(name: str, seed: int, max_queries: int | None) -> Any:
    key = str(name).lower()
    common_kwargs = {
        "max_test_queries": max_queries,
        "random_seed": seed,
    }

    if key == "natural_questions":
        dataset = NaturalQuestionsDataset(max_index_docs=200, **common_kwargs)
    elif key == "triviaqa":
        dataset = TriviaQADataset(max_index_docs=200, **common_kwargs)
    elif key == "hotpotqa":
        dataset = HotpotQADataset(max_index_docs=200, **common_kwargs)
    elif key == "squad":
        dataset = SQuADDataset(max_index_docs=200, **common_kwargs)
    elif key == "chatdoctor":
        dataset = ChatDoctorDataset(max_index_docs=200, **common_kwargs)
    elif key == "medical_dirichlet_phi3":
        dataset = MedicalDirichletPhi3Dataset(max_index_docs=200, **common_kwargs)
    else:
        raise ValueError(
            f"Dataset '{name}' no soportado para DPSparseVoteRAG. "
            "Usa natural_questions, triviaqa, hotpotqa, squad, chatdoctor o medical_dirichlet_phi3."
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
    parser = argparse.ArgumentParser(description="Ejecuta experimento DPSparseVoteRAG")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/dpsparsevoterag.yaml",
        help="Ruta al YAML de configuracion de DPSparseVoteRAG.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Imprime outputs por query (pregunta, respuesta y referencia).",
    )
    parser.add_argument(
        "--verbose-max-examples",
        type=int,
        default=None,
        help="Numero maximo de queries a imprimir en modo verbose. Por defecto imprime todas.",
    )
    return parser


def _run_verbose_query(
    query: str,
    reference: str,
    k_retrieved: int,
    embedder: HFEmbedder,
    retriever: CosineSimilarityRetriever,
    reranker: IdentityReranker,
    generator: SparseVoteRAGGenerator,
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
    raw_response = generator.generate_vote(query, retrieved_docs)
    final_response = postprocessor.process(raw_response)

    print("=" * 30)
    print(f"Pregunta: {query}")
    print("-" * 30)
    print(f"DOCS RECUPERADOS (k={k_retrieved}):")
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        text = str(chunk.get("text", ""))
        score = chunk.get("score", None)
        score_str = f"score={float(score):.4f}" if score is not None else "score=N/A"
        print(f"[{idx}] {score_str} | {text}")
    print("-" * 30)
    print(f"OUTPUT MODELO (raw): {raw_response}")
    print(f"OUTPUT FINAL: {final_response}")
    if reference:
        print(f"REFERENCIA: {reference}")
    print()

    return final_response


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    dataset_name = str(config.get("dataset", {}).get("name", "natural_questions"))
    llm_cfg = config.get("llm", {})
    llm_model_name = str(llm_cfg.get("model_name", "facebook/opt-1.3b"))
    embedder_model_name = str(config.get("models", {}).get("embedder", "sentence-transformers/all-MiniLM-L6-v2"))

    sparse_cfg = config.get("dpsparsevoterag", {})
    seed = int(sparse_cfg.get("random_seed", 42))
    m = int(sparse_cfg.get("m", 30))
    k = int(sparse_cfg.get("k", 1))
    epsilon_token = float(sparse_cfg.get("epsilon_token", 2.0))
    sparse_eps = float(sparse_cfg.get("sparse_eps", 1.0))
    delta_token = float(sparse_cfg.get("delta_token", 1e-5))
    k_bar = sparse_cfg.get("k_bar", None)
    k_bar = int(k_bar) if k_bar is not None else None
    tau = sparse_cfg.get("threshold", sparse_cfg.get("tau", None))
    tau = None if tau is None else float(tau)
    tmax_absolute = int(sparse_cfg.get("tmax_absolute", 200))
    exclude_stopwords = bool(sparse_cfg.get("exclude_stopwords", False))
    exclude_question_words = bool(sparse_cfg.get("exclude_question_words", False))
    stopwords_path = sparse_cfg.get("stopwords_path", None)

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
    output_dir = Path(str(config.get("output_dir", "results/dpsparsevoterag/")))
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    print("Cargando modelo/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    model = AutoModelForCausalLM.from_pretrained(llm_model_name)
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
        raise ValueError("El dataset no devolvio documentos para indexar.")
    if not queries:
        raise ValueError("El dataset no devolvio queries para evaluar.")

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

    for epsilon_total in epsilon_values:
        generator = SparseVoteRAGGenerator(
            model=model,
            tokenizer=tokenizer,
            m=m,
            k=k,
            epsilon_token=epsilon_token,
            sparse_eps=sparse_eps,
            delta_token=delta_token,
            epsilon_total=float(epsilon_total),
            delta_total=delta_total,
            k_bar=k_bar,
            random_seed=seed,
            tau=tau,
            tmax_absolute=tmax_absolute,
            exclude_stopwords=exclude_stopwords,
            exclude_question_words=exclude_question_words,
            stopwords_path=stopwords_path,
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
        results: list[dict[str, Any]] = []
        query_times: list[float] = []

        tokens_private = 0
        tokens_nonprivate = 0
        tokens_excluded = 0
        dp_flags_per_query: list[list[bool | None]] = []
        exit_statuses: list[int] = []

        selected_queries = queries[:max_queries] if max_queries is not None else queries
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
            results.append(
                {
                    "query": question,
                    "prediction": str(answer),
                    "reference": reference,
                    "exit_status": int(dict(generator.last_run_stats).get("exit_status", 0)),
                    "dp_flags_per_token": list(dict(generator.last_run_stats).get("dp_flags_per_token", [])),
                }
            )
            query_times.append(float(elapsed))

            stats = dict(generator.last_run_stats)
            tokens_private += int(stats.get("n_private_tokens", 0))
            tokens_nonprivate += int(stats.get("n_nonprivate_tokens", 0))
            tokens_excluded += int(stats.get("n_excluded_tokens", 0))
            dp_flags_per_query.append(list(stats.get("dp_flags_per_token", [])))
            exit_statuses.append(int(stats.get("exit_status", 0)))

        util = compute_utility_metrics(predictions, references, utility_metrics)
        priv = compute_privacy_metrics(predictions, references, private_entities, privacy_metrics, seed=seed)

        time_per_query = _safe_float(np.mean(query_times) if query_times else 0.0)
        match_acc = _safe_float(util.get("match_accuracy", 0.0))
        mia_auc = _safe_float(priv.get("mia_auc_bleu", 0.5))

        payload: dict[str, Any] = {
            "method": "dpsparsevoterag",
            "timestamp": datetime.now(UTC).isoformat(),
            "dataset": dataset_name,
            "epsilon_total": float(epsilon_total),
            "delta_total": delta_total,
            "m": m,
            "k": k,
            "epsilon_token": epsilon_token,
            "sparse_eps": sparse_eps,
            "tau": float(generator.tau),
            "tmax_absolute": tmax_absolute,
            "exclude_stopwords": exclude_stopwords,
            "exclude_question_words": exclude_question_words,
            "stopwords_path": stopwords_path,
            "tokens_private": int(tokens_private),
            "tokens_nonprivate": int(tokens_nonprivate),
            "tokens_excluded": int(tokens_excluded),
            "dp_flags_per_query": dp_flags_per_query,
            "exit_statuses": exit_statuses,
            "predictions": predictions,
            "references": references,
            "results": results,
            "match_accuracy": match_acc,
            "mia_auc": mia_auc,
            "time_per_query_sec": time_per_query,
        }

        output_path = output_dir / dataset_name / f"epsilon_{epsilon_total}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        summary_rows.append(
            {
                "epsilon_total": float(epsilon_total),
                "match_accuracy": match_acc,
                "mia_auc": mia_auc,
                "time_per_query_sec": time_per_query,
                "tokens_private": int(tokens_private),
                "tokens_nonprivate": int(tokens_nonprivate),
                "tokens_excluded": int(tokens_excluded),
                "output": str(output_path),
            }
        )

    print("\nResumen DPSparseVoteRAG:")
    for row in summary_rows:
        print(
            f"- eps={row['epsilon_total']}: "
            f"match_accuracy={row['match_accuracy']:.4f}, "
            f"mia_auc={row['mia_auc']:.4f}, "
            f"time_per_query_sec={row['time_per_query_sec']:.4f}, "
            f"tokens_private={row['tokens_private']}, "
            f"tokens_nonprivate={row['tokens_nonprivate']}, "
            f"tokens_excluded={row['tokens_excluded']}, "
            f"out={row['output']}"
        )


if __name__ == "__main__":
    main()

# Pilot verification results (update after running the script):
# - Pending execution in this workspace.
