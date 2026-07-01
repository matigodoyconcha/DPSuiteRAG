"""Benchmark script for Private-RAG (MURAG / MURAG-ADA), arXiv:2511.07637.

Multi-query differentially private RAG. A per-document individual privacy filter
(``BudgetManager``, ``E(z) = M * eps_q``) makes accumulated privacy loss depend on
how often each document is retrieved, not on the number of queries. Only the
*retrieval* differs between the two variants (chosen with ``private_rag.variant``):

- ``murag``      -> ``BudgetAwareRetriever`` (fixed public relevance threshold).
- ``murag_ada``  -> ``AdaptiveBudgetAwareRetriever`` (query-specific threshold via
                    noisy prefix sums; per-query budget splits eps_q = eps_thr + eps_RAG).

Both feed their Top-K documents to the same pure-DP Algorithm-7 generator
(``PrivateRAGGenerator``) through the existing ``DPVoteRAGPipeline``. The sweep is
over the per-query budget ``eps_q``; the headline guarantee is ``epsilon = M * eps_q``
(pure DP, delta = 0). Queries are processed in order so the budget depletes across
the stream (the whole point of the multi-query setting).

Usage:
    python experiments/run_private_rag.py --config experiments/configs/private_rag.yaml
"""

from __future__ import annotations

import argparse
import json
import os
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

from core.device import (
    cuda_oom_warning,
    free_memory,
    is_oom_error,
    model_device_kwargs,
    resolve_device,
)
from experiments.index_cache import index_cache_dir, load_or_build_index
from experiments.run_logging import log, setup_method_logging
from bench_datasets.chatdoctor import ChatDoctorDataset
from bench_datasets.hotpotqa import HotpotQADataset
from bench_datasets.medical_dirichlet_phi3 import MedicalDirichletPhi3Dataset
from bench_datasets.movielens import MovielensDataset
from bench_datasets.natural_questions import NaturalQuestionsDataset
from bench_datasets.searchqa import SearchQADataset
from bench_datasets.squad import SQuADDataset
from bench_datasets.triviaqa import TriviaQADataset
from dp_methods.per_query.dpvoterag.pipeline import DPVoteRAGPipeline
from dp_methods.per_query.private_rag.budget_manager import BudgetManager
from dp_methods.per_query.private_rag.generator import PrivateRAGGenerator
from dp_methods.per_query.private_rag.retriever import (
    AdaptiveBudgetAwareRetriever,
    BudgetAwareRetriever,
)
from metrics.privacy import compute_privacy_metrics
from metrics.privacy_accounting_private_rag import method_epsilon, summarize
from metrics.utility import compute_utility_metrics
from vanilla.chunker import FixedSizeChunker
from vanilla.embedder import build_embedder, embedder_identity
from vanilla.indexer import FAISSIndexer
from vanilla.postprocessor import IdentityPostprocessor
from vanilla.prompt_builder import SimplePromptBuilder
from vanilla.reranker import IdentityReranker

_VARIANTS = {"murag", "murag_ada"}


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
    datasets = {
        "natural_questions": NaturalQuestionsDataset,
        "triviaqa": TriviaQADataset,
        "hotpotqa": HotpotQADataset,
        "squad": SQuADDataset,
        "chatdoctor": ChatDoctorDataset,
        "medical_dirichlet_phi3": MedicalDirichletPhi3Dataset,
        "searchqa": SearchQADataset,
        "movielens": MovielensDataset,
    }
    if key not in datasets:
        raise ValueError(
            f"Dataset '{name}' not supported for Private-RAG. "
            f"Use one of: {sorted(datasets)}."
        )
    dataset = datasets[key](**common_kwargs)
    dataset.load()
    return dataset


def _iter_dataset_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    base = config.get("dataset", {}) or {}
    datasets = config.get("datasets")
    if isinstance(datasets, list) and datasets:
        return [{**base, **(entry or {})} for entry in datasets]
    return [dict(base)]


def _iter_llm_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    base = config.get("llm", {}) or {}
    llms = config.get("llms")
    if isinstance(llms, list) and llms:
        return [{**base, **(entry or {})} for entry in llms]
    return [dict(base)]


def _llm_label(llm_cfg: dict[str, Any]) -> str:
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
    parser = argparse.ArgumentParser(description="Run Private-RAG (MURAG / MURAG-ADA) experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/private_rag.yaml",
        help="Path to the Private-RAG configuration YAML.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print per-query mechanism traces.",
    )
    parser.add_argument(
        "--verbose-max-examples",
        type=int,
        default=None,
        help="Maximum number of queries to print in verbose mode.",
    )
    return parser


def _make_retriever(
    variant: str,
    indexer: Any,
    budget_manager: BudgetManager,
    epsilon_q: float,
    pr_cfg: dict[str, Any],
    seed: int,
) -> Any:
    """Build the MURAG or MURAG-ADA retriever for a given per-query budget."""
    if variant == "murag":
        return BudgetAwareRetriever(
            indexer=indexer,
            budget_manager=budget_manager,
            epsilon_q=epsilon_q,
            threshold=float(pr_cfg.get("threshold", 0.25)),
            seed=seed,
        )
    return AdaptiveBudgetAwareRetriever(
        indexer=indexer,
        budget_manager=budget_manager,
        epsilon_q=epsilon_q,
        epsilon_thr=float(pr_cfg.get("epsilon_thr", 1.0)),
        score_min=float(pr_cfg.get("score_min", 0.0)),
        score_max=float(pr_cfg.get("score_max", 1.0)),
        bin_size=float(pr_cfg.get("bin_size", 0.05)),
        seed=seed,
    )


def _run_for_llm(
    config: dict[str, Any],
    args: argparse.Namespace,
    llm_model_name: str,
    llm_label: str,
    multi_llm: bool,
) -> list[dict[str, Any]]:
    embedder_model_name = str(config.get("models", {}).get("embedder", "sentence-transformers/all-MiniLM-L6-v2"))
    _models_cfg = config.get("models", {})
    dual_encoder = bool(_models_cfg.get("dual_encoder", False))
    ctx_encoder = _models_cfg.get("ctx_encoder")
    embedder_cache_name = embedder_identity(
        embedder_model_name, dual_encoder=dual_encoder, ctx_encoder=ctx_encoder
    )

    pr_cfg = config.get("private_rag", {})
    variant = str(pr_cfg.get("variant", "murag")).lower()
    if variant not in _VARIANTS:
        raise ValueError(f"private_rag.variant must be one of {sorted(_VARIANTS)}, got {variant!r}")
    seed = int(pr_cfg.get("random_seed", 42))
    m_docs = int(pr_cfg.get("M", 1))  # max retrievals per document (headline eps = M*eps_q)
    m = int(pr_cfg.get("m", 50))      # voters
    k = int(pr_cfg.get("k", 1))       # docs per voter
    epsilon_0 = float(pr_cfg.get("epsilon_0", 2.0))
    epsilon_thr = float(pr_cfg.get("epsilon_thr", 1.0))
    vote_threshold = pr_cfg.get("vote_threshold", None)
    vote_threshold = float(vote_threshold) if vote_threshold is not None else None
    t_max = int(pr_cfg.get("t_max", 50))
    exclude_stopwords = bool(pr_cfg.get("exclude_stopwords", False))
    exclude_question_words = bool(pr_cfg.get("exclude_question_words", False))
    stopwords_path = pr_cfg.get("stopwords_path", None)
    use_chat_template = pr_cfg.get("use_chat_template", "auto")
    stop_strings = pr_cfg.get("stop_strings")

    epsilon_q_values = [float(v) for v in config.get("epsilon_q_values", [2, 5, 10, 15, 20, 30, 40])]
    k_retrieved = int(config.get("k_retrieved", m * k))
    max_queries = config.get("max_queries", 40)
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
    output_dir = Path(str(config.get("output_dir", "results/private_rag/")))
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    _kwargs = model_device_kwargs()
    _device = resolve_device()
    _ngpu = torch.cuda.device_count() if _device == "cuda" else 1
    print(f"Loading model/tokenizer... (device: {_device}, gpus: {_ngpu})")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    with cuda_oom_warning(llm_model_name, _device):
        model = AutoModelForCausalLM.from_pretrained(llm_model_name, **_kwargs)
        if not _kwargs:
            model = model.to(_device)
    model.eval()

    chunker = FixedSizeChunker(chunk_size=200, overlap=20)
    embedder = build_embedder(
        embedder_model_name, dual_encoder=dual_encoder, ctx_encoder=ctx_encoder
    )
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

        log(
            f"Dataset START: {dataset_name} (variant={variant}, "
            f"max_index_docs={max_index_docs}, max_test_queries={ds_max_queries})"
        )

        indexer = FAISSIndexer()

        try:
            dataset = load_dataset(
                dataset_name, seed=seed, max_queries=ds_max_queries, max_index_docs=max_index_docs
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

        cache_dir = index_cache_dir(
            dataset_name=dataset_name,
            max_docs=max_index_docs,
            embedder_model=embedder_cache_name,
            chunk_size=200,
            chunk_overlap=20,
            seed=seed,
        )
        index_built = False

        for epsilon_q in epsilon_q_values:
            # MURAG-ADA splits eps_q = eps_thr + eps_RAG, so it needs eps_q > eps_thr.
            if variant == "murag_ada" and epsilon_q <= epsilon_thr:
                log(
                    f"[{dataset_name}] SKIP eps_q={epsilon_q}: must exceed eps_thr={epsilon_thr} "
                    "for MURAG-ADA (eps_RAG = eps_q - eps_thr > 0)."
                )
                continue

            generation_eps = epsilon_q if variant == "murag" else (epsilon_q - epsilon_thr)
            headline_eps = method_epsilon(m_docs, epsilon_q)
            log(
                f"[{dataset_name}] RUN START: variant={variant} eps_q={epsilon_q} "
                f"(headline eps=M*eps_q={headline_eps}, gen_eps={generation_eps}, M={m_docs})"
            )

            # Fresh per-document filter for each operating point (independent runs).
            budget_manager = BudgetManager(initial_budget=headline_eps)
            retriever = _make_retriever(variant, indexer, budget_manager, epsilon_q, pr_cfg, seed)
            generator = PrivateRAGGenerator(
                model=model,
                tokenizer=tokenizer,
                m=m,
                k=k,
                epsilon=float(generation_eps),
                epsilon_0=epsilon_0,
                vote_threshold=vote_threshold,
                t_max=t_max,
                random_seed=seed,
                exclude_stopwords=exclude_stopwords,
                exclude_question_words=exclude_question_words,
                stopwords_path=stopwords_path,
                task_instruction=dataset.get_task_instruction(),
                use_chat_template=use_chat_template,
                stop_strings=stop_strings,
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

            predictions: list[str] = []
            references: list[str] = []
            results: list[dict[str, Any]] = []
            query_times: list[float] = []
            n_private_total = 0
            n_baseline_total = 0
            n_docs_retrieved_total = 0

            selected_queries = queries[:ds_max_queries] if ds_max_queries is not None else queries
            verbose_count = 0
            t_run_start = time.time()
            total_q = len(selected_queries)

            for q_idx, sample in enumerate(selected_queries, start=1):
                question = str(sample.get("query", "")).strip()
                reference = str(sample.get("answer", "")).strip()
                if not question:
                    continue

                can_print_verbose = verbose and (
                    verbose_max_examples is None or verbose_count < verbose_max_examples
                )
                generator.verbose = can_print_verbose
                generator.verbose_label = question
                retriever.verbose = can_print_verbose

                t0 = time.time()
                try:
                    answer = pipeline.query(question, k=k_retrieved)
                finally:
                    generator.verbose = False
                    retriever.verbose = False
                elapsed = time.time() - t0

                gstats = dict(generator.last_run_stats)
                rstats = dict(retriever.last_run_stats)
                n_private_total += int(gstats.get("n_private_tokens", 0))
                n_baseline_total += int(gstats.get("n_baseline_tokens", 0))
                n_docs_retrieved_total += int(rstats.get("n_returned", 0))

                predictions.append(str(answer))
                references.append(reference)
                results.append(
                    {
                        "query": question,
                        "prediction": str(answer),
                        "reference": reference,
                        "n_docs_retrieved": int(rstats.get("n_returned", 0)),
                        "n_docs_charged": int(rstats.get("n_charged", 0)),
                        "n_private_tokens": int(gstats.get("n_private_tokens", 0)),
                        "n_baseline_tokens": int(gstats.get("n_baseline_tokens", 0)),
                    }
                )
                query_times.append(float(elapsed))

                if can_print_verbose:
                    print(f"Answer: {answer}")
                    if reference:
                        print(f"Reference: {reference}")
                    verbose_count += 1

                if q_idx % 10 == 0:
                    log(f"[{dataset_name} eps_q={epsilon_q}] heartbeat: query {q_idx}/{total_q}")

                free_memory()  # Private-RAG runs m+1 forwards per token: free memory each query

            budget_stats = budget_manager.stats(cost=epsilon_q)
            _n_done = len(predictions)
            log(
                f"[{dataset_name}] RUN END: eps_q={epsilon_q} — {_n_done} queries in "
                f"{time.time() - t_run_start:.1f}s | private tokens={n_private_total} "
                f"baseline={n_baseline_total}; docs tracked={budget_stats['n_tracked']} "
                f"exhausted={budget_stats.get('n_exhausted', 0)}"
            )

            # Deferred (orchestrated run): skip in-process metrics; score_results.py
            # computes them later in a clean process (avoids BERTScore/perplexity OOM).
            if os.environ.get("BENCHDP_DEFER_METRICS") == "1":
                util, priv = {}, {}
            else:
                util = compute_utility_metrics(predictions, references, utility_metrics)
                priv = compute_privacy_metrics(predictions, references, privacy_metrics, seed=seed)

            time_per_query = _safe_float(np.mean(query_times) if query_times else 0.0)
            match_acc = _safe_float(util.get("match_accuracy", 0.0))
            contains_acc = _safe_float(util.get("contains_accuracy", 0.0))
            mia_auc = _safe_float(priv.get("mia_auc_bleu", 0.5))

            accounting = summarize(
                variant=variant,
                m_retrievals=m_docs,
                epsilon_q=epsilon_q,
                epsilon_0=epsilon_0,
                epsilon_thr=epsilon_thr if variant == "murag_ada" else None,
            )

            payload: dict[str, Any] = {
                "method": "private_rag",
                "variant": variant,
                "timestamp": datetime.now(UTC).isoformat(),
                "dataset": dataset_name,
                "llm": llm_model_name,
                "max_index_docs": max_index_docs,
                "epsilon": float(headline_eps),   # headline M*eps_q (read by score_results)
                "epsilon_q": float(epsilon_q),
                "M": m_docs,
                "epsilon_0": epsilon_0,
                "epsilon_thr": epsilon_thr if variant == "murag_ada" else None,
                "delta": 0.0,
                "m": m,
                "k": k,
                "vote_threshold": generator.vote_threshold,
                "t_max": t_max,
                "accounting": accounting,
                "budget": budget_stats,
                "n_private_tokens_total": int(n_private_total),
                "n_baseline_tokens_total": int(n_baseline_total),
                "n_docs_retrieved_total": int(n_docs_retrieved_total),
                "predictions": predictions,
                "references": references,
                "results": results,
                "metrics": {**util, **priv},
                "match_accuracy": match_acc,
                "contains_accuracy": contains_acc,
                "mia_auc": mia_auc,
                "time_per_query_sec": time_per_query,
            }

            output_path = output_dir / dataset_name
            if multi_llm:
                output_path = output_path / llm_label
            output_path = output_path / f"epsilon_q_{epsilon_q}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "llm": llm_model_name,
                    "variant": variant,
                    "epsilon_q": float(epsilon_q),
                    "epsilon": float(headline_eps),
                    "match_accuracy": match_acc,
                    "contains_accuracy": contains_acc,
                    "mia_auc": mia_auc,
                    "time_per_query_sec": time_per_query,
                    "docs_exhausted": int(budget_stats.get("n_exhausted", 0)),
                    "output": str(output_path),
                }
            )

            del generator, pipeline, retriever, budget_manager
            free_memory()

        del dataset, indexer
        free_memory()

    del model, tokenizer, embedder
    free_memory()

    return summary_rows


def main() -> None:
    setup_method_logging("private_rag")
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
            print(f"\n[WARN] skipping LLM {llm_model_name}: {exc}")
        except (RuntimeError, MemoryError) as exc:
            if not is_oom_error(exc):
                raise
            print(f"\n[WARN] OOM — skipping LLM {llm_model_name}: {exc}")
        # free_memory() runs OUTSIDE the except on purpose: while `exc` is alive
        # its traceback pins _run_for_llm's frame and locals (model, embedder,
        # generator, ...), so a gc.collect() inside the handler frees nothing.
        # Reached on the success path too — an extra empty_cache is harmless.
        free_memory()  # release the (possibly failed) model before the next LLM

    print("\nPrivate-RAG summary:")
    for row in summary_rows:
        print(
            f"- [{row['dataset']} | {row['llm']}] {row['variant']} eps_q={row['epsilon_q']} "
            f"(eps={row['epsilon']}): match_accuracy={row['match_accuracy']:.4f}, "
            f"contains_accuracy={row['contains_accuracy']:.4f}, "
            f"mia_auc={row['mia_auc']:.4f}, docs_exhausted={row['docs_exhausted']}, "
            f"time_per_query_sec={row['time_per_query_sec']:.4f}, out={row['output']}"
        )


if __name__ == "__main__":
    main()
