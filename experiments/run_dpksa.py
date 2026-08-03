"""Benchmark script for DP-KSA (Tang et al., arXiv:2602.14374).

Runs the DP-KSA pipeline (sample-and-aggregate over the top-N retrieved documents,
FindBestK + TopKWithPTR keyword release, keyword-conditioned final generation) over
the shared privacy axis crossed with the workload size.

One cell is a pair (target epsilon on the shared axis, workload size Q). The target is
what the whole Q-query stream is allowed to cost; the per-query budget handed to the
generator is derived from it, because DP-KSA pays per query and therefore has to spend
less on each question as the stream grows. Comparing methods at a fixed total over a
growing Q is what turns that scaling into a measured utility curve rather than an
argument about accounting.

Every cell answers the same questions: Q prices the guarantee, ``max_queries`` sizes the
utility sample, and the two are reported side by side.

Usage:
    python experiments/run_dpksa.py --config experiments/configs/full/dpksa.yaml
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
    privacy_sweep,
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
from dp_methods.per_query.dpksa.generator import DPKSAGenerator
from dp_methods.per_query.dpksa.pipeline import DPKSAPipeline
from metrics.privacy import compute_privacy_metrics
from metrics.utility import compute_utility_metrics
from metrics.workload_accounting import DEFAULT_DELTA, budget_for_target
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
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("The configuration YAML must be a dict.")
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(out) or np.isinf(out):
        return default
    return out


def _cell_output_path(
    output_dir: Path,
    dataset_name: str,
    llm_label: str | None,
    workload_q: int,
    epsilon_target: float,
) -> Path:
    """Where one (target, workload size) cell's record goes.

    The workload size names a directory of its own: the same target is priced for
    several stream lengths, so cells sharing a target would otherwise write over each
    other and only the last Q would survive.
    """
    path = output_dir / dataset_name
    if llm_label:
        path = path / llm_label
    return path / f"q_{workload_q}" / f"target_{epsilon_target:g}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DP-KSA experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/full/dpksa.yaml",
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


def _run_for_llm(
    config: dict[str, Any],
    args: argparse.Namespace,
    llm_model_name: str,
    llm_label: str,
    multi_llm: bool,
) -> list[dict[str, Any]]:
    embedder_model_name = str(config.get("models", {}).get("embedder", "sentence-transformers/all-MiniLM-L6-v2"))
    embedder_cache_name = embedder_identity(embedder_model_name)

    dp_cfg = config.get("dpksa", {})
    # data_seed picks the corpus and the questions; run_seed drives the DP noise, so
    # a repetition varies the mechanism without resampling or rebuilding the index.
    data_seed, seed = resolve_seeds(config)
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
    # C2 (chat template, auto by default) + C3 (optional stop strings, OFF by default).
    use_chat_template = dp_cfg.get("use_chat_template", "auto")
    stop_strings = dp_cfg.get("stop_strings")

    # Points on the shared axis, crossed with the stream lengths they are priced for.
    # Shared-axis targets to invert, or the paper's own budget values taken as
    # they stand — see experiments.common.privacy_sweep.
    sweep_kind, sweep_values, _ = privacy_sweep(config, paper_key="epsilon_values")
    epsilon_targets = sweep_values
    workload_queries = [int(x) for x in config.get("workload_queries", [100])]
    # DP-KSA's own mechanism delta: a structural parameter of the calibration, distinct
    # from the shared axis's DEFAULT_DELTA that the targets above are quoted at.
    mechanism_delta = float(config.get("delta", 1e-4))
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
    privacy_metrics = list(config.get("metrics", {}).get("privacy", []))
    output_dir = Path(str(config.get("output_dir", "results/dpksa/")))
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

        # Standalone MIA mode: index only the member subset and probe the pipeline
        # instead of running the utility QA loop (see experiments/mia_probe.py).
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
        # Stage-1 responses are a deterministic function of (query, document) once the
        # task instruction, chat template and answer length are fixed, and every cell
        # below shares those while varying only the budget, which the mechanism applies
        # after the histogram. Scoping the memo to the dataset keeps it from outliving
        # the instruction and corpus it is valid under.
        stage1_cache: dict = {}

        # One cell per (workload size, point on the shared axis).
        for workload_q, epsilon_target in product(workload_queries, epsilon_targets):
            # The per-query budget that composes to `epsilon_target` over a stream of
            # `workload_q` questions, derived from this method's own knob so a target
            # and the value implementing it cannot drift apart. A target no budget
            # reaches is reported and skipped: clamping it would put a point on the
            # shared axis the mechanism never honoured.
            try:
                epsilon = epsilon_target if sweep_kind == "knob" else budget_for_target(
                    "dpksa",
                    target_epsilon=epsilon_target,
                    queries=workload_q,
                    delta=DEFAULT_DELTA,
                    mechanism_delta=mechanism_delta,
                    ptr_noise_coupling=ptr_noise_coupling,
                )
            except ValueError as exc:
                log(
                    f"[{dataset_name}] SKIP target={epsilon_target} Q={workload_q}: "
                    f"{exc} DP-KSA spends delta_ptr on every query whatever its budget, "
                    f"so at Q={workload_q} the additive delta is the binding "
                    f"constraint, not epsilon."
                )
                continue
            log(
                f"[{dataset_name}] RUN START: target={epsilon_target}, Q={workload_q}, "
                f"epsilon={epsilon}, n_docs={n_docs}, "
                f"workload_delta={DEFAULT_DELTA}, mechanism_delta={mechanism_delta}"
            )
            generator = DPKSAGenerator(
                model=model,
                tokenizer=tokenizer,
                target_epsilon=float(epsilon),
                target_delta=mechanism_delta,
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
                task_instruction=dataset.get_task_instruction(),
                use_chat_template=use_chat_template,
                stop_strings=stop_strings,
                stage1_cache=stage1_cache,
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
                    build_fn=lambda p=pipeline: p.index(documents),
                    cache_dir=cache_dir,
                )
                index_built = True

            # Standalone MIA mode: probe the live pipeline and save the raw
            # per-probe answers (scored offline by score_results.py). DP-KSA
            # calibrates its accounting per query, so no budget cloning is needed.
            if mia_only_enabled():
                k_ret = k_retrieved
                answer_fn = lambda q: pipeline.query(q, k=k_ret)
                mia_records = run_mia_probes(
                    answer_fn,
                    mia_split,
                    anderson_doc_maxlen=mia_params["anderson_doc_maxlen"],
                )
                payload = {
                    "method": "dpksa",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "dataset": dataset_name,
                    "llm": llm_model_name,
                    "max_index_docs": max_index_docs,
                    "epsilon_target": float(epsilon_target),
                    "workload_queries": int(workload_q),
                    "epsilon": float(epsilon),
                    "delta": mechanism_delta,
                    "calibration": generator.calibration,
                    "n_docs": n_docs,
                    "k_retrieved": k_retrieved,
                    # No workload block here: the MIA run issues attack probes rather
                    # than answering a query workload, so there is no Q to compose over
                    # -- and the utility path's `predictions` does not exist yet on this
                    # branch, so reading it lost the probes that had just been generated.
                    "seed": seed,
                    "mia_probes": mia_records,
                }
                output_path = _cell_output_path(
                    output_dir,
                    dataset_name,
                    llm_label if multi_llm else None,
                    workload_q,
                    epsilon_target,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                log(
                    f"[{dataset_name}] MIA: target={epsilon_target}, Q={workload_q}, "
                    f"epsilon={epsilon} — wrote {len(mia_records)} probes -> {output_path}"
                )
                del generator, pipeline
                free_memory()
                continue

            predictions: list[str] = []
            references: list[str] = []
            results: list[dict[str, Any]] = []
            ptr_pass_count = 0
            n_keywords_sum = 0
            query_times: list[float] = []
            selected_queries = select_queries(queries, max_queries=ds_max_queries)
            verbose_count = 0
            t_run_start = time.time()
            total_q = len(selected_queries)

            for q_idx, sample in enumerate(selected_queries, start=1):
                question = str(sample.get("query", "")).strip()
                reference = str(sample.get("answer", "")).strip()
                answers = [str(a).strip() for a in (sample.get("answers") or [reference]) if str(a).strip()] or [reference]
                if not question:
                    continue

                # Enable the generator's step-by-step mechanistic logging for the
                # first `verbose_max_examples` queries only.
                can_print_verbose = verbose and (verbose_max_examples is None or verbose_count < verbose_max_examples)
                generator.verbose = can_print_verbose
                generator.verbose_label = question

                t0 = time.time()
                try:
                    answer = pipeline.query(question, k=k_retrieved)
                finally:
                    generator.verbose = False
                elapsed = time.time() - t0

                stats = dict(generator.last_run_stats)
                ptr_pass_count += int(bool(stats.get("ptr_passed", False)))
                n_keywords_sum += int(stats.get("n_keywords", 0))

                predictions.append(str(answer))
                references.append(reference)
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
                        "k_hat": int(stats.get("k_hat", 0)),
                        "ptr_passed": bool(stats.get("ptr_passed", False)),
                        "keywords": list(stats.get("keywords", [])),
                    }
                )
                query_times.append(float(elapsed))

                if can_print_verbose:
                    print(f"Keywords (k_hat={stats.get('k_hat')}, ptr_passed={stats.get('ptr_passed')}): "
                          f"{stats.get('keywords')}")
                    print(f"Answer: {answer}")
                    if reference:
                        print(f"Reference: {reference}")
                    verbose_count += 1

                if q_idx % 10 == 0:
                    log(f"[{dataset_name} eps={epsilon}] heartbeat: query {q_idx}/{total_q}")

                free_memory()  # DP-KSA runs N generations per query: free GPU/host memory each query

            _n_done = len(predictions)
            _ptr_rate = (100.0 * ptr_pass_count / _n_done) if _n_done else 0.0
            _mean_kw = (n_keywords_sum / _n_done) if _n_done else 0.0
            log(
                f"[{dataset_name}] RUN END: target={epsilon_target}, Q={workload_q}, "
                f"epsilon={epsilon} — "
                f"{_n_done} queries in {time.time() - t_run_start:.1f}s | "
                f"PTR released keywords {ptr_pass_count}/{_n_done} ({_ptr_rate:.1f}%); "
                f"mean keywords/query={_mean_kw:.2f}"
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
            rouge_l = _safe_float(util.get("rouge_l", 0.0))
            ptr_pass_rate = _safe_float(ptr_pass_count / len(results)) if results else 0.0

            payload: dict[str, Any] = {
                "method": "dpksa",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset": dataset_name,
                "llm": llm_model_name,
                "max_index_docs": max_index_docs,
                # The cell's coordinates on the shared axis: the total the stream is
                # allowed to cost, the stream length it is priced for, and the
                # per-query budget those two imply.
                "epsilon_target": float(epsilon_target),
                "epsilon": float(epsilon),
                "delta": mechanism_delta,
                "n_docs": n_docs,
                "calibration": generator.calibration,
                "k_retrieved": k_retrieved,
                # Q is the declared workload the guarantee covers, not the number of
                # questions this cell happened to answer: the questions are the utility
                # sample and `max_queries` sizes them, while Q sets how much budget each
                # one gets. Passing the answered count would make the privacy number a
                # function of the sample size and no two cells would be comparable.
                **workload_accounting_block(
                    "dpksa", queries=workload_q, run_seed=seed,
                    gumbel_scale=float(generator.calibration["gumbel_scale"]),
                    sigma=float(generator.calibration["sigma"]),
                    delta_ptr=float(generator.calibration["delta_ptr"]),
                ),
                "questions_answered": _n_done,
                # The share of queries whose PTR test released keywords. At these
                # budgets it is what separates a DP-KSA result from a cell that
                # released nothing and answered every question zero-shot.
                "ptr_pass_rate": ptr_pass_rate,
                "predictions": predictions,
                "references": references,
                "results": results,
                "metrics": {**util, **priv},
                "match_accuracy": match_acc,
                "contains_accuracy": contains_acc,
                "rouge_l": rouge_l,
                "time_per_query_sec": time_per_query,
            }

            output_path = _cell_output_path(
                output_dir,
                dataset_name,
                llm_label if multi_llm else None,
                workload_q,
                epsilon_target,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "llm": llm_model_name,
                    "epsilon_target": float(epsilon_target),
                    "workload_queries": int(workload_q),
                    "epsilon": float(epsilon),
                    "match_accuracy": match_acc,
                    "contains_accuracy": contains_acc,
                    "rouge_l": rouge_l,
                    "ptr_pass_rate": ptr_pass_rate,
                    "time_per_query_sec": time_per_query,
                    "output": str(output_path),
                }
            )

            del generator, pipeline
            free_memory()  # release memory between cells

        del dataset, indexer, retriever
        free_memory()  # release the dataset index before the next dataset

    del model, tokenizer, embedder
    free_memory()

    return summary_rows


def main() -> None:
    setup_method_logging("dpksa")
    args = build_parser().parse_args()
    config = load_config(args.config)

    llm_configs = _iter_llm_configs(config)
    multi_llm = len(llm_configs) > 1
    # Run each LLM in its own subprocess so GPU memory is fully reclaimed between
    # models; returns None in the orchestrator once all LLMs have run as children.
    todo = resolve_llm_execution(llm_configs, method_name="dpksa")
    if todo is None:
        return
    summary_rows: list[dict[str, Any]] = []
    for _idx, llm_cfg in todo:
        llm_model_name = str(llm_cfg.get("model_name", "distilgpt2"))
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

    print("\nDP-KSA summary:")
    for row in summary_rows:
        print(
            f"- [{row['dataset']} | {row['llm']}] target={row['epsilon_target']} "
            f"Q={row['workload_queries']} eps={row['epsilon']}: "
            f"match_accuracy={row['match_accuracy']:.4f}, "
            f"contains_accuracy={row['contains_accuracy']:.4f}, "
            f"rouge_l={row['rouge_l']:.4f}, "
            f"ptr_pass_rate={row['ptr_pass_rate']:.2f}, "
            f"time_per_query_sec={row['time_per_query_sec']:.4f}, "
            f"out={row['output']}"
        )


if __name__ == "__main__":
    main()
