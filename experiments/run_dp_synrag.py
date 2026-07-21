"""Benchmark script for DP-SynRAG (index-time differential privacy via synthetic corpus).

Builds a differentially private synthetic document corpus once (Stages 1a–2e from the
paper), then serves all queries using a vanilla RAG pipeline over that corpus.

Reference: Anonymous et al., "DP-SynRAG: Differentially Private RAG via Synthetic
Document Generation", arXiv:2510.06719v2 (2025).

Usage:
    python experiments/run_dp_synrag.py --config experiments/configs/full/dp_synrag.yaml
    python experiments/run_dp_synrag.py --config experiments/configs/full/dp_synrag.yaml --verbose
"""

from __future__ import annotations

import argparse
import os
import atexit
import copy
import hashlib
import json
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.common import select_datasets
from experiments.mia_probe import mia_only_enabled, resolve_mia_params, run_mia_probes
from experiments.llm_isolation import resolve_llm_execution
from experiments.run_logging import log, setup_method_logging
from bench_datasets.chatdoctor import ChatDoctorDataset
from bench_datasets.hotpotqa import HotpotQADataset
from bench_datasets.medical_dirichlet_phi3 import MedicalDirichletPhi3Dataset
from bench_datasets.movielens import MovielensDataset
from bench_datasets.natural_questions import NaturalQuestionsDataset
from bench_datasets.searchqa import SearchQADataset
from bench_datasets.squad import SQuADDataset
from bench_datasets.triviaqa import TriviaQADataset
from core.device import free_memory, is_oom_error
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
    default_path = PROJECT_ROOT / "experiments" / "configs" / "full" / "dp_synrag.yaml"
    cfg_path = Path(config_path) if config_path else default_path
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("DP-SynRAG config YAML must define a dictionary.")
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
    ``dataset:`` block when no list is present. This lets a single YAML build a
    synthetic corpus for several datasets, each with its own ``max_samples``.
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


def _merge_overrides(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` into a copy of ``base`` (override wins at the leaves).

    Used to apply a dataset entry's ``overrides:`` block over the global config so
    each dataset can carry its own DP hyperparameters (the paper varies R, k,
    self_filter and inference.k_retrieved per dataset; reconstructed, not from the paper).
    """
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_overrides(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_dataset(dataset_cfg: dict[str, Any], seed: int):
    name = str(dataset_cfg.get("name", "searchqa")).lower()
    max_samples = dataset_cfg.get("max_samples")
    max_samples = int(max_samples) if max_samples is not None else None
    # Optional separate query cap; falls back to max_samples (corpus == queries).
    max_queries = dataset_cfg.get("max_test_queries")
    max_queries = int(max_queries) if max_queries is not None else max_samples
    split = str(dataset_cfg.get("split", "validation"))

    if name == "searchqa":
        dataset = SearchQADataset(
            index_split="train",
            test_split=split,
            max_index_docs=max_samples,
            max_test_queries=max_queries,
            random_seed=seed,
            # Opt-in support-binning filter (paper: >=40 answer-bearing snippets,
            # 6 bins, 17 per bin). Absent keys default to None => uniform sampling.
            min_answer_snippets=dataset_cfg.get("min_answer_snippets"),
            n_support_bins=dataset_cfg.get("n_support_bins"),
            per_bin=dataset_cfg.get("per_bin"),
            max_scan_questions=dataset_cfg.get("max_scan_questions"),
        )
    elif name == "movielens":
        dataset = MovielensDataset(
            max_index_docs=max_samples,
            max_test_queries=max_queries,
            random_seed=seed,
            # Opt-in restriction to the N most frequently rated movies (paper: 30).
            # Absent => None => full catalogue.
            top_movies=dataset_cfg.get("top_movies"),
        )
    elif name == "natural_questions":
        dataset = NaturalQuestionsDataset(
            index_split="train",
            test_split=split,
            max_index_docs=max_samples,
            max_test_queries=max_queries,
            random_seed=seed,
        )
    elif name == "squad":
        dataset = SQuADDataset(
            index_split="train",
            test_split=split,
            max_index_docs=max_samples,
            max_test_queries=max_queries,
            random_seed=seed,
        )
    elif name == "triviaqa":
        dataset = TriviaQADataset(
            index_split="train",
            test_split=split,
            max_index_docs=max_samples,
            max_test_queries=max_queries,
            random_seed=seed,
        )
    elif name == "hotpotqa":
        dataset = HotpotQADataset(
            index_split="train",
            test_split=split,
            max_index_docs=max_samples,
            max_test_queries=max_queries,
            random_seed=seed,
        )
    elif name == "chatdoctor":
        dataset = ChatDoctorDataset(
            index_split="train",
            test_split=split,
            max_index_docs=max_samples,
            max_test_queries=max_queries,
            random_seed=seed,
        )
    elif name == "medical_dirichlet_phi3":
        dataset = MedicalDirichletPhi3Dataset(
            index_split="train",
            test_split=split,
            max_index_docs=max_samples,
            max_test_queries=max_queries,
            random_seed=seed,
        )
    else:
        raise ValueError(f"Unsupported dataset for DP-SynRAG: '{name}'.")

    dataset.load()
    return dataset


def _synth_cache_dir(config: dict[str, Any], dataset_cfg: dict[str, Any]) -> Path:
    """Derive a stable temp path keyed on all DP hyperparams that affect the synthetic corpus."""
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
        "tau": sg_cfg.get("tau", 1.0),
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
        min_sim=float(cr_cfg.get("min_sim", -1.0)),
        max_sim=float(cr_cfg.get("max_sim", 1.0)),
        seed=seed,
    )
    prompt_template = str(sg_cfg.get(
        "prompt_template",
        # DPSYN-06: paper rephrase prompt fallback (kept in sync with dp_synrag.yaml).
        "Rephrase the following document without altering the important information "
        "contained within it.\n\nDocument:\n{document}\n\nRephrased document:\n",
    ))
    _vme = config.get("verbose_max_examples")
    synthetic_generator = DPSyntheticGenerator(
        model_name=llm_name,
        T=int(sg_cfg.get("T", 30)),
        c=float(sg_cfg.get("c", 1.0)),
        tau=float(sg_cfg.get("tau", 1.0)),
        prompt_template=prompt_template,
        # Clusters with <= batch_threshold docs take the batched logits path, which
        # materializes a (n_docs, seq_len, vocab) tensor; on long documents that is
        # several GiB. 0 forces the sequential path (logits_to_keep=1). Same math.
        batch_threshold=int(sg_cfg.get("batch_threshold", 8)),
        seed=seed,
        # Step-by-step mechanistic logging for the first N synthetic documents.
        verbose=bool(config.get("verbose", False)),
        verbose_max_examples=int(_vme) if _vme is not None else None,
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


def _run_dataset(
    config: dict[str, Any],
    dataset_cfg: dict[str, Any],
    embedder: HFEmbedder,
    seed: int,
    llm_label: str = "",
    multi_llm: bool = False,
    c_subdir: str | None = None,
) -> None:
    dataset = _resolve_dataset(dataset_cfg, seed)
    documents = dataset.get_documents()
    queries = dataset.get_queries()

    # Standalone MIA mode: synthesize ONLY the member corpus so the offline scorer
    # can distinguish member (indexed) from held-out non-member documents. Swap the
    # synthesis input to the member subset before the (uncached, in-memory) build.
    mia_split = None
    if mia_only_enabled():
        mia_params = resolve_mia_params(config)
        mia_split = dataset.get_mia_split(
            n_per_class=mia_params["n_per_class"],
            index_size=mia_params["index_size"],
            seed=mia_params["seed"],
        )
        if mia_split is None:
            log(f"[{dataset_cfg.get('name', 'unknown')}] MIA: dataset has no member/non-member split; skipping.")
            return
        documents = mia_split["index_documents"]

    indexer = _build_indexer(config, embedder, seed)

    cache_dir = _synth_cache_dir(config, dataset_cfg)
    task_prefix = str(config.get("synthetic_generation", {}).get("task_prefix", ""))

    if mia_only_enabled():
        # Never serve a cached synthetic corpus in MIA mode: _synth_cache_dir keys on
        # the config, not on the member subset, so a utility-run (full-corpus) index of
        # the same config would otherwise be loaded here and make members and
        # non-members indistinguishable. Always synthesize the member subset fresh.
        log(f"Synthetic corpus BUILD START from {len(documents)} member documents (MIA, uncached)")
        _t_build = time.time()
        indexer.build_from_documents(documents, task_prefix=task_prefix)
        log(f"Synthetic corpus BUILD END in {time.time() - _t_build:.1f}s")
    elif cache_dir.exists():
        log(f"Synthetic corpus: loading cached index from {cache_dir}")
        indexer.load(str(cache_dir))
    else:
        log(f"Synthetic corpus BUILD START from {len(documents)} documents")
        _t_build = time.time()
        indexer.build_from_documents(documents, task_prefix=task_prefix)
        indexer.save(str(cache_dir))
        atexit.register(shutil.rmtree, str(cache_dir), ignore_errors=True)
        log(f"Synthetic corpus BUILD END in {time.time() - _t_build:.1f}s (cached at {cache_dir}, deleted on exit)")

    stats = indexer.last_stats
    print(
        f"Synthetic corpus: {stats.get('n_original_documents', '?')} original docs → "
        f"{stats.get('n_non_empty_subsets', '?')} non-empty clusters → "
        f"{stats.get('n_synthetic_kept', '?')} synthetic docs"
    )
    # Noise impact of the private-prediction stage (Stage 2d): how often the
    # clipped-logit sampler diverged from the pre-noise argmax(z_n).
    _sg = stats.get("synthetic_generator", {}) if isinstance(stats, dict) else {}
    _sg_tokens = int(_sg.get("n_tokens_total", 0))
    _sg_flipped = int(_sg.get("n_noise_flipped_total", 0))
    if _sg_tokens:
        log(
            f"Synthetic generation: DP noise flipped {_sg_flipped}/{_sg_tokens} "
            f"sampled tokens ({100.0 * _sg_flipped / _sg_tokens:.1f}%) over the corpus"
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
        tau=float(sg_cfg.get("tau", 1.0)),
        delta=delta,
    )
    print(print_zcdp_analysis(zcdp_result))

    # Vanilla inference over the synthetic corpus
    retriever = CosineSimilarityRetriever(indexer=indexer)
    reranker = IdentityReranker()
    # Per-dataset prompt: the task instruction comes from the dataset class
    # (same prompt across methods for a given dataset -> fair comparison). The
    # template ends right at "Answer:" so the model continues the answer.
    prompt_builder = SimplePromptBuilder(template=dataset.get_prompt_template())
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

    # At a tight budget the self-filter can reject every synthetic document, so
    # there is no index to search. Answer from an empty context instead: this is
    # DP-SynRAG degrading to no-RAG, a real point on the epsilon-utility curve.
    corpus_empty = indexer.index is None

    def _answer_without_retrieval(query: str) -> str:
        return postproc.process(generator.generate(prompt_builder.build(query, [])))

    # Standalone MIA mode: issue the attack probes over the live synthetic corpus
    # instead of the utility QA loop. The answers are scored offline; the saved
    # payload also carries `synthetic_texts` so the scorer can compute the retention
    # AUC (whether member input docs survived into the generated corpus). Honor the
    # empty-corpus guard so a fully self-filtered synthetic corpus does not crash.
    if mia_only_enabled():
        answer_fn = lambda q: (
            pipeline.query(q, k=k_retrieved) if not corpus_empty else _answer_without_retrieval(q)
        )
        log(
            f"MIA START: probing {len(mia_split.get('members', []))} members + "
            f"{len(mia_split.get('non_members', []))} non-members over the synthetic corpus"
        )
        mia_records = run_mia_probes(
            answer_fn,
            mia_split,
            anderson_doc_maxlen=resolve_mia_params(config)["anderson_doc_maxlen"],
        )
        dataset_name = str(dataset_cfg.get("name", "unknown"))
        mia_payload = {
            "method": "dp_synrag",
            "dataset": dataset_name,
            "epsilon_total": zcdp_result.epsilon,
            "privacy_accounting": {
                "zcdp": {
                    "epsilon": zcdp_result.epsilon,
                    "delta": zcdp_result.delta,
                    "rho": zcdp_result.rho,
                    "components": zcdp_result.components,
                }
            },
            "llm": str(config["llm"]["model_name"]),
            "embedder": str(config["models"]["embedder"]),
            "seed": seed,
            "mia_probes": mia_records,
            # Offline scorer computes the retention AUC from these (did member input
            # docs survive into the generated corpus).
            "synthetic_texts": indexer.synthetic_texts,
            "n_synthetic_docs": len(indexer.synthetic_texts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        output_dir = Path(str(config.get("output_dir", "results/dp_synrag/"))) / dataset_name
        if multi_llm:
            output_dir = output_dir / llm_label
        if c_subdir:
            output_dir = output_dir / c_subdir
        out_path = output_dir / "run.json"
        _save_results(out_path, mia_payload)
        log(f"MIA END: {len(mia_records)} probe records saved to {out_path}")
        print(f"Saved: {out_path}")
        return

    predictions: list[str] = []
    references: list[str] = []
    reference_sets: list[list[str]] = []  # per-item gold sets for hit_rate (Movielens)
    results: list[dict[str, str]] = []
    verbose_count = 0
    if corpus_empty:
        log("Inference START with an EMPTY synthetic corpus: retrieval disabled, "
            f"{len(queries)} queries answered from parametric knowledge only")
    else:
        log(f"Inference START over synthetic corpus: {len(queries)} queries, k_retrieved={k_retrieved}")
    _t_infer = time.time()
    _total_q = len(queries)

    for q_idx, item in enumerate(tqdm(queries, desc="DP-SynRAG inference"), start=1):
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        reference = str(item.get("answer", "")).strip()
        answers = [str(a).strip() for a in (item.get("answers") or [reference]) if str(a).strip()] or [reference]

        can_print_verbose = verbose and (
            verbose_max_examples is None or verbose_count < verbose_max_examples
        )

        if can_print_verbose:
            if corpus_empty:
                retrieved_chunks = []
                response = _answer_without_retrieval(query)
            else:
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
        elif corpus_empty:
            response = _answer_without_retrieval(query)
        else:
            response = pipeline.query(query, k=k_retrieved)

        predictions.append(str(response))
        references.append(reference)
        favorites = item.get("favorites")
        reference_sets.append([str(x) for x in favorites] if favorites else [reference])
        result_item = {"query": query, "prediction": str(response), "reference": reference, "answers": answers}
        if favorites:
            result_item["favorites"] = [str(x) for x in favorites]
        results.append(result_item)
        if q_idx % 10 == 0:
            log(f"[DP-SynRAG inference] heartbeat: query {q_idx}/{_total_q}")

    log(f"Inference END: {len(predictions)} queries in {time.time() - _t_infer:.1f}s")

    utility_metric_names = config.get("metrics", {}).get("utility", [])
    privacy_metric_names = config.get("metrics", {}).get("privacy", [])

    # Deferred (orchestrated run): skip in-process metrics; score_results.py
    # computes them later in a clean process (avoids BERTScore/perplexity OOM).
    if os.environ.get("BENCHDP_DEFER_METRICS") == "1":
        utility_metrics, privacy_metrics = {}, {}
    else:
        utility_metrics = compute_utility_metrics(
            predictions,
            references,
            utility_metric_names,
            texts=indexer.synthetic_texts,  # used by perplexity metric
            reference_sets=reference_sets,  # used by hit_rate metric (Movielens)
        )
        privacy_metrics = compute_privacy_metrics(
            predictions,
            references,
            privacy_metric_names,
            seed=seed,
        )
    metrics_payload = {**utility_metrics, **privacy_metrics}

    dataset_name = str(dataset_cfg.get("name", "unknown"))
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
        "tau": float(sg_cfg.get("tau", 1.0)),
        "delta": delta,
        "k_retrieved": k_retrieved,
        "embedder": str(config["models"]["embedder"]),
        "llm": str(config["llm"]["model_name"]),
    }

    payload = {
        "method": "dp_synrag",
        "dataset": dataset_name,
        # Top-level epsilon so the offline scorer (score_results.py _csv_row reads
        # epsilon_total -> epsilon) records it, like the other run scripts; the full
        # breakdown still lives under privacy_accounting.zcdp below.
        "epsilon_total": zcdp_result.epsilon,
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
        "synthetic_texts": indexer.synthetic_texts,  # for offline perplexity scoring
        "n_queries": len(predictions),
        "n_synthetic_docs": len(indexer.synthetic_texts),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    output_dir = Path(str(config.get("output_dir", "results/dp_synrag/"))) / dataset_name
    if multi_llm:
        output_dir = output_dir / llm_label
    if c_subdir:
        output_dir = output_dir / c_subdir
    out_path = output_dir / "run.json"
    _save_results(out_path, payload)
    print(f"Saved: {out_path}")
    print(f"ε = {zcdp_result.epsilon:.4f}, δ = {zcdp_result.delta:.1e}, ρ = {zcdp_result.rho:.4f}")
    print(f"Metrics: {metrics_payload}")


def main(
    config_path: str | None = None,
    verbose_override: bool | None = None,
    verbose_max_examples_override: int | None = None,
) -> None:
    setup_method_logging("dp_synrag")
    config = load_config(config_path)
    if verbose_override is not None:
        config["verbose"] = bool(verbose_override)
    if verbose_max_examples_override is not None:
        config["verbose_max_examples"] = int(verbose_max_examples_override)

    seed = int(config.get("random_seed", 42))
    set_seed(seed)

    # The embedder is dataset- and LLM-independent; build it once and reuse it.
    embedder = HFEmbedder(model_name=str(config["models"]["embedder"]))

    llm_configs = _iter_llm_configs(config)
    multi_llm = len(llm_configs) > 1
    todo = resolve_llm_execution(llm_configs, method_name="dp_synrag")
    if todo is None:
        return
    for _idx, llm_cfg in todo:
        # Override the singular ``llm`` block so every downstream read
        # (_build_indexer, generator, hyperparams, synth-cache key) picks up the
        # current model. The synthetic-corpus cache is keyed on the model name,
        # so each LLM gets its own cached corpus.
        config["llm"] = llm_cfg
        llm_label = _llm_label(llm_cfg)
        if multi_llm:
            print(f"\n##### LLM: {llm_cfg.get('model_name', '')} #####")

        for dataset_cfg in select_datasets(_iter_dataset_configs(config)):
            # Each `datasets:` entry may carry an `overrides:` block to set the
            # per-dataset DP hyperparameters the paper varies (clustering.R,
            # cluster_retrieval.k, self_filter.enabled, inference.k_retrieved, …);
            # they deep-merge over the global blocks for this dataset only.
            overrides = dataset_cfg.get("overrides") or {}
            base_ds_config = _merge_overrides(config, overrides) if overrides else config
            # Sweep the softmax-sharpness c: lower c => tighter (smaller) reported epsilon
            # but flatter/noisier synthetic text. Each c is its own DP operating point (the
            # synth-cache key and the reported epsilon already include c). A single value
            # (or omitting `c_values`, falling back to `c`) leaves the output path unchanged.
            _sg = base_ds_config.get("synthetic_generation", {})
            c_values = [float(x) for x in (_sg.get("c_values") or [_sg.get("c", 1.0)])]
            sweep_c = len(c_values) > 1
            for c_val in c_values:
                ds_config = copy.deepcopy(base_ds_config)
                ds_config.setdefault("synthetic_generation", {})["c"] = c_val
                c_subdir = f"c_{c_val:g}" if sweep_c else None
                log(
                    f"Dataset START: {dataset_cfg.get('name', 'unknown')} "
                    f"(c={c_val:g}, max_samples={dataset_cfg.get('max_samples')})"
                )
                try:
                    _run_dataset(ds_config, dataset_cfg, embedder, seed, llm_label, multi_llm, c_subdir=c_subdir)
                except OSError as exc:
                    # A model we cannot load (e.g. gated HF repo without access, or
                    # an offline/missing checkpoint) should not abort the whole
                    # multi-LLM run; skip this LLM/dataset and keep going.
                    print(
                        f"\n[WARN] skipping LLM {llm_cfg.get('model_name', '')} "
                        f"on dataset {dataset_cfg.get('name', 'unknown')}: {exc}"
                    )
                except Exception as exc:
                    # Any failure on one (model, dataset, c) — OOM or otherwise — should
                    # skip that operating point and free memory, not abort the whole sweep.
                    tag = "OOM — " if is_oom_error(exc) else ""
                    print(
                        f"\n[WARN] {tag}skipping LLM {llm_cfg.get('model_name', '')} "
                        f"on dataset {dataset_cfg.get('name', 'unknown')}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                free_memory()  # release the synthetic index / LLMs before the next (c, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DP-SynRAG experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "configs" / "full" / "dp_synrag.yaml"),
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
