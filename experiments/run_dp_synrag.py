"""Benchmark script for DP-SynRAG (index-time differential privacy via synthetic corpus).

Builds a differentially private synthetic document corpus once (Stages 1a–2e from the
paper), then serves all queries using a vanilla RAG pipeline over that corpus.

The sweep is the shared axis crossed with the workload size: a target total epsilon
(``synthetic_generation.epsilon_targets``) and the number of questions Q the guarantee
has to cover (``workload_queries``). The corpus is built before any question is asked
and every answer is post-processing of it, so this method's cost does not depend on Q:
one build per target serves the whole workload column, and the Q axis only changes how
the same run is priced.

Reference: Anonymous et al., "DP-SynRAG: Differentially Private RAG via Synthetic
Document Generation", arXiv:2510.06719v2 (2025).

Usage:
    python experiments/run_dp_synrag.py --config experiments/configs/full/dp_synrag.yaml
    python experiments/run_dp_synrag.py --config experiments/configs/full/dp_synrag.yaml --verbose
"""

from __future__ import annotations

import argparse
import os
import copy
import hashlib
import json
import random
import shutil
import sys
import tempfile
import time
import traceback
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

from experiments.common import (
    build_chunker,
    workload_accounting_block,
    load_dataset_from_config,
    resolve_seeds,
    select_datasets,
    select_queries,
)
from experiments.extraction_cache import CachedKeywordExtractor
from experiments.mia_probe import mia_only_enabled, resolve_mia_params, run_mia_probes
from experiments.llm_isolation import resolve_llm_execution
from experiments.run_logging import log, setup_method_logging
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
from metrics.workload_accounting import DEFAULT_DELTA, budget_for_target
from metrics.utility import compute_utility_metrics
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
    synthetic corpus for several datasets, each with its own ``max_index_docs``.
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


#: Written after `indexer.save`, so a build interrupted partway is rebuilt rather than
#: loaded back as a finished synthetic corpus.
_SYNTH_COMPLETE = ".synth_complete"

#: Fallbacks for the stage parameters, in one place. They are read from eleven call
#: sites — the calibration, the accounting, the cache key, the indexer — and a config
#: omitting one would otherwise be calibrated against a different value than it is
#: priced and cached at, silently. The shipped configs set all of them; these are what
#: a partial config falls back to.
_DEFAULT_L = 5
_DEFAULT_T = 70
_DEFAULT_DELTA = 1e-2


def _sweep_points(config: dict[str, Any], sg_cfg: dict[str, Any]) -> list[dict[str, float]]:
    """One operating point per sweep entry, with all four stage parameters.

    ``epsilon_targets`` names points on the shared axis and derives the parameters that
    reach them, so a target cannot drift from the values that implement it. The four
    stages divide one zCDP budget, and the calibration scales them together; listing
    only ``c`` in the config leaves the three noise scales wherever they were and puts
    a floor under the reachable epsilon.

    ``c_values`` remains for the paper-faithful sweep, where c is the knob and the
    scales are the paper's.

    The workload size is absent from every point: the four stages run before a single
    question is asked, so nothing here depends on how many questions the guarantee is
    asked to cover. A target the calibration cannot reach is logged and dropped rather
    than clamped -- a clamped point would sit on the shared axis at an epsilon the
    mechanism never honoured.
    """
    targets = sg_cfg.get("epsilon_targets")
    if targets:
        from metrics.privacy_accounting_zcdp import calibrate_dp_synrag

        cl = config.get("clustering", {})
        cr = config.get("cluster_retrieval", {})
        kw = config.get("keyword_extractor", {})
        points = []
        for target in targets:
            try:
                cal = calibrate_dp_synrag(
                    K=int(kw.get("K", 10)),
                    sigma_h=float(cl.get("sigma_h", 1.0)),
                    L=int(cl.get("L", _DEFAULT_L)),
                    epsilon_theta_s=float(cr.get("epsilon_theta_s", 1.0)),
                    sigma_mu=float(cr.get("sigma_mu", 0.5)),
                    T=int(sg_cfg.get("T", _DEFAULT_T)),
                    c=float(sg_cfg.get("c", 1.0)),
                    tau=float(sg_cfg.get("tau", 1.0)),
                    target_epsilon=float(target),
                    delta=float(config.get("delta", _DEFAULT_DELTA)),
                    # Calibrated against the rho the SHARED AXIS prices, not the one the
                    # classic zCDP conversion implies. The two differ — the axis
                    # converts through the Renyi curve and lands tighter — so
                    # calibrating on the classic form put this method's cells up to 38%
                    # below the target every other method hits exactly, which is the
                    # one thing a common axis cannot afford.
                    target_rho=budget_for_target(
                        "dp_synrag", target_epsilon=float(target), queries=1,
                        delta=DEFAULT_DELTA,
                    ),
                )
            except (ValueError, ZeroDivisionError) as exc:
                log(f"SKIP epsilon_target={target}: {type(exc).__name__}: {exc}")
                continue
            points.append({
                "epsilon_target": float(target),
                "c": float(cal["c"]),
                "sigma_h": float(cal["sigma_h"]),
                "sigma_mu": float(cal["sigma_mu"]),
                "epsilon_theta_s": float(cal["epsilon_theta_s"]),
            })
        return points

    values = [float(x) for x in (sg_cfg.get("c_values") or [sg_cfg.get("c", 1.0)])]
    return [
        {
            # c is the knob here, so the point is named by the value that implements
            # it and no target is claimed.
            "epsilon_target": None,
            "c": v,
            "sigma_h": float(config.get("clustering", {}).get("sigma_h", 1.0)),
            "sigma_mu": float(config.get("cluster_retrieval", {}).get("sigma_mu", 0.5)),
            "epsilon_theta_s": float(
                config.get("cluster_retrieval", {}).get("epsilon_theta_s", 1.0)
            ),
        }
        for v in values
    ]


def _synth_cache_dir(
    config: dict[str, Any], dataset_cfg: dict[str, Any], run_seed: int
) -> Path:
    """Derive a stable temp path keyed on all DP hyperparams that affect the synthetic corpus.

    ``run_seed`` belongs in the key because for this method the synthetic corpus *is*
    the mechanism's output — the noisy clustering and the budgeted generation happen
    while building it. Keying on the config's seed instead would hand every repetition
    the corpus the first one produced, so a sweep over seeds would differ only in the
    reader's token sampling and would report a mechanism variance of about zero.

    Every field below changes that output, so a key that misses one serves a corpus
    built under different privacy parameters. What is *not* here is the run id: the
    same key means the same corpus whichever invocation produced it, and the marker
    file distinguishes a finished build from an abandoned one.

    The workload size is deliberately absent as well, and the key reads only the config
    blocks the build itself consumes so it cannot arrive by accident. The build happens
    before any question is asked, so the corpus every workload size is priced against is
    literally the same corpus; keying on Q would rebuild it once per column and turn the
    benchmark's most expensive stage into four identical copies of itself.
    """
    kw_cfg = config.get("keyword_extractor", {})
    cl_cfg = config.get("clustering", {})
    cr_cfg = config.get("cluster_retrieval", {})
    sg_cfg = config.get("synthetic_generation", {})
    sf_cfg = config.get("self_filter", {})

    key_dict = {
        "dataset_name": dataset_cfg.get("name", "searchqa"),
        "max_index_docs": dataset_cfg.get("max_index_docs"),
        "run_seed": run_seed,
        # What selects the INPUT documents: two corpora drawn with different seeds or
        # splits are different data, and must not share an entry.
        "data_seed": config.get("data_seed", config.get("random_seed", 42)),
        "test_split": dataset_cfg.get("test_split", dataset_cfg.get("split")),
        "self_filter_prompt": (config.get("self_filter", {}) or {}).get("prompt_template"),
        "max_document_words": (config.get("chunking", {}) or {}).get("max_document_words"),
        "embedder": config.get("models", {}).get("embedder", ""),
        "llm": config.get("llm", {}).get("model_name", ""),
        "K": kw_cfg.get("K", 10),
        # These three also decide the extraction, and their absence meant a change to
        # any of them was served the previous vocabulary's corpus without a word.
        "vocab_min_word_len": kw_cfg.get("vocab_min_word_len", 3),
        "kw_exclude_stopwords": kw_cfg.get("exclude_stopwords", True),
        "kw_max_new_tokens": kw_cfg.get("max_new_tokens", 50),
        "R": cl_cfg.get("R", 50),
        "L": cl_cfg.get("L", _DEFAULT_L),
        "sigma_h": cl_cfg.get("sigma_h", 1.0),
        "k_cluster": cr_cfg.get("k", 10),
        "sigma_mu": cr_cfg.get("sigma_mu", 0.5),
        "epsilon_theta_s": cr_cfg.get("epsilon_theta_s", 1.0),
        "T": sg_cfg.get("T", _DEFAULT_T),
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

    # Stage 1a is deterministic (greedy) and runs before any DP noise, so its output
    # is shared across every budget and seed the sweep visits; see extraction_cache.
    keyword_extractor = CachedKeywordExtractor(
        KeywordExtractor(
            model_name=llm_name,
            K=int(kw_cfg.get("K", 10)),
            vocab=KeywordExtractor.build_default_vocab(
                min_word_len=int(kw_cfg.get("vocab_min_word_len", 3)),
                exclude_stopwords=bool(kw_cfg.get("exclude_stopwords", True)),
            ),
            max_new_tokens=int(kw_cfg.get("max_new_tokens", 50)),
            seed=seed,
        )
    )
    clusterer = DPHistogramClusterer(
        R=int(cl_cfg.get("R", 50)),
        L=int(cl_cfg.get("L", _DEFAULT_L)),
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
        T=int(sg_cfg.get("T", _DEFAULT_T)),
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
    data_seed: int,
    llm_label: str = "",
    multi_llm: bool = False,
    c_subdir: str | None = None,
    workload_sizes: tuple[int, ...] = (100,),
    epsilon_target: float | None = None,
) -> None:
    dataset = load_dataset_from_config(
        dataset_cfg, data_seed,
        default_max_queries=dataset_cfg.get("max_test_queries"),
    )
    # The document cap has to be applied here, not through the pipeline: DP-SynRAG
    # indexes with `indexer.build_from_documents`, and `RAGPipeline.chunker` is only
    # consulted by `RAGPipeline.index()`, which this method never calls. Handing the
    # chunker to the pipeline therefore did nothing, and the input corpus reached the
    # synthesis prompt at full length -- exactly where a 4800-word MovieLens profile
    # caused the documented OOM.
    _chunker = build_chunker(config)
    documents = [_chunker.chunk(doc)[0] for doc in dataset.get_documents()]
    queries = select_queries(
        dataset.get_queries(), max_queries=dataset_cfg.get("max_test_queries")
    )

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

    cache_dir = _synth_cache_dir(config, dataset_cfg, seed)
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
    elif (cache_dir / _SYNTH_COMPLETE).exists():
        log(f"Synthetic corpus: loading cached index from {cache_dir}")
        indexer.load(str(cache_dir))
    else:
        # A directory without the marker was left by a build that died partway. Its
        # contents are a fragment of a synthetic corpus, so they are discarded rather
        # than loaded: half a mechanism's output carries no guarantee at all.
        if cache_dir.exists():
            log(f"Synthetic corpus: discarding an incomplete build at {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)
        log(f"Synthetic corpus BUILD START from {len(documents)} documents")
        _t_build = time.time()
        indexer.build_from_documents(documents, task_prefix=task_prefix)
        indexer.save(str(cache_dir))
        (cache_dir / _SYNTH_COMPLETE).write_text("", encoding="utf-8")
        log(f"Synthetic corpus BUILD END in {time.time() - _t_build:.1f}s (cached at {cache_dir})")

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
    delta = float(config.get("delta", _DEFAULT_DELTA))

    zcdp_result = zcdp_dp_synrag(
        K=int(kw_cfg.get("K", 10)),
        sigma_h=float(cl_cfg.get("sigma_h", 1.0)),
        L=int(cl_cfg.get("L", _DEFAULT_L)),
        epsilon_theta_s=float(cr_cfg.get("epsilon_theta_s", 1.0)),
        sigma_mu=float(cr_cfg.get("sigma_mu", 0.5)),
        T=int(sg_cfg.get("T", _DEFAULT_T)),
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
    # The answer returned to the user. Held at one value across every method, so a
    # text metric measures the answer rather than the length it was allowed to fill.
    inference_max_new_tokens = int(config.get("inference", {}).get("max_new_tokens", 64))
    generator = HFGenerator(
        model_name=str(config["llm"]["model_name"]),
        max_new_tokens=inference_max_new_tokens,
    )
    postproc = IdentityPostprocessor()

    pipeline = RAGPipeline(
        chunker=build_chunker(config),
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
        result_item = {
            "query": query,
            "query_id": item.get("query_id", ""),
            "prediction": str(response),
            "reference": reference,
            "answers": answers,
        }
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
        "L": int(cl_cfg.get("L", _DEFAULT_L)),
        "sigma_h": float(cl_cfg.get("sigma_h", 1.0)),
        "k_cluster": int(cr_cfg.get("k", 10)),
        "sigma_mu": float(cr_cfg.get("sigma_mu", 0.5)),
        "epsilon_theta_s": float(cr_cfg.get("epsilon_theta_s", 1.0)),
        "T": int(sg_cfg.get("T", _DEFAULT_T)),
        "c": float(sg_cfg.get("c", 1.0)),
        "tau": float(sg_cfg.get("tau", 1.0)),
        "delta": delta,
        "k_retrieved": k_retrieved,

        "embedder": str(config["models"]["embedder"]),
        "llm": str(config["llm"]["model_name"]),
    }

    output_dir = Path(str(config.get("output_dir", "results/dp_synrag/"))) / dataset_name
    if multi_llm:
        output_dir = output_dir / llm_label
    if c_subdir:
        output_dir = output_dir / c_subdir

    # One inference run, recorded once per workload size. Q is a property of the
    # guarantee, not of the measurement: the questions answered above are the utility
    # sample and stay whatever `max_test_queries` asked for, while Q says how long a
    # stream the epsilon has to cover. Here the two never interact, because the budget
    # was spent building the corpus and answering from it depletes nothing -- so every
    # workload size carries the same utility and the same epsilon.
    #
    # That flatness is the finding, and it is only visible if the rows exist at every Q
    # to be flat across. A single record filed under one Q disappears from every panel
    # drawn at another: the per-query methods are read at Q=100 and this method, filed
    # at Q=1, was simply absent from that comparison rather than shown as unchanged.
    # So the identical result is written at each size, each record naming its siblings.
    #
    # The generated corpus and the per-question results are written ONCE, in the record
    # for the smallest size, and the others point at it: they are identical by
    # construction — one inference run — and a synthetic corpus stored four times per
    # (dataset, LLM, target) is four copies of the same megabytes. Everything a reader
    # or a per-Q filter needs is in every record; only the bulk is shared.
    canonical_q = min(workload_sizes)
    canonical_name = f"q{canonical_q}.json"
    for workload_q in workload_sizes:
        payload = {
            "method": "dp_synrag",
            "dataset": dataset_name,
            # The point on the shared axis this run implements, next to the four stage
            # parameters derived from it, so a target and the values that reach it
            # cannot drift apart in the record.
            "epsilon_target": epsilon_target,
            # Every workload size this measurement is filed at. The cost is paid before
            # any question is asked, so these records are copies of one another and a
            # reader who finds several must not read them as repeated measurements.
            "workload_queries_covered": [int(q) for q in workload_sizes],
            "c": float(sg_cfg.get("c", 1.0)),
            "sigma_h": float(cl_cfg.get("sigma_h", 1.0)),
            "sigma_mu": float(cr_cfg.get("sigma_mu", 0.5)),
            "epsilon_theta_s": float(cr_cfg.get("epsilon_theta_s", 1.0)),
            # This method's own headline, the classic closed-form zCDP conversion
            # rho + sqrt(4 rho ln(1/delta)), at the same top level the other run scripts
            # write theirs. It is NOT the shared axis: the axis converts the same rho
            # through the Renyi curve and lands tighter, by 48% at target 1 down to 3% at
            # target 50, so it is reported (as `epsilon_method_own` /
            # `epsilon_classic_zcdp`) beside `epsilon_workload` rather than in place of
            # it. The full breakdown still lives under privacy_accounting.zcdp below.
            "epsilon_total": zcdp_result.epsilon,
            # Top level, like every other method: the analysis layer reads the
            # shared-axis fields from the payload root, and nesting them left DP-SynRAG
            # off the common epsilon axis entirely and its repetitions without a seed
            # label. `queries` is the declared workload, never the number of questions
            # answered, which would tie the privacy axis to how many questions happened
            # to be sampled; `n_queries` below keeps that count.
            **workload_accounting_block(
                "dp_synrag", queries=workload_q, run_seed=seed,
                answer_tokens=inference_max_new_tokens,
                rho=zcdp_result.rho,
            ),
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

        if workload_q != canonical_q:
            # The bulk lives in the canonical record; this one names it so the offline
            # scorer can read the predictions it must score and the corpus it must
            # measure perplexity over. Popped by name rather than rebuilt field by field,
            # so a field added to the payload above cannot be forgotten here.
            for heavy in ("synthetic_texts", "results"):
                payload.pop(heavy, None)
            payload["shared_payload"] = canonical_name

        out_path = output_dir / f"q{workload_q}.json"
        _save_results(out_path, payload)
        print(f"Saved: {out_path}")
        print(
            f"Q={workload_q}: ε_workload = {payload.get('epsilon_workload')}, "
            f"regime = {payload.get('workload_regime')}"
        )

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

    # data_seed picks the corpus and the questions; run_seed drives the DP noise, so
    # a repetition varies the mechanism without resampling or rebuilding the index.
    data_seed, seed = resolve_seeds(config)
    set_seed(seed)

    # The embedder is dataset- and LLM-independent; build it once and reuse it.
    embedder = HFEmbedder(model_name=str(config["models"]["embedder"]))

    # Workload sizes the shared axis is crossed with. A method that pays per query has
    # to spend less on each question as the stream grows; this one spends its budget
    # before the first question, so the same run answers every workload size and each
    # is priced from the corpus already in hand rather than from another build.
    workload_sizes = tuple(int(q) for q in (config.get("workload_queries") or [100]))

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
            points = _sweep_points(base_ds_config, _sg)
            sweep_c = len(points) > 1
            for point in points:
                ds_config = copy.deepcopy(base_ds_config)
                ds_config.setdefault("synthetic_generation", {})["c"] = point["c"]
                # The four stages share one budget, so a point on the axis moves all
                # four together. Writing only `c` into the config left the three noise
                # scales at the previous point's calibration and the reported epsilon
                # stopped matching the target -- and with the scales pinned there is a
                # floor below which no c can reach, so the tightest points were not
                # merely wrong but unreachable.
                ds_config.setdefault("clustering", {})["sigma_h"] = point["sigma_h"]
                cr = ds_config.setdefault("cluster_retrieval", {})
                cr["sigma_mu"] = point["sigma_mu"]
                cr["epsilon_theta_s"] = point["epsilon_theta_s"]
                c_val = point["c"]
                # Named by the axis coordinate, not by the calibrated c. Two targets
                # whose c agrees to six significant digits would share a directory, and
                # a run where the calibration drops a point writes to a different path
                # than a complete one — both silent. The target is what identifies the
                # cell, and it is stable whatever the calibration returns. The
                # paper-faithful sweep has no target, so it keeps naming itself by c.
                target = point.get("epsilon_target")
                if not sweep_c:
                    c_subdir = None
                elif target is not None:
                    c_subdir = f"target_{target:g}"
                else:
                    c_subdir = f"c_{c_val:g}"
                # One corpus per point, serving every workload size: the point is the
                # whole of what the build depends on, so the loop over Q lives inside
                # `_run_dataset`, after the corpus and the answers exist.
                log(
                    f"Dataset START: {dataset_cfg.get('name', 'unknown')} "
                    f"(epsilon_target={point['epsilon_target']}, "
                    f"c={c_val:g}, sigma_h={point['sigma_h']:.4g}, "
                    f"sigma_mu={point['sigma_mu']:.4g}, "
                    f"eps_theta_s={point['epsilon_theta_s']:.4g}, "
                    f"workload_queries={list(workload_sizes)}, "
                    f"max_index_docs={dataset_cfg.get('max_index_docs')})"
                )
                try:
                    _run_dataset(
                        ds_config, dataset_cfg, embedder, seed, data_seed,
                        llm_label, multi_llm, c_subdir=c_subdir,
                        workload_sizes=workload_sizes,
                        epsilon_target=point["epsilon_target"],
                    )
                except OSError as exc:
                    # A model we cannot load (e.g. gated HF repo without access, or
                    # an offline/missing checkpoint) should not abort the whole
                    # multi-LLM run; skip this LLM/dataset and keep going.
                    print(
                        f"\n[WARN] skipping LLM {llm_cfg.get('model_name', '')} "
                        f"on dataset {dataset_cfg.get('name', 'unknown')}: {exc}"
                    )
                except Exception as exc:
                    # A run out of memory on one cell should cost that cell, not the
                    # sweep. Anything else is a defect, and swallowing it here turns a
                    # crash into a warning on stdout with no result files — the failure
                    # this benchmark has already paid for twice, discovered only after
                    # the GPU time was spent. Those propagate, with the traceback in the
                    # run's own log rather than in whatever captured stdout.
                    if not is_oom_error(exc):
                        log(
                            f"FATAL on LLM {llm_cfg.get('model_name', '')} / dataset "
                            f"{dataset_cfg.get('name', 'unknown')}: "
                            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                        )
                        raise
                    log(
                        f"OOM — skipping LLM {llm_cfg.get('model_name', '')} "
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
