"""Benchmark script for the no-retrieval baseline (zero-shot).

The same generator answering the same questions with no document in the prompt. It is
the floor the whole benchmark is read against: `vanilla` says what retrieval is worth
when nothing is protected, and this says what the corpus is worth at all. A DP method
scoring here is answering from the model's parametric knowledge, whatever its budget
bought.

Nothing in this runner retrieves: no embedder, no index, no reranker, no context. The
prompt is the dataset's own task instruction plus the question, which is the same
no-context prompt the DP methods fall back to when their mechanism releases no
document (see ``BaseDataset.get_task_instruction``).

Usage:
    python experiments/run_no_rag.py --config experiments/configs/full/no_rag.yaml --dataset squad
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from bench_llms import load_llm
from core.device import free_memory, is_oom_error
from experiments.common import (
    iter_dataset_configs,
    iter_llm_configs,
    llm_label,
    load_dataset_from_config,
    only_dataset,
    resolve_seeds,
    select_datasets,
    select_queries,
)
from experiments.llm_isolation import resolve_llm_execution
from experiments.mia_probe import mia_only_enabled, resolve_mia_params, run_mia_probes
from experiments.run_logging import setup_method_logging


def build_prompt(instruction: str, query: str) -> str:
    """The whole prompt: the dataset's task instruction, then the question.

    Nothing sits between them, which is the condition this runner measures. The shape is
    the no-context prompt the DP methods themselves fall back to when their mechanism
    releases no document, so the floor is not measured on a prompt of its own.
    """
    return f"{instruction}\nQuestion: {query}\nAnswer:"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_config() -> dict[str, Any]:
    return {
        "method": "no_rag",
        "dataset": {
            "name": "squad",
            "index_split": "train",
            "test_split": "validation",
            "max_index_docs": 20000,
            "max_test_queries": 40,
        },
        "llm": {
            "name": "huggingface",
            "model_name": "sshleifer/tiny-gpt2",
        },
        "generation": {
            # The protocol's answer length, the same number the YAML configs set and the
            # other eight methods are held to. Every text metric is monotone in it, so a
            # fallback that disagreed with the configs would make the floor the one row
            # measured over longer answers than the curves read against it.
            "max_new_tokens": 30,
        },
        "random_seed": 42,
        "output_path": "results/no_rag/no_rag.json",
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
        return config

    cfg_path = Path(path)
    if not cfg_path.exists():
        return config

    with cfg_path.open("r", encoding="utf-8") as f:
        file_config = yaml.safe_load(f) or {}
    if not isinstance(file_config, dict):
        raise ValueError("The config YAML file must be an object/dict.")
    return deep_update(config, file_config)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset is not None:
        config["dataset"]["name"] = args.dataset
        # The datasets: list would otherwise win the merge and the run would sweep
        # every entry, quietly ignoring the dataset that was asked for.
        config.pop("datasets", None)
    if args.llm_name is not None:
        config["llm"]["name"] = args.llm_name
    if args.llm_model is not None:
        config["llm"]["model_name"] = args.llm_model
    if args.max_new_tokens is not None:
        config["generation"]["max_new_tokens"] = args.max_new_tokens
    if args.seed is not None:
        config["random_seed"] = args.seed
    if args.output is not None:
        config["output_path"] = args.output

    # Dataset-scoped overrides land on the singular block AND on every entry of the
    # datasets: list, because an entry's own key wins the merge — writing only to the
    # singular block would leave the flag with no effect whenever a list is present.
    dataset_overrides = {
        "index_split": args.index_split,
        "test_split": args.test_split,
        "max_index_docs": args.max_index_docs,
        "max_test_queries": args.max_test_queries,
    }
    targets = [config["dataset"], *(config.get("datasets") or [])]
    for key, value in dataset_overrides.items():
        if value is None:
            continue
        for target in targets:
            if isinstance(target, dict):
                target[key] = value

    return config


def run_experiment(
    config: dict[str, Any],
    dataset_cfg: dict[str, Any],
    llm_cfg: dict[str, Any],
) -> dict[str, Any]:
    # data_seed picks the corpus and the questions; run_seed drives generation, so a
    # repetition varies the run without resampling the question set.
    data_seed, seed = resolve_seeds(config)
    set_seed(seed)

    dataset_name = str(dataset_cfg.get("name", ""))
    llm = load_llm(llm_cfg)
    llm_config = llm.get_config()
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    gen_cfg = config.get("generation", {})
    use_chat_template = gen_cfg.get("use_chat_template", "auto")
    stop_strings = gen_cfg.get("stop_strings")

    # The same loader call, with the same filters, that every DP method makes: the
    # baseline is only a baseline if the questions are the ones the curves were measured
    # on. The corpus it builds along the way is never read.
    dataset = load_dataset_from_config(
        dataset_cfg, data_seed, default_max_queries=dataset_cfg.get("max_test_queries")
    )
    instruction = dataset.get_task_instruction()
    generator = llm.build_generator(
        max_new_tokens=max_new_tokens,
        use_chat_template=use_chat_template,
        stop_strings=stop_strings,
    )

    print(
        "Configuracion activa:\n"
        f"- dataset: {dataset_name}\n"
        f"- llm: {llm_config['name']} ({llm_config['model_name']})\n"
        f"- retrieval: none (zero-shot)\n"
        f"- max_new_tokens: {max_new_tokens}"
    )

    base_payload = {
        "method": "no_rag",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "dataset_config": dataset_cfg,
        "llm": llm_config,
        "generation": config["generation"],
        # The answer length this row was measured at, under the name every other method
        # writes it under, so the floor joins the comparison on the same key.
        "answer_tokens": max_new_tokens,
        # No mechanism to price, so no workload block and no epsilon of any kind: this
        # condition reads no private data, and a number on the privacy axis would claim
        # it sits somewhere on a scale it is off entirely.
        "random_seed": seed,
        "data_seed": data_seed,
        # Named as every other runner names it, so one reader serves all of them.
        "run_seed": seed,
        "seed": seed,
    }

    # Standalone MIA run: the attack's floor. No document reaches the prompt, so any
    # separation between members and non-members here is the generator recognising text
    # it was pretrained on — the AUC an attack scores without reading the corpus at all,
    # and the level a DP method's own AUC has to be read against.
    if mia_only_enabled():
        mia_params = resolve_mia_params(config)
        mia_split = dataset.get_mia_split(
            n_per_class=mia_params["n_per_class"],
            index_size=mia_params["index_size"],
            seed=mia_params["seed"],
        )
        if mia_split is None:
            raise ValueError(f"Dataset '{dataset_name}' has no MIA member/non-member split.")
        mia_records = run_mia_probes(
            lambda q: generator.generate(build_prompt(instruction, q)),
            mia_split,
            anderson_doc_maxlen=mia_params["anderson_doc_maxlen"],
        )
        return {**base_payload, "mia_probes": mia_records}

    queries = select_queries(
        dataset.get_queries(), max_queries=dataset_cfg.get("max_test_queries")
    )
    if not queries:
        raise ValueError("No queries to evaluate.")

    results = []
    for row in queries:
        query = str(row.get("query", "")).strip()
        reference = str(row.get("answer", "")).strip()
        answers = [
            str(a).strip() for a in (row.get("answers") or [reference]) if str(a).strip()
        ] or [reference]

        answer = generator.generate(build_prompt(instruction, query))

        # ``prediction`` is the key the scorer reads; every method writes it under that
        # name so a single scoring path covers the baselines and the DP methods.
        result = {
            "query": query,
            "query_id": row.get("query_id", ""),
            "prediction": answer,
            "reference": reference,
            "answers": answers,
        }
        # Set-valued gold (Movielens held-out favorites) drives hit_rate.
        if row.get("favorites"):
            result["favorites"] = [str(f) for f in row["favorites"]]
        results.append(result)

        print("=" * 30)
        print(f"Question: {query}")
        print(f"Answer: {answer}")
        if reference:
            print(f"Reference: {reference}")
        print()

    return {
        **base_payload,
        "num_test_queries": len(results),
        "n_queries": len(results),
        "predictions": [r["prediction"] for r in results],
        "references": [r["reference"] for r in results],
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Runner del baseline sin recuperacion (zero-shot)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/full/no_rag.yaml",
        help="Path to configuration YAML",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--llm-name", type=str, default=None, help="Override LLM type")
    parser.add_argument("--llm-model", type=str, default=None, help="Override LLM model")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max new tokens")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output", type=str, default=None, help="Override JSON output path")

    parser.add_argument("--index-split", type=str, default=None, help="Split the corpus is drawn from")
    parser.add_argument("--test-split", type=str, default=None, help="Split used for evaluation")
    # The corpus caps select nothing this runner reads, but several loaders partition
    # one dataset between documents and questions, so they still decide which questions
    # get asked — and the questions have to be the DP methods' own.
    parser.add_argument("--max-index-docs", type=int, default=None, help="Maximum number of corpus docs")
    parser.add_argument("--max-test-queries", type=int, default=None, help="Maximum number of test queries")
    parser.add_argument(
        "--mia-only",
        action="store_true",
        help="Run the standalone membership inference attack (S2MIA + Anderson Yes/No) "
        "instead of the utility loop: probe members vs held-out non-members with no "
        "document in the prompt, which is the AUC an attack reaches without the corpus.",
    )
    return parser


def main() -> None:
    setup_method_logging("no_rag")
    args = build_parser().parse_args()
    if args.mia_only:
        os.environ["BENCHDP_MIA_ONLY"] = "1"

    config = load_yaml_config(args.config)
    config = apply_cli_overrides(config, args)

    all_dataset_cfgs = iter_dataset_configs(config)
    names = [str(d.get("name")) for d in all_dataset_cfgs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        # Two entries with the same name share an output directory, so the second would
        # overwrite the first's results — silently, and only when their filters differ.
        raise ValueError(
            f"duplicate dataset names in the config: {sorted(duplicates)}. Give each "
            "entry a distinct name; entries differing only in filters still collide "
            "on the output path."
        )

    dataset_cfgs = select_datasets(all_dataset_cfgs)
    if not dataset_cfgs:
        requested = only_dataset()
        raise ValueError(
            f"BENCHDP_ONLY_DATASET={requested!r} matches no dataset in this config "
            f"(it lists {names}). Returning without running would report the cell as "
            "OK while producing no results."
        )

    llm_cfgs = iter_llm_configs(config)
    multi_llm = len(llm_cfgs) > 1
    # One subprocess per LLM, so GPU memory is fully reclaimed between models and the
    # baseline is isolated the same way every DP method is.
    todo = resolve_llm_execution(llm_cfgs, method_name="no_rag")
    if todo is None:
        return

    fname = "mia.json" if mia_only_enabled() else "results.json"
    # The orchestrator (experiments/runner.py) redirects output under the run
    # directory; a standalone invocation keeps the config's own output_path.
    out_dir = os.environ.get("BENCHDP_OUTPUT_DIR") or config.get("output_dir")

    for ds_cfg in dataset_cfgs:
        dataset_name = str(ds_cfg.get("name", ""))
        for _idx, llm_cfg in todo:
            if multi_llm:
                print(f"\n##### LLM: {llm_cfg.get('model_name')} #####")
            if out_dir:
                parts = [Path(str(out_dir)), dataset_name]
            else:
                parts = [Path(str(config["output_path"])).parent, dataset_name]
            if multi_llm:
                parts.append(llm_label(llm_cfg))
            output_path = Path(*parts) / fname
            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                payload = run_experiment(config, ds_cfg, llm_cfg)
            except OSError as exc:
                # A model we cannot load (a gated repo, a missing checkpoint) must not
                # abort the remaining cells.
                print(f"\n[WARN] skipping LLM {llm_cfg.get('model_name')}: {exc}")
                free_memory()
                continue
            except (RuntimeError, MemoryError) as exc:
                if not is_oom_error(exc):
                    raise
                print(f"\n[WARN] OOM — skipping LLM {llm_cfg.get('model_name')}: {exc}")
                free_memory()
                continue

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"Resultados guardados en: {output_path}")
            # Outside the except on purpose: a live exception pins the frame that holds
            # the model, so collecting inside the handler frees nothing. Harmless on the
            # success path.
            free_memory()


if __name__ == "__main__":
    main()
