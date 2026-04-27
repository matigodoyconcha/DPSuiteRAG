from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from bench_datasets.chatdoctor import ChatDoctorDataset
from bench_datasets.hotpotqa import HotpotQADataset
from bench_datasets.natural_questions import NaturalQuestionsDataset
from bench_datasets.squad import SQuADDataset
from bench_datasets.triviaqa import TriviaQADataset
from dp_methods.dpvoterag.generator import VoteRAGGenerator
from dp_methods.dpvoterag.pipeline import DPVoteRAGPipeline
from dp_methods.dpvoterag.sparse_vote_generator import SparseVoteRAGGenerator
from metrics.utility import match_accuracy
from vanilla.chunker import FixedSizeChunker
from vanilla.embedder import HFEmbedder
from vanilla.indexer import FAISSIndexer
from vanilla.postprocessor import IdentityPostprocessor
from vanilla.prompt_builder import SimplePromptBuilder
from vanilla.reranker import IdentityReranker
from vanilla.retriever import CosineSimilarityRetriever


def load_yaml(path: str) -> dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config invalida: {path}")
    return data


def load_dataset(name: str, seed: int, max_queries: int) -> Any:
    key = str(name).lower()
    common_kwargs = {
        "max_test_queries": int(max_queries),
        "random_seed": int(seed),
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
    else:
        raise ValueError(f"Dataset no soportado: {name}")

    dataset.load()
    return dataset


def normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def n_tokens(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer(str(text), add_special_tokens=False).get("input_ids", [])
    return int(len(token_ids))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compara DPVoteRAG vs DPSparseVoteRAG")
    parser.add_argument("--dp-config", type=str, default="experiments/configs/dpvoterag.yaml")
    parser.add_argument("--sparse-config", type=str, default="experiments/configs/dpsparsevoterag.yaml")
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=None, help="Si se pasa, fuerza epsilon_total para ambos.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    dp_cfg = load_yaml(args.dp_config)
    sparse_cfg = load_yaml(args.sparse_config)

    dataset_name = str(sparse_cfg.get("dataset", {}).get("name", "natural_questions"))
    llm_model_name = str(sparse_cfg.get("llm", {}).get("model_name", "facebook/opt-1.3b"))
    embedder_model_name = str(sparse_cfg.get("models", {}).get("embedder", "sentence-transformers/all-MiniLM-L6-v2"))

    sparse_method_cfg = sparse_cfg.get("dpsparsevoterag", {})
    dp_method_cfg = dp_cfg.get("dpvoterag", {})

    seed = int(sparse_method_cfg.get("random_seed", 42))
    m = int(sparse_method_cfg.get("m", 30))
    k = int(sparse_method_cfg.get("k", 1))
    epsilon_token = float(sparse_method_cfg.get("epsilon_token", 2.0))
    delta_token = float(sparse_method_cfg.get("delta_token", 1e-5))
    k_bar = sparse_method_cfg.get("k_bar", None)
    k_bar = int(k_bar) if k_bar is not None else None
    tau = sparse_method_cfg.get("tau", None)
    tau = None if tau is None else float(tau)
    tmax_absolute = int(sparse_method_cfg.get("tmax_absolute", 200))

    epsilon_total = args.epsilon
    if epsilon_total is None:
        sparse_eps_values = sparse_cfg.get("epsilon_values", [10.0])
        epsilon_total = float(sparse_eps_values[0])

    delta_total = float(sparse_cfg.get("delta_total", 1e-4))
    k_retrieved = int(sparse_cfg.get("k_retrieved", m * k))

    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    model = AutoModelForCausalLM.from_pretrained(llm_model_name)
    model.eval()

    dataset = load_dataset(dataset_name, seed=seed, max_queries=int(args.max_queries))
    documents = dataset.get_documents()
    queries = dataset.get_queries()[: int(args.max_queries)]

    chunker = FixedSizeChunker(chunk_size=200, overlap=20)
    embedder = HFEmbedder(model_name=embedder_model_name)
    indexer = FAISSIndexer()
    retriever = CosineSimilarityRetriever(indexer=indexer)
    reranker = IdentityReranker()
    prompt_builder = SimplePromptBuilder()
    postprocessor = IdentityPostprocessor()

    dp_generator = VoteRAGGenerator(
        model=model,
        tokenizer=tokenizer,
        m=int(dp_method_cfg.get("m", m)),
        k=int(dp_method_cfg.get("k", k)),
        epsilon_token=float(dp_method_cfg.get("epsilon_token", epsilon_token)),
        delta_token=float(dp_method_cfg.get("delta_token", delta_token)),
        epsilon_total=float(epsilon_total),
        delta_total=delta_total,
        k_bar=int(dp_method_cfg.get("k_bar", k_bar if k_bar is not None else m)),
        random_seed=seed,
    )

    sparse_generator = SparseVoteRAGGenerator(
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
        tau=tau,
        tmax_absolute=tmax_absolute,
    )

    dp_pipeline = DPVoteRAGPipeline(
        chunker=chunker,
        embedder=embedder,
        indexer=indexer,
        retriever=retriever,
        reranker=reranker,
        prompt_builder=prompt_builder,
        generator=dp_generator,
        postprocessor=postprocessor,
    )

    sparse_pipeline = DPVoteRAGPipeline(
        chunker=chunker,
        embedder=embedder,
        indexer=indexer,
        retriever=retriever,
        reranker=reranker,
        prompt_builder=prompt_builder,
        generator=sparse_generator,
        postprocessor=postprocessor,
    )

    print("Indexando documentos una sola vez para ambos metodos...")
    dp_pipeline.index(documents)

    rows: list[dict[str, Any]] = []
    dp_preds: list[str] = []
    sparse_preds: list[str] = []
    refs: list[str] = []

    dp_ntokens: list[int] = []
    sparse_ntokens: list[int] = []

    for idx, sample in enumerate(queries):
        query = str(sample.get("query", "")).strip()
        ref = str(sample.get("answer", "")).strip()
        if not query:
            continue

        dp_ans = dp_pipeline.query(query, k=k_retrieved)
        sparse_ans = sparse_pipeline.query(query, k=k_retrieved)

        dp_tok = n_tokens(tokenizer, dp_ans)
        sparse_tok = int(sparse_generator.last_run_stats.get("total_tokens", 0))

        dp_ok = int(normalize(dp_ans) == normalize(ref))
        sparse_ok = int(normalize(sparse_ans) == normalize(ref))

        rows.append(
            {
                "query_id": idx,
                "dpvoterag_answer": dp_ans,
                "dpsparse_answer": sparse_ans,
                "reference": ref,
                "dpvoterag_correct": dp_ok,
                "dpsparse_correct": sparse_ok,
                "dpvoterag_ntokens": dp_tok,
                "dpsparse_ntokens": sparse_tok,
            }
        )

        dp_preds.append(dp_ans)
        sparse_preds.append(sparse_ans)
        refs.append(ref)
        dp_ntokens.append(dp_tok)
        sparse_ntokens.append(sparse_tok)

    print(
        "query_id | dpvoterag_answer | dpsparse_answer | reference | "
        "dpvoterag_correct | dpsparse_correct | dpvoterag_ntokens | dpsparse_ntokens"
    )
    for row in rows:
        print(
            f"{row['query_id']} | {row['dpvoterag_answer']} | {row['dpsparse_answer']} | "
            f"{row['reference']} | {row['dpvoterag_correct']} | {row['dpsparse_correct']} | "
            f"{row['dpvoterag_ntokens']} | {row['dpsparse_ntokens']}"
        )

    dp_acc = match_accuracy(dp_preds, refs)
    sparse_acc = match_accuracy(sparse_preds, refs)
    dp_avg_tok = float(np.mean(dp_ntokens)) if dp_ntokens else 0.0
    sparse_avg_tok = float(np.mean(sparse_ntokens)) if sparse_ntokens else 0.0

    print("\nResumen:")
    print(f"- DPVoteRAG match_accuracy: {dp_acc:.4f}")
    print(f"- DPSparseVoteRAG match_accuracy: {sparse_acc:.4f}")
    print(f"- DPVoteRAG avg tokens: {dp_avg_tok:.2f}")
    print(f"- DPSparseVoteRAG avg tokens: {sparse_avg_tok:.2f}")


if __name__ == "__main__":
    main()

# Comparative table from pilot run (update after execution):
# - Pending execution in this workspace.
