import random
import warnings

from datasets import load_dataset

from bench_datasets.base_dataset import BaseDataset
from bench_datasets.dpr_corpus import (
    DEFAULT_DPR_CORPUS_PATH,
    corpus_available,
    load_dpr_passages,
)

class TriviaQADataset(BaseDataset):
    """
    TriviaQA: A reading comprehension dataset with trivia questions and evidence documents
    from Wikipedia and the web. Each question has multiple verified answers and associated
    passages. Used here in the 'rc' (reading comprehension) configuration.

    Source: Joshi et al., "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset
    for Reading Comprehension", ACL 2017. HuggingFace: `trivia_qa` (config: `rc`).
    """

    # Task: trivia QA. Short factual answer expected.
    TASK_INSTRUCTION = (
        "Answer the trivia question using the context. Give the answer only, in "
        "a few words, with no explanation"
    )

    def __init__(
        self,
        index_split: str = "train",
        test_split: str = "validation",
        max_index_docs: int | None = 200,
        max_test_queries: int | None = 50,
        random_seed: int = 42,
        dpr_corpus_path=DEFAULT_DPR_CORPUS_PATH,
    ):
        self.index_split = index_split
        self.test_split = test_split
        self.max_index_docs = max_index_docs
        self.max_test_queries = max_test_queries
        self.random_seed = random_seed
        self.dpr_corpus_path = dpr_corpus_path
        self._documents: list[str] = []
        self._queries: list[dict] = []

    def load(self) -> None:
        # Use the namespaced repo id: newer huggingface_hub rejects the bare
        # canonical id "trivia_qa" (requires 'namespace/name'). Same dataset.
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc", streaming=True)

        # Paper-faithful retrieval corpus: DPR psgs_w100 Wikipedia passages (the
        # same corpus TriviaQA retrieves from in the DP-KSA / Private-RAG papers).
        # Fall back to the bundled entity_pages wiki_context when the corpus is
        # not downloaded so that offline runs and smoke tests still work.
        if corpus_available(self.dpr_corpus_path):
            self._documents = load_dpr_passages(
                self.dpr_corpus_path, self.max_index_docs, self.random_seed
            )
        else:
            warnings.warn(
                f"DPR corpus not found at {self.dpr_corpus_path}; falling back to "
                "bundled entity_pages wiki_context for TriviaQA (capped, partial). "
                "Run `python scripts/download_dpr_corpus.py` for the paper-faithful corpus.",
                stacklevel=2,
            )
            unique_contexts: dict[str, None] = {}
            for row in dataset[self.index_split]:
                entity_pages = row.get("entity_pages") or {}
                wiki_contexts = entity_pages.get("wiki_context") or []

                if isinstance(wiki_contexts, str):
                    wiki_contexts = [wiki_contexts]

                for ctx in wiki_contexts:
                    ctx_text = str(ctx).strip()
                    if ctx_text:
                        unique_contexts[ctx_text] = None

                if self.max_index_docs is not None and len(unique_contexts) >= self.max_index_docs:
                    break

            self._documents = list(unique_contexts.keys())

        queries: list[dict] = []
        for row in dataset[self.test_split]:
            question = (row.get("question") or "").strip()
            answer = row.get("answer", {}).get("value", "").strip()

            if not question:
                continue

            queries.append({
                "query": question,
                "answer": answer,
                "context": ""
            })
            if self.max_test_queries is not None and len(queries) >= self.max_test_queries:
                break

        self._queries = queries

    def get_documents(self) -> list[str]:
        if not self._documents:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self._documents

    def get_queries(self) -> list[dict]:
        if not self._queries:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self._queries
