import random
from datasets import load_dataset
from bench_datasets.base_dataset import BaseDataset

class TriviaQADataset(BaseDataset):
    """
    TriviaQA: A reading comprehension dataset with trivia questions and evidence documents
    from Wikipedia and the web. Each question has multiple verified answers and associated
    passages. Used here in the 'rc' (reading comprehension) configuration.

    Source: Joshi et al., "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset
    for Reading Comprehension", ACL 2017. HuggingFace: `trivia_qa` (config: `rc`).
    """
    def __init__(
        self,
        index_split: str = "train",
        test_split: str = "validation",
        max_index_docs: int | None = 200,
        max_test_queries: int | None = 50,
        random_seed: int = 42,
    ):
        self.index_split = index_split
        self.test_split = test_split
        self.max_index_docs = max_index_docs
        self.max_test_queries = max_test_queries
        self.random_seed = random_seed
        self._documents: list[str] = []
        self._queries: list[dict] = []

    def load(self) -> None:
        # Use the namespaced repo id: newer huggingface_hub rejects the bare
        # canonical id "trivia_qa" (requires 'namespace/name'). Same dataset.
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc", streaming=True)

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

    def get_private_entities(self) -> list[str]:
        return []
