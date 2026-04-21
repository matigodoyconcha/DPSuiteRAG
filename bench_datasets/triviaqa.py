import random
from datasets import load_dataset
from bench_datasets.base_dataset import BaseDataset

class TriviaQADataset(BaseDataset):
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
        dataset = load_dataset("trivia_qa", "rc")
        train_rows = list(dataset[self.index_split])
        test_rows = list(dataset[self.test_split])

        rng = random.Random(self.random_seed)
        rng.shuffle(train_rows)
        rng.shuffle(test_rows)

        unique_contexts: dict[str, None] = {}
        for row in train_rows:
            search_results = row.get("search_results", {})
            if "search_context" in search_results:
                for ctx in search_results["search_context"]:
                    if ctx.strip():
                        unique_contexts[ctx.strip()] = None
            if self.max_index_docs is not None and len(unique_contexts) >= self.max_index_docs:
                break

        self._documents = list(unique_contexts.keys())

        queries: list[dict] = []
        for row in test_rows:
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
            raise RuntimeError("Dataset no cargado. Ejecuta load().")
        return self._documents

    def get_queries(self) -> list[dict]:
        if not self._queries:
            raise RuntimeError("Dataset no cargado. Ejecuta load().")
        return self._queries

    def get_private_entities(self) -> list[str]:
        return []
