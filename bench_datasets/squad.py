import random

from datasets import load_dataset

from bench_datasets.base_dataset import BaseDataset


class SQuADDataset(BaseDataset):
    """
    SQuAD (Stanford Question Answering Dataset): Reading comprehension dataset where
    questions are posed on Wikipedia articles and answers are spans extracted directly
    from the corresponding passage. Uses SQuAD v1.1 (all questions are answerable).

    Source: Rajpurkar et al., "SQuAD: 100,000+ Questions for Machine Comprehension
    of Text", EMNLP 2016. HuggingFace: `squad`.
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
        dataset = load_dataset("rajpurkar/squad")  # namespaced id (bare "squad" no longer loads)
        if self.index_split not in dataset:
            raise ValueError(f"Split '{self.index_split}' no existe en SQuAD.")
        if self.test_split not in dataset:
            raise ValueError(f"Split '{self.test_split}' no existe en SQuAD.")

        train_rows = list(dataset[self.index_split])
        test_rows = list(dataset[self.test_split])

        rng = random.Random(self.random_seed)
        rng.shuffle(train_rows)
        rng.shuffle(test_rows)

        # Build test queries FIRST so their gold contexts can be guaranteed into
        # the index below.
        queries: list[dict] = []
        for row in test_rows:
            question = (row.get("question") or "").strip()
            answers = row.get("answers") or {}
            answer_list = answers.get("text") if isinstance(answers, dict) else []
            answer = answer_list[0].strip() if answer_list else ""

            if not question:
                continue

            queries.append(
                {
                    "query": question,
                    "answer": answer,
                    "id": row.get("id"),
                    "context": (row.get("context") or "").strip(),
                }
            )
            if self.max_test_queries is not None and len(queries) >= self.max_test_queries:
                break

        self._queries = queries

        # Build the index from unique contexts. SQuAD's train and validation use
        # DISJOINT Wikipedia articles, so indexing only `index_split` would never
        # contain the answer passage for a `test_split` query — retrieval would
        # return irrelevant contexts and no method could answer. To keep retrieval
        # meaningful at small corpus sizes, seed the index with each test query's
        # own gold context, then top up with other `index_split` contexts up to
        # `max_index_docs`. (SQuAD is only used for smoke tests in this repo, so
        # this guaranteed-recall corpus is intentional, not a leakage concern for
        # the paper-faithful full configs, which use other datasets.)
        unique_contexts: dict[str, None] = {}
        for q in queries:
            ctx = q["context"]
            if ctx:
                unique_contexts[ctx] = None
        for row in train_rows:
            if self.max_index_docs is not None and len(unique_contexts) >= self.max_index_docs:
                break
            context = (row.get("context") or "").strip()
            if context:
                unique_contexts[context] = None

        self._documents = list(unique_contexts.keys())

    def get_documents(self) -> list[str]:
        if not self._documents:
            raise RuntimeError("Dataset not loaded. Call load() before get_documents().")
        return self._documents

    def get_queries(self) -> list[dict]:
        if not self._queries:
            raise RuntimeError("Dataset not loaded. Call load() before get_queries().")
        return self._queries

    def get_private_entities(self) -> list[str]:
        return []
