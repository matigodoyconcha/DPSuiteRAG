import random
from datasets import load_dataset
from bench_datasets.base_dataset import BaseDataset
from metrics.utility import has_scoreable_reference

class HotpotQADataset(BaseDataset):
    """
    HotpotQA: A multi-hop question answering dataset requiring reasoning across multiple
    Wikipedia passages to answer questions. Used in the 'distractor' setting, where 10
    candidate paragraphs are provided (2 gold + 8 distractors).

    Source: Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question
    Answering", EMNLP 2018. HuggingFace: `hotpot_qa` (config: `distractor`).
    """

    # Task: multi-hop QA. Short factual answer expected.
    TASK_INSTRUCTION = (
        "Answer the multi-hop question using the context. Give a short factual "
        "answer only (a few words), with no explanation"
    )

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
        dataset = load_dataset("hotpot_qa", "distractor")
        train_rows = list(dataset[self.index_split])
        test_rows = list(dataset[self.test_split])

        rng = random.Random(self.random_seed)
        rng.shuffle(train_rows)
        rng.shuffle(test_rows)

        unique_contexts: dict[str, None] = {}
        for row in train_rows:
            context_list = row.get("context", {})
            sentences = context_list.get("sentences", [])
            for sents in sentences:
                ctx = " ".join(sents)
                if ctx.strip():
                    unique_contexts[ctx.strip()] = None
            if self.max_index_docs is not None and len(unique_contexts) >= self.max_index_docs:
                break

        self._documents = list(unique_contexts.keys())

        queries: list[dict] = []
        for row in test_rows:
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()

            # No question, or no gold left after answer normalization: nothing here
            # can be answered right or wrong, and an unscoreable reference turns an
            # empty prediction into a perfect one.
            if not question or not has_scoreable_reference(answer):
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
