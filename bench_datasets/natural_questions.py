import random
import warnings

from datasets import load_dataset

from bench_datasets.base_dataset import BaseDataset
from bench_datasets.dpr_corpus import (
    DEFAULT_DPR_CORPUS_PATH,
    corpus_available,
    load_dpr_passages,
)

class NaturalQuestionsDataset(BaseDataset):
    """
    Natural Questions (Open): Real questions issued to the Google search engine, with
    answers extracted from Wikipedia articles. Uses the open-domain variant (`nq_open`),
    which omits passage-level annotations. Since no retrievable passages are provided,
    the index is built from Q+A pairs derived from the training split.

    Source: Kwiatkowski et al., "Natural Questions: A Benchmark for Question Answering
    Research", TACL 2019. HuggingFace: `nq_open`.
    """

    # Task: open-domain QA. Short factual answer expected.
    TASK_INSTRUCTION = (
        "Answer the question using the context. Give a short factual answer only "
        "(a few words), with no explanation"
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
        # Utilizing nq_open which is simpler than the full natural_questions
        dataset = load_dataset("google-research-datasets/nq_open")
        if self.index_split not in dataset:
            self.index_split = "train"
        if self.test_split not in dataset:
            self.test_split = "validation"

        train_rows = list(dataset[self.index_split])
        test_rows = list(dataset[self.test_split])

        rng = random.Random(self.random_seed)
        rng.shuffle(train_rows)
        rng.shuffle(test_rows)

        # Paper-faithful retrieval corpus: DPR psgs_w100 Wikipedia passages (the
        # same corpus NQ retrieves from in the DP-KSA / Private-RAG papers). Fall
        # back to Q+A pseudo-documents when the corpus is not downloaded so that
        # offline runs and smoke tests still work (nq_open ships no passages).
        if corpus_available(self.dpr_corpus_path):
            self._documents = load_dpr_passages(
                self.dpr_corpus_path, self.max_index_docs, self.random_seed
            )
        else:
            warnings.warn(
                f"DPR corpus not found at {self.dpr_corpus_path}; falling back to "
                "Q+A pseudo-documents for Natural Questions (degenerate retrieval). "
                "Run `python scripts/download_dpr_corpus.py` for the paper-faithful corpus.",
                stacklevel=2,
            )
            unique_contexts: dict[str, None] = {}
            for row in train_rows:
                question = (row.get("question") or "").strip()
                answers = row.get("answer") or []
                ans = answers[0] if answers else ""
                if question and ans:
                    qa_doc = f"Q: {question}\nA: {ans}"
                    unique_contexts[qa_doc] = None
                if self.max_index_docs is not None and len(unique_contexts) >= self.max_index_docs:
                    break
            self._documents = list(unique_contexts.keys())

        queries: list[dict] = []
        for row in test_rows:
            question = (row.get("question") or "").strip()
            # NQ-open gives several acceptable short answers; keep them all so the
            # metrics score max-over-aliases instead of only the first.
            all_ans = [str(a).strip() for a in (row.get("answer") or []) if str(a).strip()]
            ans = all_ans[0] if all_ans else ""

            if not question:
                continue

            queries.append({
                "query": question,
                "answer": ans,
                "answers": all_ans or [ans],
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
