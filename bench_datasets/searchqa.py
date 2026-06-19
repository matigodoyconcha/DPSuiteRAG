"""SearchQA dataset loader.

Source: Dunn et al., "SearchQA: A New Q&A Dataset Augmented with Context from a
Search Engine", arXiv:1704.05179 (2017).
HuggingFace: ``lucadiliello/searchqa`` (snippets used as retrieval documents).

Used by the DP-SynRAG paper (arXiv:2510.06719v2) as one of three benchmark
datasets. The 7054 search snippets serve as the retrieval corpus and the
Jeopardy! questions as evaluation queries.
"""

from __future__ import annotations

import random
import re
from typing import Iterable

from datasets import load_dataset

from bench_datasets.base_dataset import BaseDataset


class SearchQADataset(BaseDataset):
    """SearchQA: Jeopardy! questions + Google search snippets."""

    # Task: Jeopardy!-style QA. Short factual answer expected.
    TASK_INSTRUCTION = (
        "Answer the question using the context. Give the answer only, in a few "
        "words, with no explanation"
    )

    HF_NAME = "lucadiliello/searchqa"

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
        dataset = load_dataset(self.HF_NAME)
        if self.index_split not in dataset:
            self.index_split = next(iter(dataset.keys()))
        if self.test_split not in dataset:
            self.test_split = self.index_split

        rng = random.Random(self.random_seed)

        index_rows = list(dataset[self.index_split])
        rng.shuffle(index_rows)

        unique_docs: dict[str, None] = {}
        for row in index_rows:
            for snippet in self._extract_snippets(row):
                clean = self._clean_snippet(snippet)
                if clean:
                    unique_docs[clean] = None
            if (
                self.max_index_docs is not None
                and len(unique_docs) >= self.max_index_docs
            ):
                break
        self._documents = list(unique_docs.keys())

        test_rows = list(dataset[self.test_split])
        rng.shuffle(test_rows)
        queries: list[dict] = []
        for row in test_rows:
            question = (row.get("question") or row.get("query") or "").strip()
            answer = self._extract_answer(row)
            if not question or not answer:
                continue
            queries.append({"query": question, "answer": answer, "context": ""})
            if (
                self.max_test_queries is not None
                and len(queries) >= self.max_test_queries
            ):
                break
        self._queries = queries

    # SearchQA snippets ship with structural delimiters ([DOC], [TLE], [PAR], ...)
    # that the question never references. Left in, they dominate the keyword
    # histogram and DP-SynRAG ends up synthesizing boilerplate like "[DOC] [TLE] ..."
    # instead of real content, so we strip every bracketed all-caps marker.
    _MARKER_RE = re.compile(r"\[[A-Z]+\]")

    @classmethod
    def _clean_snippet(cls, snippet: str) -> str:
        text = cls._MARKER_RE.sub(" ", snippet)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _extract_snippets(row: dict) -> Iterable[str]:
        for key in ("search_results", "snippets", "context", "contexts", "search_snippets"):
            val = row.get(key)
            if val is None:
                continue
            if isinstance(val, str):
                yield val
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        yield item
                    elif isinstance(item, dict):
                        for k in ("snippet", "text", "context"):
                            if isinstance(item.get(k), str):
                                yield item[k]
                                break
            elif isinstance(val, dict):
                snippets = val.get("snippet") or val.get("snippets")
                if isinstance(snippets, list):
                    for s in snippets:
                        if isinstance(s, str):
                            yield s

    @staticmethod
    def _extract_answer(row: dict) -> str:
        for key in ("answer", "answers", "target", "label"):
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, list) and val:
                first = val[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()
                if isinstance(first, dict):
                    text = first.get("text") or first.get("value")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
            if isinstance(val, dict):
                text = val.get("text") or val.get("value")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return ""

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
