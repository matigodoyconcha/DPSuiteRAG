"""SearchQA dataset loader.

Source: Dunn et al., "SearchQA: A New Q&A Dataset Augmented with Context from a
Search Engine", arXiv:1704.05179 (2017). This is the original full-support
SearchQA release — the same data as the GitHub ``nyu-dl/dl4ir-searchQA`` repo —
distributed on HuggingFace as ``kyunghyuncho/search_qa`` (config
``train_test_val``). Each Jeopardy! question ships with the full set of Google
search snippets collected for it (tens per question, ~60 on average), so a gold
answer is present in that question's own snippets by construction.

Used by the DP-SynRAG paper (arXiv:2510.06719v2) as one of three benchmark
datasets: the search snippets are the retrieval corpus and the Jeopardy!
questions are the evaluation queries.

Construction (same-split, question-anchored):
  The retrieval corpus is the union of the *evaluated questions' own* snippets.
  By default ``max_test_queries`` questions are sampled uniformly at random from
  the split, and the corpus is built from exactly those questions' snippets.
  Because corpus and queries come from the same questions, every gold answer that
  appears in its question's snippets is retrievable (~100% gold-in-corpus
  coverage), while retrieval stays non-trivial: the reader must find the right
  question's snippets among the several-thousand pooled documents.

Support-binning filter (opt-in, default off):
  When ``min_answer_snippets`` is set, the evaluation is restricted to questions
  whose own snippets contain the answer at least that many times (the paper uses
  40). Eligible questions are stratified into ``n_support_bins`` bins of fixed
  width 10 starting at ``min_answer_snippets`` (the paper uses 6 bins), and
  ``per_bin`` questions (the paper uses 17) are drawn from each with the seeded
  RNG, yielding up to ``n_support_bins * per_bin`` questions (102 in the paper).
  ``max_scan_questions`` optionally caps how many question files are read while
  filtering, to bound the cost of scanning the zip. With these parameters left at
  ``None`` the loader samples questions uniformly at random and applies no
  support threshold.
"""

from __future__ import annotations

import html
import json
import random
import re
import zipfile

from huggingface_hub import hf_hub_download

from bench_datasets.base_dataset import BaseDataset
from metrics.utility import has_scoreable_reference


class SearchQADataset(BaseDataset):
    """SearchQA: Jeopardy! questions + Google search snippets."""

    # Task: Jeopardy!-style QA. Short factual answer expected.
    TASK_INSTRUCTION = (
        "Answer the question using the context. Give the answer only, in a few "
        "words, with no explanation"
    )

    # The original full-support SearchQA on the Hub. Its config `train_test_val`
    # ships the raw per-question JSON in three zipped splits, one JSON file per
    # question. The repo carries a legacy loading script that the current
    # `datasets` library refuses to run, so the split zips are fetched directly.
    HF_NAME = "kyunghyuncho/search_qa"
    _SPLIT_ZIP = {
        "train": "data/train_test_val/train.zip",
        "validation": "data/train_test_val/val.zip",
        "val": "data/train_test_val/val.zip",
        "test": "data/train_test_val/test.zip",
    }

    def __init__(
        self,
        index_split: str = "train",
        test_split: str = "train",
        max_index_docs: int | None = 200,
        max_test_queries: int | None = 50,
        random_seed: int = 42,
        min_answer_snippets: int | None = None,
        n_support_bins: int | None = None,
        per_bin: int | None = None,
        max_scan_questions: int | None = None,
    ):
        # The corpus and the queries are built from the SAME questions, sampled
        # from ``index_split`` (the paper evaluates SearchQA on training
        # questions). ``test_split`` is accepted for interface parity with the
        # other loaders but is not used: queries never come from a split disjoint
        # from the corpus here.
        self.index_split = index_split
        self.test_split = test_split
        self.max_index_docs = max_index_docs
        self.max_test_queries = max_test_queries
        self.random_seed = random_seed
        # Support-binning filter (opt-in). None on min_answer_snippets selects the
        # uniform-sampling path; a value turns on the filter, with n_support_bins,
        # per_bin and max_scan_questions falling back to the paper's defaults.
        self.min_answer_snippets = int(min_answer_snippets) if min_answer_snippets is not None else None
        self.n_support_bins = int(n_support_bins) if n_support_bins is not None else None
        self.per_bin = int(per_bin) if per_bin is not None else None
        self.max_scan_questions = int(max_scan_questions) if max_scan_questions is not None else None
        self._documents: list[str] = []
        self._queries: list[dict] = []

    def load(self) -> None:
        split_zip = self._SPLIT_ZIP.get(self.index_split, self._SPLIT_ZIP["train"])
        zip_path = hf_hub_download(self.HF_NAME, split_zip, repo_type="dataset")

        rng = random.Random(self.random_seed)
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".json")]
            rng.shuffle(names)

            if self.min_answer_snippets is not None:
                selected = self._select_by_support(zf, names, rng)
            else:
                selected = self._select_uniform(zf, names)

        self._queries = [
            {"query": q, "answer": a, "answers": [a], "context": ""}
            for q, a, _ in selected
        ]
        self._documents = self._build_corpus(selected)

    def _parse_question(self, row: dict) -> tuple[str, str, list[str]] | None:
        """Parse a question row into ``(question, answer, snippets)``.

        Returns ``None`` when the question, the answer or the cleaned snippet set is
        empty — the only eligibility requirements shared by both selection paths.
        """
        question = str(row.get("question") or "").strip()
        answer = self._extract_answer(row)
        # An answer that normalization empties (an article, punctuation alone) is as
        # unusable as a missing one: the question could not be answered right or
        # wrong, and an empty prediction would match it exactly.
        if not question or not has_scoreable_reference(answer):
            return None
        snippets = [
            clean
            for clean in (
                self._clean_snippet(s) for s in self._extract_snippets(row)
            )
            if clean
        ]
        if not snippets:
            return None
        return question, answer, snippets

    def _select_uniform(
        self, zf: zipfile.ZipFile, names: list[str]
    ) -> list[tuple[str, str, list[str]]]:
        """Sample questions uniformly at random, up to ``max_test_queries``.

        A question is eligible when it has a non-empty question, a non-empty answer
        and at least one snippet — no support-count threshold.
        """
        n_target = self.max_test_queries
        selected: list[tuple[str, str, list[str]]] = []
        for name in names:
            if n_target is not None and len(selected) >= n_target:
                break
            with zf.open(name) as fh:
                row = json.load(fh)
            parsed = self._parse_question(row)
            if parsed is None:
                continue
            selected.append(parsed)
        return selected

    def _select_by_support(
        self, zf: zipfile.ZipFile, names: list[str], rng: random.Random
    ) -> list[tuple[str, str, list[str]]]:
        """Restrict to well-supported questions, stratified into support bins.

        Questions with at least ``min_answer_snippets`` answer-bearing snippets are
        binned by support count (fixed width 10, starting at ``min_answer_snippets``)
        and ``per_bin`` are sampled from each bin with the seeded RNG.
        ``max_test_queries``, when set, acts only as an upper cap on the result.
        """
        n_bins = self.n_support_bins if self.n_support_bins is not None else 6
        per_bin = self.per_bin if self.per_bin is not None else 17
        min_support = self.min_answer_snippets

        bins: list[list[tuple[str, str, list[str]]]] = [[] for _ in range(n_bins)]
        n_scanned = 0
        for name in names:
            if self.max_scan_questions is not None and n_scanned >= self.max_scan_questions:
                break
            n_scanned += 1
            with zf.open(name) as fh:
                row = json.load(fh)
            parsed = self._parse_question(row)
            if parsed is None:
                continue
            _, answer, snippets = parsed
            count = self._count_answer_snippets(answer, snippets)
            if count < min_support:
                continue
            bin_idx = min(n_bins - 1, (count - min_support) // 10)
            bins[bin_idx].append(parsed)

        selected: list[tuple[str, str, list[str]]] = []
        per_bin_counts: list[int] = []
        for b_idx, bucket in enumerate(bins):
            if len(bucket) <= per_bin:
                take = list(bucket)
                if len(bucket) < per_bin:
                    print(
                        f"[SearchQA] support bin {b_idx} holds only {len(bucket)} "
                        f"question(s) (< per_bin={per_bin}); taking all of them."
                    )
            else:
                take = rng.sample(bucket, per_bin)
            per_bin_counts.append(len(take))
            selected.extend(take)

        print(
            f"[SearchQA] support-binned selection: scanned {n_scanned} question(s), "
            f"min_answer_snippets={min_support}, per-bin selected={per_bin_counts} "
            f"(n_bins={n_bins}, per_bin={per_bin}); total selected={len(selected)}"
        )

        cap = self.max_test_queries
        if cap is not None and len(selected) > cap:
            print(
                f"[SearchQA] {len(selected)} selected questions capped to "
                f"{cap} (max_test_queries)"
            )
            selected = selected[:cap]
        return selected

    @staticmethod
    def _count_answer_snippets(answer: str, snippets: list[str]) -> int:
        """Number of snippets that contain the answer core (case-insensitive).

        Uses the same answer-bearing test as ``_build_corpus``: a snippet contains
        the answer iff the parenthetical-stripped, lower-cased answer core is a
        substring of the lower-cased snippet.
        """
        core = _strip_parentheticals(answer).lower()
        if not core:
            return 0
        return sum(1 for snippet in snippets if core in snippet.lower())

    def _build_corpus(
        self, selected: list[tuple[str, str, list[str]]]
    ) -> list[str]:
        """Union of the selected questions' snippets, deduplicated.

        Answer-bearing snippets are added first so that, if ``max_index_docs``
        caps the corpus, the documents that carry a gold answer are never the ones
        dropped — gold-in-corpus coverage is preserved regardless of the cap.
        """
        gold_docs: dict[str, None] = {}
        other_docs: dict[str, None] = {}
        for _, answer, snippets in selected:
            core = _strip_parentheticals(answer).lower()
            for snippet in snippets:
                bucket = gold_docs if core and core in snippet.lower() else other_docs
                bucket.setdefault(snippet, None)

        corpus = list(gold_docs)
        for doc in other_docs:
            if doc not in gold_docs:
                corpus.append(doc)

        cap = self.max_index_docs
        if cap is not None and len(corpus) > cap:
            # Answer-bearing snippets lead the list, so capping only trims
            # non-gold context; log it so a shrunk corpus is never silent.
            print(
                f"[SearchQA] corpus of {len(corpus)} snippets capped to {cap} "
                f"(max_index_docs); {len(gold_docs)} answer-bearing snippets kept"
            )
            corpus = corpus[:cap]
        return corpus

    # Snippets are plain-ish Google result text. A light clean unescapes HTML
    # entities, drops any stray tags and collapses embedded newlines / runs of
    # whitespace into single spaces.
    _TAG_RE = re.compile(r"<[^>]+>")

    @classmethod
    def _clean_snippet(cls, snippet: str) -> str:
        text = html.unescape(str(snippet))
        text = cls._TAG_RE.sub(" ", text)
        return " ".join(text.split())

    @staticmethod
    def _extract_snippets(row: dict):
        for item in row.get("search_results") or []:
            if isinstance(item, dict):
                snippet = item.get("snippet")
                if isinstance(snippet, str) and snippet.strip():
                    yield snippet

    @staticmethod
    def _extract_answer(row: dict) -> str:
        answer = row.get("answer")
        if isinstance(answer, str):
            return answer.strip()
        if isinstance(answer, list) and answer:
            first = answer[0]
            if isinstance(first, str):
                return first.strip()
        return ""

    def get_documents(self) -> list[str]:
        if not self._documents:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self._documents

    def get_queries(self) -> list[dict]:
        if not self._queries:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self._queries


def _strip_parentheticals(text: str) -> str:
    """Drop parenthetical groups, e.g. '(Lou) Gehrig' -> 'Gehrig'.

    SearchQA golds often carry a parenthetical first name/alias that the snippet
    may omit; matching the answer against the stripped core avoids classifying a
    snippet that names only the core as answer-free.
    """
    return " ".join(re.sub(r"\([^)]*\)", " ", str(text)).split())
