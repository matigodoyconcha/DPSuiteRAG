import random
import warnings

from datasets import load_dataset

from bench_datasets.base_dataset import BaseDataset
from bench_datasets.dpr_corpus import (
    DEFAULT_DPR_CORPUS_PATH,
    corpus_available,
    load_dpr_passages,
)
from bench_datasets.dpr_relevance import (
    DEFAULT_NQ_RELEVANCE_PATH,
    load_relevance_records,
    relevance_available,
)

class NaturalQuestionsDataset(BaseDataset):
    """
    Natural Questions (Open): Real questions issued to the Google search engine, with
    answers extracted from Wikipedia articles. Uses the open-domain variant (`nq_open`),
    which omits passage-level annotations. Since no retrievable passages are provided,
    the index is built from Q+A pairs derived from the training split.

    Source: Kwiatkowski et al., "Natural Questions: A Benchmark for Question Answering
    Research", TACL 2019. HuggingFace: `nq_open`.

    Relevance-binning query filter (opt-in; off unless ``relevance_min`` is set):
    when enabled and the DPR relevance file (``relevance_path``) is available, the
    evaluation queries are built directly from the DPR biencoder dev records whose
    relevance (the number of gold positive passages, ``len(positive_ctxs)``) lies
    in ``[relevance_min, relevance_max)``, split into ``n_relevance_bins``
    equal-width bins with ``per_bin`` questions sampled per bin. This matches Koga
    et al. (arXiv:2412.04697v3), who evaluate on the DPR dev questions themselves
    and keep 20 <= relevance < 60 in 4 bins of 25. When the relevance file is
    missing the filter is skipped (with a warning) and query selection falls back
    to the ordinary nq_open validation path. See ``bench_datasets.dpr_relevance``.
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
        relevance_min: int | None = None,
        relevance_max: int | None = None,
        n_relevance_bins: int | None = None,
        per_bin: int | None = None,
        relevance_path=DEFAULT_NQ_RELEVANCE_PATH,
    ):
        self.index_split = index_split
        self.test_split = test_split
        self.max_index_docs = max_index_docs
        self.max_test_queries = max_test_queries
        self.random_seed = random_seed
        self.dpr_corpus_path = dpr_corpus_path
        # Relevance-binning query filter (opt-in): active only when relevance_min
        # is set. relevance_max/n_relevance_bins/per_bin fall back to the paper's
        # 60/4/25 when left None.
        self.relevance_min = relevance_min
        self.relevance_max = relevance_max
        self.n_relevance_bins = n_relevance_bins
        self.per_bin = per_bin
        self.relevance_path = relevance_path
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

        # Relevance-binning query filter (opt-in). When enabled and the DPR
        # relevance file is present, the binned selection replaces the ordinary
        # query set; when the file is missing the filter is skipped (warning) and
        # the ordinary path below runs unchanged.
        if self.relevance_min is not None:
            if relevance_available(self.relevance_path):
                records = load_relevance_records(self.relevance_path)
                self._queries = self._relevance_binned_queries(records)
                return
            warnings.warn(
                f"DPR relevance file not found at {self.relevance_path}; the "
                "relevance-binning query filter is skipped for Natural Questions. "
                "Run `python scripts/download_dpr_relevance.py` to enable it.",
                stacklevel=2,
            )

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

    def _relevance_binned_queries(self, records: list[dict]) -> list[dict]:
        """Select eval queries by DPR relevance bins (Koga et al.).

        Builds the query set directly from the DPR biencoder dev records, so each
        question is paired with its own relevance (gold positive-passage count) and
        gold answers. Keeps records whose relevance is in
        ``[relevance_min, relevance_max)``, buckets them into ``n_relevance_bins``
        equal-width bins and samples ``per_bin`` per bin with a seeded RNG. Bins
        with fewer than ``per_bin`` records contribute all of theirs (a shortfall
        is warned). ``max_test_queries`` caps the final list.
        """
        rng = random.Random(self.random_seed)
        relevance_max = self.relevance_max if self.relevance_max is not None else 60
        n_bins = self.n_relevance_bins if self.n_relevance_bins is not None else 4
        per_bin = self.per_bin if self.per_bin is not None else 25
        bin_width = max(1, (relevance_max - self.relevance_min) // n_bins)

        bins: list[list[dict]] = [[] for _ in range(n_bins)]
        n_in_band = 0
        for record in records:
            relevance = record.get("relevance", 0)
            if not (self.relevance_min <= relevance < relevance_max):
                continue
            n_in_band += 1
            bin_idx = min((relevance - self.relevance_min) // bin_width, n_bins - 1)
            answers = [a for a in (record.get("answers") or []) if a]
            ans = answers[0] if answers else ""
            bins[bin_idx].append({
                "query": record.get("question", ""),
                "answer": ans,
                "answers": answers or [ans],
                "context": "",
            })

        selected: list[dict] = []
        per_bin_selected: list[int] = []
        for b, bucket in enumerate(bins):
            if len(bucket) > per_bin:
                chosen = rng.sample(bucket, per_bin)
            else:
                chosen = list(bucket)
                if len(bucket) < per_bin:
                    lo = self.relevance_min + b * bin_width
                    warnings.warn(
                        f"Natural Questions relevance bin {b} [{lo},{lo + bin_width}) "
                        f"has {len(bucket)} record(s) < per_bin={per_bin}; taking all.",
                        stacklevel=2,
                    )
            per_bin_selected.append(len(chosen))
            selected.extend(chosen)

        print(
            f"[NaturalQuestions] relevance filter: {n_in_band} DPR record(s) in "
            f"[{self.relevance_min},{relevance_max}) (bin_width={bin_width}); "
            f"per-bin selected {per_bin_selected}; total selected {len(selected)}."
        )

        if self.max_test_queries is not None:
            selected = selected[: self.max_test_queries]
        return selected

    def get_documents(self) -> list[str]:
        if not self._documents:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self._documents

    def get_queries(self) -> list[dict]:
        if not self._queries:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self._queries
