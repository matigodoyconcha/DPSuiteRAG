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
    DEFAULT_TRIVIA_RELEVANCE_PATH,
    load_relevance_records,
    relevance_available,
)
from metrics.utility import has_scoreable_reference

class TriviaQADataset(BaseDataset):
    """
    TriviaQA: A reading comprehension dataset with trivia questions and evidence documents
    from Wikipedia and the web. Each question has multiple verified answers and associated
    passages. Used here in the 'rc' (reading comprehension) configuration.

    Source: Joshi et al., "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset
    for Reading Comprehension", ACL 2017. HuggingFace: `trivia_qa` (config: `rc`).

    Relevance-binning query filter (opt-in; off unless ``relevance_min`` is set):
    when enabled and the DPR relevance file (``relevance_path``) is available, the
    evaluation queries are built directly from the DPR biencoder dev records whose
    relevance (the number of gold positive passages, ``len(positive_ctxs)``) lies
    in ``[relevance_min, relevance_max)``, split into ``n_relevance_bins``
    equal-width bins with ``per_bin`` questions sampled per bin. This matches Koga
    et al. (arXiv:2412.04697v3), who evaluate on the DPR dev questions themselves
    and keep 20 <= relevance < 60 in 4 bins of 25. When the relevance file is
    missing the filter is skipped (with a warning) and query selection falls back
    to the ordinary trivia_qa validation path. See ``bench_datasets.dpr_relevance``.
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
        relevance_min: int | None = None,
        relevance_max: int | None = None,
        n_relevance_bins: int | None = None,
        per_bin: int | None = None,
        relevance_path=DEFAULT_TRIVIA_RELEVANCE_PATH,
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
                "relevance-binning query filter is skipped for TriviaQA. "
                "Run `python scripts/download_dpr_relevance.py` to enable it.",
                stacklevel=2,
            )

        queries: list[dict] = []
        for row in dataset[self.test_split]:
            question = (row.get("question") or "").strip()
            ans_obj = row.get("answer", {}) or {}
            answer = str(ans_obj.get("value", "")).strip()
            # Keep the verified aliases (TriviaQA gives many accepted forms); the
            # utility metrics score max-over-aliases so a correct answer in a
            # different surface form is not a false negative.
            answers = [answer] + [str(a).strip() for a in (ans_obj.get("aliases") or [])]
            answers = [a for a in dict.fromkeys(answers) if a] or [answer]

            # Nothing left to score against after answer normalization means the
            # question cannot be answered right or wrong, and an empty prediction
            # would match it exactly. It is not asked.
            if not question or not has_scoreable_reference(answers):
                continue

            queries.append({
                "query": question,
                "answer": answer,
                "answers": answers,
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
            answers = [a for a in (record.get("answers") or []) if a]
            answer = answers[0] if answers else ""
            # A record with no gold left after answer normalization is not an
            # available question: it cannot be answered right or wrong, so it is
            # neither binned nor counted towards the band.
            if not has_scoreable_reference(answers or [answer]):
                continue
            n_in_band += 1
            bin_idx = min((relevance - self.relevance_min) // bin_width, n_bins - 1)
            bins[bin_idx].append({
                "query": record.get("question", ""),
                "answer": answer,
                "answers": answers or [answer],
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
                        f"TriviaQA relevance bin {b} [{lo},{lo + bin_width}) "
                        f"has {len(bucket)} record(s) < per_bin={per_bin}; taking all.",
                        stacklevel=2,
                    )
            per_bin_selected.append(len(chosen))
            selected.extend(chosen)

        print(
            f"[TriviaQA] relevance filter: {n_in_band} DPR record(s) in "
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
