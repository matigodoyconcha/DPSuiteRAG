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

    # Task: reading comprehension. Answer is an exact short span from the context.
    TASK_INSTRUCTION = "Answer with the exact short span from the context only"

    def __init__(
        self,
        index_split: str = "train",
        test_split: str = "validation",
        max_index_docs: int | None = 200,
        max_test_queries: int | None = 50,
        random_seed: int = 42,
        relevant_docs_per_query: int = 12,
    ):
        self.index_split = index_split
        self.test_split = test_split
        self.max_index_docs = max_index_docs
        self.max_test_queries = max_test_queries
        self.random_seed = random_seed
        # Voting / sample-and-aggregate methods (DPVoteRAG, DPSparseVoteRAG, DP-KSA)
        # assume MANY relevant documents per query (the papers filter to 20-59).
        # SQuAD has exactly one gold context per question, so the m voters / N
        # samples mostly receive irrelevant docs and never agree (LD-Gumbel halts;
        # DP-KSA's keyword gaps stay <=2 so PTR never releases). To hand retrieval
        # several relevant documents we pull additional DISTINCT corpus passages
        # that actually contain the query's answer span — never duplicate copies of
        # the same passage. If a query has fewer such real passages than requested
        # we add only the ones that exist (no artificial padding). Set to 1 to
        # index just the single gold context per query.
        self.relevant_docs_per_query = int(relevant_docs_per_query)

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

        documents = list(unique_contexts.keys())

        # Hand the voting / sample-and-aggregate methods several RELEVANT documents
        # per query by adding additional DISTINCT corpus passages whose text
        # contains the query's answer span (the gold context is already seeded
        # above). These are real, different passages — not copies — so voters see
        # genuine agreement signal. Queries with fewer answer-bearing passages than
        # requested simply get fewer; nothing is fabricated.
        if self.relevant_docs_per_query > 1:
            seen = set(documents)
            # Search pool: shuffled train contexts plus the test contexts, each
            # lowercased once so the per-query substring scan stays cheap.
            candidates: list[tuple[str, str]] = []
            candidate_seen: set[str] = set()
            for row in train_rows + test_rows:
                ctx = (row.get("context") or "").strip()
                if ctx and ctx not in candidate_seen:
                    candidate_seen.add(ctx)
                    candidates.append((ctx, ctx.lower()))

            for q in queries:
                answer = (q.get("answer") or "").strip()
                if not answer:
                    continue
                needle = answer.lower()
                # The gold context (if it contains the answer) already counts as one.
                found = 1 if (q["context"] and needle in q["context"].lower()) else 0
                for ctx, ctx_lower in candidates:
                    if found >= self.relevant_docs_per_query:
                        break
                    if ctx in seen:
                        continue
                    if needle in ctx_lower:
                        documents.append(ctx)
                        seen.add(ctx)
                        found += 1

        self._documents = documents

    def get_documents(self) -> list[str]:
        if not self._documents:
            raise RuntimeError("Dataset not loaded. Call load() before get_documents().")
        return self._documents

    def get_queries(self) -> list[dict]:
        if not self._queries:
            raise RuntimeError("Dataset not loaded. Call load() before get_queries().")
        return self._queries
