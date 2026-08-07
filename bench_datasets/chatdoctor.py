import random
from datasets import load_dataset
from bench_datasets.base_dataset import BaseDataset
from metrics.utility import has_scoreable_reference

class ChatDoctorDataset(BaseDataset):
    """
    ChatDoctor (HealthCareMagic-100k): A medical dialogue dataset containing 100k
    real patient-doctor conversations from the HealthCareMagic platform. Each entry
    includes a patient question and a doctor's response. Used here as a privacy-sensitive
    benchmark, as documents contain personal health information.

    Source: Li et al., "ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language
    Model Meta-AI (LLaMA) Using Medical Domain Knowledge", Cureus 2023.
    HuggingFace: `lavita/ChatDoctor-HealthCareMagic-100k`.
    """

    # Task: medical dialogue. References are paragraph-length doctor advice, so
    # frame the model as a doctor (utility best seen via ROUGE-L / BERTScore).
    TASK_INSTRUCTION = (
        "You are a doctor. Using the context, give brief medical advice for the "
        "patient's question"
    )

    def __init__(
        self,
        index_split: str = "train",
        test_split: str = "train",
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
        dataset_name = "lavita/ChatDoctor-HealthCareMagic-100k"
        try:
            dataset = load_dataset(dataset_name, split="train")
        except Exception as e:
            raise RuntimeError(f"Failed to load {dataset_name}. Error: {e}")
        
        all_data = list(dataset)
        rng = random.Random(self.random_seed)
        rng.shuffle(all_data)

        # 1. INDEX: Q + A concatenated (dataset has no explicit context field)
        index_limit = self.max_index_docs if self.max_index_docs else 200
        index_data = all_data[:index_limit]

        unique_docs: dict[str, None] = {}
        for row in index_data:
            # Prefer 'input' field for the patient question; fall back to 'instruction'
            p_input = row.get("input", "").strip() or row.get("instruction", "").strip()
            p_output = row.get("output", "").strip()

            if p_input and p_output:
                full_doc = f"Q: {p_input}\nA: {p_output}"
                unique_docs[full_doc] = None

        self._documents = list(unique_docs.keys())

        # 2. QUERIES: evaluate using the patient's question as the query.
        # Taken from the END of the shuffled data rather than from just past the
        # corpus, so the question set depends only on the seed and the query count.
        # Anchoring them at index_limit made two methods reading the same dataset at
        # different corpus sizes answer disjoint questions — a 1600-document run and a
        # 20000-document one shared not a single question — which turns a comparison
        # between those methods into a comparison between two different test sets.
        # Disjointness from the corpus is preserved as long as the two do not meet.
        test_limit = self.max_test_queries if self.max_test_queries else 50
        if index_limit + test_limit > len(all_data):
            raise ValueError(
                f"ChatDoctor has {len(all_data)} rows, too few for {index_limit} "
                f"corpus documents plus {test_limit} disjoint test questions."
            )
        test_data = all_data[len(all_data) - test_limit :]
        
        queries = []
        for row in test_data:
            patient_query = row.get("input", "").strip() or row.get("instruction", "").strip()
            answer = row.get("output", "").strip()
            
            # A conversation whose doctor response is empty gives the question no
            # reference to be scored against, and every text metric then reads an
            # unanswered query as a perfect one. It is not asked.
            if patient_query and has_scoreable_reference(answer):
                queries.append({
                    "query": patient_query,
                    "answer": answer
                })
        
        self._queries = queries

    def get_documents(self) -> list[str]:
        return self._documents

    def get_queries(self) -> list[dict]:
        return self._queries

    @staticmethod
    def _split_qa_doc(doc: str) -> tuple[str, str]:
        """Recover (patient question, doctor answer) from a ``Q: ...\\nA: ...`` doc."""
        text = str(doc)
        marker = "\nA:"
        idx = text.find(marker)
        if idx == -1:
            return text, ""
        question = text[:idx].removeprefix("Q:").strip()
        answer = text[idx + len(marker):].strip()
        return question, answer

    def get_mia_split(
        self,
        n_per_class: int = 100,
        index_size: int | None = 2000,
        seed: int = 42,
    ) -> dict | None:
        """S2MIA split matching the DPVoteRAG (Koga et al.) ChatDoctor setup.

        Members are the indexed ``Q: .. A: ..`` conversations; non-members are the
        held-out test conversations (disjoint rows never added to the index). The
        patient question is the attack query and the doctor answer is the held-out
        reference, exactly as in DPVoteRAG's S2MIA evaluation.
        """
        member_docs = list(self._documents)
        if index_size is not None:
            member_docs = member_docs[:index_size]

        member_set = set(member_docs)
        non_member_docs = [
            f"Q: {q['query']}\nA: {q.get('answer', '')}"
            for q in self._queries
            if f"Q: {q['query']}\nA: {q.get('answer', '')}" not in member_set
        ]

        n = min(n_per_class, len(member_docs), len(non_member_docs))
        if n < 1:
            return None

        rng = random.Random(seed)
        member_sample = rng.sample(member_docs, n)
        non_member_sample = rng.sample(non_member_docs, n)

        def probe(doc: str) -> dict:
            question, answer = self._split_qa_doc(doc)
            return {"query": question, "gold": answer, "doc": doc}

        return {
            "index_documents": member_docs,
            "members": [probe(d) for d in member_sample],
            "non_members": [probe(d) for d in non_member_sample],
        }