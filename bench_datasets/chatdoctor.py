import random
from datasets import load_dataset
from bench_datasets.base_dataset import BaseDataset

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

        # 2. QUERIES: evaluate using the patient's question as the query
        test_limit = self.max_test_queries if self.max_test_queries else 50
        test_data = all_data[index_limit : index_limit + test_limit]
        
        queries = []
        for row in test_data:
            patient_query = row.get("input", "").strip() or row.get("instruction", "").strip()
            answer = row.get("output", "").strip()
            
            if patient_query:
                queries.append({
                    "query": patient_query,
                    "answer": answer
                })
        
        self._queries = queries

    def get_documents(self) -> list[str]:
        return self._documents

    def get_queries(self) -> list[dict]:
        return self._queries

    def get_private_entities(self) -> list[str]:
        return ["disease", "medication", "symptoms"]