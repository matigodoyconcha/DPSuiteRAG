import random

from datasets import load_dataset

from bench_datasets.base_dataset import BaseDataset


class MedicalDirichletPhi3Dataset(BaseDataset):
    """
    Synthetic medical QA dataset generated with Phi-3.5 and a Dirichlet-based disease
    distribution.

    Source: sarus-tech/medical_dirichlet_phi3 on Hugging Face.
    The dataset has a single train split with patient requests, doctor responses,
    symptoms, diseases, and treatments.
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
        dataset = load_dataset("sarus-tech/medical_dirichlet_phi3")

        if self.index_split not in dataset:
            self.index_split = "train"
        if self.test_split not in dataset:
            self.test_split = "train"

        index_rows = list(dataset[self.index_split])
        test_rows = list(dataset[self.test_split])

        rng = random.Random(self.random_seed)
        rng.shuffle(index_rows)
        rng.shuffle(test_rows)

        unique_docs: dict[str, None] = {}
        for row in index_rows:
            patient_request = (row.get("patient_request") or "").strip()
            doctor_response = (row.get("doctor_response") or "").strip()

            if patient_request and doctor_response:
                document = f"Patient: {patient_request}\nDoctor: {doctor_response}"
                unique_docs[document] = None

            if self.max_index_docs is not None and len(unique_docs) >= self.max_index_docs:
                break

        self._documents = list(unique_docs.keys())

        queries: list[dict] = []
        for row in test_rows:
            patient_request = (row.get("patient_request") or "").strip()
            doctor_response = (row.get("doctor_response") or "").strip()

            if not patient_request:
                continue

            queries.append(
                {
                    "query": patient_request,
                    "answer": doctor_response,
                    "id": row.get("patient_id"),
                    "patient_name": (row.get("patient_name") or "").strip(),
                    "symptoms": row.get("symptom") or [],
                    "disease": (row.get("disease") or "").strip(),
                    "treatment": (row.get("treatment") or "").strip(),
                }
            )

            if self.max_test_queries is not None and len(queries) >= self.max_test_queries:
                break

        self._queries = queries

    def get_documents(self) -> list[str]:
        if not self._documents:
            raise RuntimeError("Dataset not loaded. Call load() before get_documents().")
        return self._documents

    def get_queries(self) -> list[dict]:
        if not self._queries:
            raise RuntimeError("Dataset not loaded. Call load() before get_queries().")
        return self._queries

    def get_private_entities(self) -> list[str]:
        return ["patient_name", "symptom", "disease", "treatment"]