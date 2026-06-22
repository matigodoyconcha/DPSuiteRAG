import random

from datasets import load_dataset

from bench_datasets.base_dataset import BaseDataset


class MedicalDirichletPhi3Dataset(BaseDataset):
    """
    Synthetic medical QA dataset generated with Phi-3.5 and a Dirichlet-based disease
    distribution.

    Source: sarus-tech/medical_dirichlet_phi3 on Hugging Face (the dataset
    introduced by DP-RAG, Grislain et al.). DP-SynRAG reuses this same dataset
    in its evaluation under the name *Medical Synth*.

    The dataset has a single train split with patient requests, doctor responses,
    symptoms, diseases, and treatments.
    """

    # Task: fictional-disease QA. The model must name the disease only, so the
    # reference is the disease name (not the full doctor passage) — this makes
    # contains/match accuracy meaningful instead of pinned at 0.
    TASK_INSTRUCTION = (
        "Using the context, answer with the name of the disease only"
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

        # DPRAG-02: index ONLY the doctor_response. The reference implementation
        # (Grislain et al., dp-rag/test_data.py:68) indexes the doctor turn alone,
        # and the retrieval exponential-mechanism score window [-0.5, 0.8] was
        # calibrated to that distribution. Prepending the patient turn shifts the
        # similarity range and invalidates the window, so we keep the doctor
        # response only. This applies to every DP method that reuses this dataset.
        unique_docs: dict[str, None] = {}
        for row in index_rows:
            doctor_response = (row.get("doctor_response") or "").strip()

            if doctor_response:
                unique_docs[doctor_response] = None

            if self.max_index_docs is not None and len(unique_docs) >= self.max_index_docs:
                break

        self._documents = list(unique_docs.keys())

        queries: list[dict] = []
        for row in test_rows:
            doctor_response = (row.get("doctor_response") or "").strip()
            disease = (row.get("disease") or "").strip()

            # DPRAG-03: build the query from the symptom list with the reference
            # wording (dp-rag/test_dp_rag.py:70), not the raw patient_request. The
            # dense retriever was calibrated against this symptom-template phrasing;
            # feeding raw patient text shifts the query embedding out of the
            # calibrated similarity range. The `symptom` field is a list of strings
            # (we also accept a plain string for robustness).
            raw_symptoms = row.get("symptom") or []
            if isinstance(raw_symptoms, str):
                symptom_text = raw_symptoms.strip()
                symptoms = [symptom_text] if symptom_text else []
            else:
                symptoms = [str(s).strip() for s in raw_symptoms if str(s).strip()]
                symptom_text = ", ".join(symptoms)

            # Disease-prediction task: the gold answer is the disease name and the
            # query is the symptom template, so skip rows missing either (an empty
            # reference or empty query makes every metric meaningless).
            if not symptom_text or not disease:
                continue

            query = (
                f"I am experiencing the following symptoms: {symptom_text}. "
                "What is my disease?"
            )

            queries.append(
                {
                    "query": query,
                    "answer": disease,  # reference = disease name only
                    "doctor_response": doctor_response,  # full passage kept for reference/debug
                    "id": row.get("patient_id"),
                    "patient_name": (row.get("patient_name") or "").strip(),
                    "symptoms": symptoms,
                    "disease": disease,
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