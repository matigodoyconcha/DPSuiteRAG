"""Medical Synth dataset — best-effort reproduction of DP-SynRAG's benchmark.

Reference: arXiv:2510.06719v2, Section 5.1 / Appendix B.

The paper builds a synthetic medical-record corpus around 100 fictional disease
names. For each document it samples a disease, draws a top-k subset of symptoms
from a Dirichlet over a fixed 50-symptom vocabulary, and writes a short record.
The exact generator is not released, so we provide a deterministic best-effort
reconstruction here:

    - 100 fictional disease names (mix of Greek/Latin morphemes).
    - 50-symptom vocabulary.
    - Dirichlet(alpha=0.5) over symptoms per disease.
    - Template-based record:
        "Patient with {disease} presents with {sym1}, {sym2}, {sym3}.
         Recommended workup: {workup}."

Documents are indexed; queries are "Given these symptoms: ..., what disease?"
with the disease name as gold answer. The 100 disease names form the
``private_entities`` set used by leakage metrics.
"""

from __future__ import annotations

import random
from typing import List

import numpy as np

from bench_datasets.base_dataset import BaseDataset


_DISEASE_PREFIXES = [
    "neuro", "cardio", "hepato", "nephro", "pulmo", "gastro", "osteo",
    "dermo", "endo", "hemo", "musculo", "lympho", "myo", "psycho", "onco",
    "rheumato", "ophthalmo", "oto", "pharyngo", "thrombo",
]
_DISEASE_ROOTS = [
    "morbus", "patica", "tropia", "plasia", "trophia", "algia", "itis",
    "osis", "emia", "pathia",
]
_DISEASE_SUFFIXES = ["", "-ix", "-on", "-us", "-ica", "-um", "-ax"]

_SYMPTOMS = [
    "fever", "fatigue", "headache", "nausea", "vomiting", "dizziness",
    "blurred vision", "chest pain", "shortness of breath", "cough",
    "abdominal pain", "joint pain", "muscle weakness", "rash", "swelling",
    "loss of appetite", "weight loss", "weight gain", "night sweats", "chills",
    "tremor", "numbness", "tingling", "memory loss", "confusion", "seizures",
    "palpitations", "dry mouth", "thirst", "frequent urination",
    "constipation", "diarrhea", "bloating", "heartburn", "anxiety",
    "insomnia", "drowsiness", "ringing in ears", "hearing loss", "tinnitus",
    "back pain", "stiffness", "bruising", "bleeding", "itching", "redness",
    "dry skin", "hair loss", "brittle nails", "yellow eyes",
]

_WORKUPS = [
    "complete blood count",
    "comprehensive metabolic panel",
    "thyroid panel",
    "lipid panel",
    "abdominal ultrasound",
    "chest X-ray",
    "MRI of the brain",
    "ECG",
    "urinalysis",
    "liver function tests",
]


class MedicalSynthDataset(BaseDataset):
    """Best-effort reconstruction of DP-SynRAG's Medical Synth corpus."""

    def __init__(
        self,
        n_diseases: int = 100,
        n_documents: int = 1000,
        n_queries: int = 200,
        symptoms_per_doc: int = 4,
        dirichlet_alpha: float = 0.5,
        max_index_docs: int | None = None,
        max_test_queries: int | None = None,
        random_seed: int = 42,
    ):
        if n_diseases <= 0:
            raise ValueError("n_diseases must be >= 1.")
        if n_documents <= 0:
            raise ValueError("n_documents must be >= 1.")
        self.n_diseases = int(n_diseases)
        self.n_documents = int(n_documents)
        self.n_queries = int(n_queries)
        self.symptoms_per_doc = int(symptoms_per_doc)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.max_index_docs = max_index_docs
        self.max_test_queries = max_test_queries
        self.random_seed = int(random_seed)

        self._documents: list[str] = []
        self._queries: list[dict] = []
        self._private_entities: list[str] = []

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def load(self) -> None:
        rng = random.Random(self.random_seed)
        np_rng = np.random.default_rng(self.random_seed)

        disease_names = self._build_disease_names(rng, self.n_diseases)
        symptom_pool = list(_SYMPTOMS)
        symptom_distributions = {
            d: np_rng.dirichlet(
                [self.dirichlet_alpha] * len(symptom_pool)
            )
            for d in disease_names
        }

        n_docs = (
            min(self.n_documents, self.max_index_docs)
            if self.max_index_docs
            else self.n_documents
        )
        documents: list[str] = []
        for _ in range(n_docs):
            disease = rng.choice(disease_names)
            probs = symptom_distributions[disease]
            picks_idx = np_rng.choice(
                len(symptom_pool),
                size=min(self.symptoms_per_doc, len(symptom_pool)),
                replace=False,
                p=probs,
            )
            picks = [symptom_pool[i] for i in picks_idx]
            workup = rng.choice(_WORKUPS)
            documents.append(
                f"Patient with {disease} presents with "
                f"{', '.join(picks[:-1])} and {picks[-1]}. "
                f"Recommended workup: {workup}."
            )

        n_q = (
            min(self.n_queries, self.max_test_queries)
            if self.max_test_queries
            else self.n_queries
        )
        queries: list[dict] = []
        for _ in range(n_q):
            disease = rng.choice(disease_names)
            probs = symptom_distributions[disease]
            picks_idx = np_rng.choice(
                len(symptom_pool),
                size=min(self.symptoms_per_doc, len(symptom_pool)),
                replace=False,
                p=probs,
            )
            picks = [symptom_pool[i] for i in picks_idx]
            queries.append(
                {
                    "query": (
                        "A patient presents with "
                        f"{', '.join(picks[:-1])} and {picks[-1]}. "
                        "What is the most likely disease?"
                    ),
                    "answer": disease,
                }
            )
        self._documents = documents
        self._queries = queries
        self._private_entities = list(disease_names)

    @staticmethod
    def _build_disease_names(rng: random.Random, n: int) -> list[str]:
        seen: set[str] = set()
        names: list[str] = []
        attempts = 0
        while len(names) < n and attempts < n * 50:
            attempts += 1
            name = (
                rng.choice(_DISEASE_PREFIXES)
                + rng.choice(_DISEASE_ROOTS)
                + rng.choice(_DISEASE_SUFFIXES)
            )
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def get_documents(self) -> List[str]:
        if not self._documents:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self._documents

    def get_queries(self) -> List[dict]:
        if not self._queries:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self._queries

    def get_private_entities(self) -> List[str]:
        return self._private_entities
