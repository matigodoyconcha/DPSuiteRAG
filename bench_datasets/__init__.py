"""Dataset loaders for RAG benchmarking.

Provides unified interfaces for multiple QA datasets used in evaluating
RAG and differentially private RAG methods.

Datasets included:
- SQuAD: Stanford Question Answering Dataset
- ChatDoctor: Medical QA dataset
- Natural Questions: Google Natural Questions
- TriviaQA: Large-scale trivia QA
- HotpotQA: Multi-hop reasoning QA
"""

from bench_datasets.base_dataset import BaseDataset
from bench_datasets.squad import SQuADDataset
from bench_datasets.chatdoctor import ChatDoctorDataset
from bench_datasets.natural_questions import NaturalQuestionsDataset
from bench_datasets.triviaqa import TriviaQADataset
from bench_datasets.hotpotqa import HotpotQADataset
from bench_datasets.medical_dirichlet_phi3 import MedicalDirichletPhi3Dataset

__all__ = [
    "BaseDataset",
    "SQuADDataset",
    "ChatDoctorDataset",
    "NaturalQuestionsDataset",
    "TriviaQADataset",
    "HotpotQADataset",
    "MedicalDirichletPhi3Dataset",
]
