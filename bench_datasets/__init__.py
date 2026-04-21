from bench_datasets.base_dataset import BaseDataset
from bench_datasets.squad import SQuADDataset
from bench_datasets.chatdoctor import ChatDoctorDataset
from bench_datasets.natural_questions import NaturalQuestionsDataset
from bench_datasets.triviaqa import TriviaQADataset
from bench_datasets.hotpotqa import HotpotQADataset

__all__ = [
    "BaseDataset",
    "SQuADDataset",
    "ChatDoctorDataset",
    "NaturalQuestionsDataset",
    "TriviaQADataset",
    "HotpotQADataset",
]
