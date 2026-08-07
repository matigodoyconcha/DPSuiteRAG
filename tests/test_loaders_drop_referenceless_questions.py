"""A question with no gold to score against is never asked.

The scorer refuses to compare an empty prediction with an empty reference, but a skip
there is per-cell: whether an item is dropped depends on what the method answered, so
two methods would be measured over different question sets. The question set has to be
the same for everyone, so the loaders are where a reference-less row leaves — the
scorer's skip stays as the backstop for older result files and for golds that only
normalization empties.

The loaders are exercised against in-memory rows; nothing is downloaded.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bench_datasets.chatdoctor as chatdoctor  # noqa: E402
import bench_datasets.hotpotqa as hotpotqa  # noqa: E402
import bench_datasets.natural_questions as natural_questions  # noqa: E402
import bench_datasets.squad as squad  # noqa: E402
import bench_datasets.triviaqa as triviaqa  # noqa: E402


def _questions(dataset) -> list[str]:
    with warnings.catch_warnings():
        # The DPR corpus/relevance files are not present in a test environment; the
        # loaders warn and fall back, which is not what these tests are about.
        warnings.simplefilter("ignore")
        dataset.load()
    return [q["query"] for q in dataset.get_queries()]


def test_natural_questions_drops_a_question_with_no_answer(monkeypatch):
    rows = {
        "train": [{"question": "seed", "answer": ["Paris"]}],
        "validation": [
            {"question": "answerable", "answer": ["Paris"]},
            {"question": "no gold at all", "answer": []},
            {"question": "gold is an article", "answer": ["the"]},
        ],
    }
    monkeypatch.setattr(natural_questions, "load_dataset", lambda *a, **k: rows)
    dataset = natural_questions.NaturalQuestionsDataset(
        max_index_docs=5, max_test_queries=10, dpr_corpus_path="/nonexistent"
    )
    assert _questions(dataset) == ["answerable"]


def test_triviaqa_drops_a_question_with_no_answer(monkeypatch):
    rows = {
        "train": [{"question": "seed", "answer": {"value": "Paris"}, "search_results": {}}],
        "validation": [
            {"question": "answerable", "answer": {"value": "Paris", "aliases": []}},
            {"question": "no gold at all", "answer": {"value": "", "aliases": []}},
            {"question": "gold is an article", "answer": {"value": "a", "aliases": []}},
        ],
    }
    monkeypatch.setattr(triviaqa, "load_dataset", lambda *a, **k: rows)
    dataset = triviaqa.TriviaQADataset(
        max_index_docs=5, max_test_queries=10, dpr_corpus_path="/nonexistent"
    )
    assert _questions(dataset) == ["answerable"]


def test_squad_drops_a_question_with_no_answer_span(monkeypatch):
    rows = {
        "train": [
            {"question": "seed", "context": "ctx", "answers": {"text": ["Paris"]}, "id": "0"}
        ],
        "validation": [
            {"question": "answerable", "context": "c", "answers": {"text": ["Paris"]}, "id": "1"},
            {"question": "no gold at all", "context": "c", "answers": {"text": []}, "id": "2"},
            {"question": "gold is an article", "context": "c", "answers": {"text": ["The"]}, "id": "3"},
        ],
    }
    monkeypatch.setattr(squad, "load_dataset", lambda *a, **k: rows)
    dataset = squad.SQuADDataset(max_index_docs=5, max_test_queries=10)
    assert _questions(dataset) == ["answerable"]


def test_hotpotqa_drops_a_question_with_no_answer(monkeypatch):
    ctx = {"title": ["T"], "sentences": [["a sentence."]]}
    rows = {
        "train": [{"question": "seed", "answer": "Paris", "context": ctx}],
        "validation": [
            {"question": "answerable", "answer": "Paris", "context": ctx},
            {"question": "no gold at all", "answer": "", "context": ctx},
            {"question": "gold is an article", "answer": "an", "context": ctx},
        ],
    }
    monkeypatch.setattr(hotpotqa, "load_dataset", lambda *a, **k: rows)
    dataset = hotpotqa.HotpotQADataset(max_index_docs=5, max_test_queries=10)
    assert _questions(dataset) == ["answerable"]


def test_chatdoctor_drops_a_conversation_with_no_doctor_response(monkeypatch):
    # The loader takes its questions from the END of the shuffled data, so give it
    # rows whose corpus half and question half are both well formed.
    corpus = [{"input": f"corpus {i}", "output": "advice"} for i in range(4)]
    questions = [
        {"input": "answerable", "output": "take an aspirin"},
        {"input": "no gold at all", "output": ""},
    ]
    monkeypatch.setattr(chatdoctor, "load_dataset", lambda *a, **k: corpus + questions)
    dataset = chatdoctor.ChatDoctorDataset(max_index_docs=4, max_test_queries=2)
    asked = _questions(dataset)
    assert "no gold at all" not in asked


def test_searchqa_refuses_a_question_whose_answer_normalisation_empties():
    from bench_datasets.searchqa import SearchQADataset

    dataset = SearchQADataset(max_test_queries=1)
    snippets = {"search_results": [{"snippet": "a helpful snippet"}]}
    assert dataset._parse_question({"question": "q", "answer": "Paris", **snippets}) is not None
    assert dataset._parse_question({"question": "q", "answer": "", **snippets}) is None
    assert dataset._parse_question({"question": "q", "answer": "the", **snippets}) is None
