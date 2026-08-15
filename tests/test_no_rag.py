"""The zero-shot floor: same questions, same answer length, no corpus anywhere.

The no-retrieval condition is what separates a method that used the private corpus from
one that answered out of the generator's pretraining. That reading only holds if the two
runs differ in retrieval alone — so what is checked here is that the questions, the
answer length and the payload schema are the ones every other method is measured on, and
that nothing in the runner touches a corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.run_no_rag as rn  # noqa: E402
from experiments.runner import METHODS  # noqa: E402
from experiments.score_results import _extract_preds_refs  # noqa: E402

CONFIG_DIR = PROJECT_ROOT / "experiments" / "configs"


def _config(variant: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / variant / "no_rag.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Stubs: no model, no dataset download
# ---------------------------------------------------------------------------

class _StubGenerator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "stub answer"


class _StubLLM:
    def __init__(self, cfg, generator):
        self._cfg = {"name": "stub", "model_name": cfg.get("model_name", "stub")}
        self._generator = generator

    def get_config(self):
        return self._cfg

    def build_generator(self, **_):
        return self._generator


class _StubDataset:
    TASK_INSTRUCTION = "Answer the question with a short factual answer"

    def get_task_instruction(self) -> str:
        return self.TASK_INSTRUCTION

    def get_queries(self) -> list[dict]:
        return [
            {"query": "who wrote hamlet", "answer": "Shakespeare",
             "answers": ["Shakespeare", "William Shakespeare"]},
            {"query": "capital of chile", "answer": "Santiago",
             "favorites": ["Santiago", "Valparaiso"]},
        ]


@pytest.fixture
def run(monkeypatch):
    """Run ``run_experiment`` against the stubs; returns (payload, generator)."""
    def _run(dataset_cfg: dict | None = None, config: dict | None = None):
        generator = _StubGenerator()
        monkeypatch.setattr(rn, "load_llm", lambda cfg: _StubLLM(cfg, generator))
        monkeypatch.setattr(
            rn, "load_dataset_from_config", lambda cfg, seed, **kw: _StubDataset()
        )
        cfg = config or rn.default_config()
        payload = rn.run_experiment(cfg, dataset_cfg or {"name": "squad"}, cfg["llm"])
        return payload, generator

    return _run


# ---------------------------------------------------------------------------
# The prompt is the question and nothing else
# ---------------------------------------------------------------------------

def test_the_prompt_is_the_task_instruction_and_the_question(run):
    _, generator = run()
    assert generator.prompts[0] == (
        "Answer the question with a short factual answer\n"
        "Question: who wrote hamlet\nAnswer:"
    )


def test_the_instruction_is_the_dataset_s_own(run):
    # The task depends on the dataset, not on the method, so the floor is asked for the
    # same kind of answer the curves above it were asked for.
    _, generator = run()
    assert generator.prompts[0].startswith(_StubDataset.TASK_INSTRUCTION)


def test_the_runner_never_builds_an_index(run):
    """No embedder, no index, no retriever — in the source, not only in this run.

    A stubbed run proves nothing about a corpus the runner never had: what would make
    this condition wrong is code that quietly retrieves, and on the DPR datasets that is
    also ~20 h and tens of GB spent to produce a prompt that discards it.
    """
    source = (PROJECT_ROOT / "experiments" / "run_no_rag.py").read_text(encoding="utf-8")
    for forbidden in ("HFEmbedder", "FAISSIndexer", "Retriever", "index_cache", "RAGPipeline"):
        assert forbidden not in source, f"run_no_rag references {forbidden}"


# ---------------------------------------------------------------------------
# The payload the scorer and the analysis read
# ---------------------------------------------------------------------------

def test_the_scorer_finds_the_predictions(run):
    payload, _ = run()
    preds, refs = _extract_preds_refs(payload)
    assert preds == ["stub answer", "stub answer"]
    assert refs == ["Shakespeare", "Santiago"]


def test_the_payload_names_the_method_and_the_dataset(run):
    payload, _ = run({"name": "movielens"})
    assert payload["method"] == "no_rag"
    # A plain name: the CSV writes this field verbatim.
    assert payload["dataset"] == "movielens"


def test_the_alias_answers_and_the_set_valued_gold_survive(run):
    payload, _ = run()
    items = payload["results"]
    assert items[0]["answers"] == ["Shakespeare", "William Shakespeare"]
    # Movielens-style gold drives hit_rate, which is scored per item.
    assert items[1]["favorites"] == ["Santiago", "Valparaiso"]


def test_every_question_carries_its_id(run):
    payload, _ = run()
    ids = [it["query_id"] for it in payload["results"]]
    assert all(ids), "without query_id no run can be asked whether it answered the same set"
    assert len(set(ids)) == len(ids)


def test_the_answer_length_is_reported_where_every_method_reports_it(run):
    payload, _ = run()
    assert payload["answer_tokens"] == payload["generation"]["max_new_tokens"] == 30


def test_the_payload_claims_no_place_on_the_privacy_axis(run):
    # This condition reads no private data at all. An epsilon here would put the floor
    # somewhere on a scale it is off entirely, and the analysis reads these keys.
    payload, _ = run()
    for key in ("epsilon", "epsilon_total", "epsilon_target", "epsilon_workload",
                "workload_queries", "epsilon_workload_by_q"):
        assert key not in payload


def test_the_seeds_are_recorded_apart(run):
    config = rn.default_config()
    config["data_seed"] = 7
    config["run_seed"] = 11
    payload, _ = run(config=config)
    assert (payload["data_seed"], payload["run_seed"]) == (7, 11)


def test_the_analysis_reads_the_record(run):
    from analysis.load_results import _dispatch

    payload, _ = run()
    payload["metrics"] = {"contains_accuracy": 0.5}
    record = _dispatch(
        payload, source_file="no_rag/squad/stub/results.json", variant="full",
        method="no_rag", recompute_vanilla=False, bertscore=False,
    )
    assert record.method == "no_rag"
    assert record.model == "sshleifer/tiny-gpt2"
    assert record.epsilon_value is None
    assert record.answer_tokens == 30
    assert record.utility["contains_accuracy"] == 0.5


# ---------------------------------------------------------------------------
# The grid is the one every other method runs
# ---------------------------------------------------------------------------

def test_the_orchestrator_can_run_it():
    info = METHODS["no_rag"]
    for key in ("script", "full_config", "smoke_config", "mia_config"):
        assert (PROJECT_ROOT / info[key]).exists(), f"{key} -> {info[key]} does not exist"


def test_the_same_generators_as_the_reference_rag_condition():
    vanilla = yaml.safe_load((CONFIG_DIR / "full" / "vanilla.yaml").read_text(encoding="utf-8"))
    assert _config("full")["llms"] == vanilla["llms"]


def test_the_same_datasets_in_the_same_order():
    vanilla = yaml.safe_load((CONFIG_DIR / "full" / "vanilla.yaml").read_text(encoding="utf-8"))
    names = [d["name"] for d in _config("full")["datasets"]]
    assert names == [d["name"] for d in vanilla["datasets"]]


def test_the_question_selecting_filters_match_the_reference_condition():
    """Same questions, which is the entire reason this condition exists.

    Four of the six loaders partition one dataset between corpus and questions, so their
    corpus size selects questions too and has to match. Natural Questions and TriviaQA
    draw their questions from the QA splits independently of the passages — the same fact
    that lets them share one index cache key across differing query filters — so the
    baseline caps a corpus it never reads instead of loading 21M passages per cell.
    """
    from experiments.common import _QUESTION_INDEPENDENT_CORPORA

    vanilla = yaml.safe_load((CONFIG_DIR / "full" / "vanilla.yaml").read_text(encoding="utf-8"))
    mine = {d["name"]: d for d in _config("full")["datasets"]}
    theirs = {d["name"]: d for d in vanilla["datasets"]}
    assert _config("full")["dataset"]["max_test_queries"] == vanilla["dataset"]["max_test_queries"]
    for name, entry in theirs.items():
        if name in _QUESTION_INDEPENDENT_CORPORA:
            continue
        assert mine[name] == entry, f"{name}: the baseline would ask different questions"


def test_the_attack_probes_the_documents_the_methods_are_attacked_on():
    """An AUC floor bounds only the AUCs measured on the same targets.

    The member / non-member split is drawn from the loaded corpus, so a different corpus
    cap samples different documents and with them different probe queries. The corpus cap
    in the utility config is a marker that this runner reads no corpus, not a corpus
    size, and carrying it into the attack config would size the pool the targets come
    from.
    """
    vanilla = yaml.safe_load((CONFIG_DIR / "mia" / "vanilla.yaml").read_text(encoding="utf-8"))
    mine = {d["name"]: d for d in _config("mia")["datasets"]}
    for entry in vanilla["datasets"]:
        name = entry["name"]
        assert mine[name]["max_index_docs"] == entry["max_index_docs"], (
            f"{name}: the floor would be probed on a different sample of the corpus"
        )


def test_the_answer_length_matches_the_protocol():
    for variant in ("full", "smoke", "mia"):
        assert _config(variant)["generation"]["max_new_tokens"] == 30


def test_the_metric_set_is_the_shared_one():
    metrics = _config("full")["metrics"]
    assert metrics["utility"] == [
        "match_accuracy", "contains_accuracy", "rouge_l", "f1_score",
        "bert_score_f1", "perplexity",
    ]
    # The attack is a run of its own (python experiments/runner.py --mia-only).
    assert metrics["privacy"] == []


def test_no_budget_knob_is_swept():
    config = _config("full")
    for key in ("epsilon_targets", "workload_queries"):
        assert key not in config, f"the non-private floor declares {key}"
