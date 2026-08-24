"""Drive every runner's epsilon_targets x workload_queries sweep with no model.

The runners have ~0% line coverage: the accounting is tested, the loop that calls it
is not.  A NameError/TypeError/KeyError in that loop is only discovered after hours of
GPU time, so here each runner's `_run_for_llm` / `_run_dataset` is executed end to end
with the model, tokenizer, embedder, index, dataset and DP generator replaced by stubs.
Everything the sweep itself does -- budget derivation, cell naming, payload assembly,
file writing -- is the real code.

Stubs that stand in for a constructed object validate their kwargs against the real
class's signature, so a renamed generator parameter still fails here.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGETS = [10.0, 50.0]
WORKLOADS = [1, 5]
DOCS = ["doc one about paris", "doc two about rome", "doc three about berlin"]
QUERIES = [
    {"query": "capital of france?", "answer": "Paris", "query_id": "q1", "answers": ["Paris"]},
    {"query": "capital of italy?", "answer": "Rome", "query_id": "q2", "answers": ["Rome"]},
]


# ---------------------------------------------------------------------------
# Generic stubs
# ---------------------------------------------------------------------------

class StubDataset:
    def get_documents(self):
        return list(DOCS)

    def get_queries(self):
        return [dict(q) for q in QUERIES]

    def get_task_instruction(self):
        return "Answer the question."

    def get_prompt_template(self):
        return "Context: {context}\nQuestion: {query}\nAnswer:"

    def get_mia_split(self, **_):
        return None

    def get_documents_with_ids(self):
        return list(enumerate(DOCS))


class StubEmbedder:
    model_name = "stub-embedder"

    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class StubIndexer:
    def __init__(self, *a, **kw):
        self.index = None
        self.metadata = []
        self.embeddings_matrix = None
        self.documents = []

    def add_documents(self, *a, **kw):
        return None

    def build(self, *a, **kw):
        return None

    def search(self, *a, **kw):
        return []


class StubRetriever:
    def __init__(self, *a, **kw):
        self.last_retrieved_ids = [0, 1]
        # Runners read per-query DP state off the retriever (private_rag's stateful
        # per-document filter is the reason RecordingRetriever forwards attributes).
        self.last_run_stats = {
            "n_private": 1, "n_baseline": 1, "n_retrieved": 2,
            "epsilon_spent": 0.1, "n_exhausted": 0, "n_docs_blocked": 0,
        }
        self.epsilon = 1.0

    def retrieve(self, query_vector, k=5, **kw):
        return [
            {"doc_id": i, "chunk_id": 0, "text": t, "score": 1.0 / (i + 1)}
            for i, t in enumerate(DOCS[:k or 3])
        ]


class StubTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    vocab_size = 10
    model_max_length = 512

    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()

    def __call__(self, text, **kw):
        return {"input_ids": [[1, 2, 3]]}

    def decode(self, ids, **kw):
        return "stub"

    def encode(self, text, **kw):
        return [1, 2, 3]


class StubModel:
    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()

    def to(self, *a, **kw):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter([])

    device = "cpu"
    config = SimpleNamespace(max_position_embeddings=2048)


def _checked_ctor(real_cls, extra_attrs=None, after_init=None):
    """Stub class that rejects kwargs the real constructor would reject.

    ``after_init(stub, kwargs)`` runs the real calibration a generator would do, so the
    numbers the runner copies into its payload are the run's own and not constants --
    otherwise every cell reports the same epsilon and the report is about the stub.
    """
    sig = inspect.signature(real_cls.__init__)

    class _Stub:
        def __init__(self, *a, **kw):
            sig.bind(self, *a, **kw)  # TypeError on a renamed/removed parameter
            self.init_kwargs = kw
            self.last_run_stats = {
                "dp_spent_epsilon": 0.1, "dp_spent_delta": 1e-6, "exit_status": 0,
                "n_private_tokens": 3, "tokens_generated": 5, "n_noise_flipped": 1,
                "n_baseline_tokens": 5, "keywords": ["a"], "k_hat": 1,
                "ptr_passed": True, "n_tokens": 5, "n_public_tokens": 2,
                "n_retrieved": 2, "budget_exhausted": False, "released_tokens": 3,
                "n_keywords": 1, "epsilon_spent": 0.1,
            }
            self.token_epsilon = 0.01
            # Numeric attributes runners read straight off the generator when building
            # the payload; a callable here would only surface as a TypeError in the stub.
            for attr, val in {
                "tau": 1.0, "sparse_eps": 0.1, "epsilon_token": 0.5, "epsilon_total": 1.0,
                "sigma": 1.0, "gumbel_scale": 1.0, "t_max": 5, "tmax": 5, "k_hat": 1,
                "threshold": 1.0, "delta_ptr": 1e-6, "epsilon": 1.0, "delta": 1e-5,
                "vote_threshold": 1.0, "epsilon_0": 0.5, "c": 1,
            }.items():
                setattr(self, attr, val)
            self.calibration = {
                "gumbel_scale": 1.0, "sigma": 1.0, "delta_ptr": 1e-6,
                "epsilon": 1.0, "delta": 1e-5,
            }
            self.model = StubModel()
            self.tokenizer = StubTokenizer()
            for k, v in (extra_attrs or {}).items():
                setattr(self, k, v)
            if after_init is not None:
                after_init(self, kw)

        def generate(self, *a, **kw):
            return "stub answer"

        def index(self, documents, *a, **kw):
            return None

        def query(self, q, *a, **kw):
            return "stub answer"

        def __getattr__(self, name):
            return lambda *a, **kw: None

    _Stub.__name__ = f"Stub{real_cls.__name__}"
    return _Stub


def _stub_metrics(preds, refs, names, **kw):
    return {"match_accuracy": 0.5, "contains_accuracy": 0.5}


def _patch_common(monkeypatch, mod, tmp_path):
    """Everything every runner shares."""
    for name, repl in [
        ("load_dataset_from_config", lambda *a, **kw: StubDataset()),
        ("load_or_build_index", lambda indexer, build_fn, cache_dir: build_fn()),
        ("nonprivate_ranking", lambda *a, **kw: [0, 1]),
        ("compute_utility_metrics", _stub_metrics),
        ("compute_privacy_metrics", _stub_metrics),
        ("free_memory", lambda *a, **kw: None),
        ("FAISSIndexer", StubIndexer),
        ("CosineSimilarityRetriever", StubRetriever),
        ("AutoTokenizer", StubTokenizer),
        ("AutoModelForCausalLM", StubModel),
        ("build_embedder", lambda *a, **kw: StubEmbedder()),
        ("HFEmbedder", lambda *a, **kw: StubEmbedder()),
        ("HFGenerator", lambda *a, **kw: SimpleNamespace(generate=lambda p: "stub")),
        ("corpus_available", lambda *a, **kw: False),
        ("mia_only_enabled", lambda: False),
    ]:
        if hasattr(mod, name):
            monkeypatch.setattr(mod, name, repl)
    # index_cache_dir must land inside tmp so nothing touches the real cache
    if hasattr(mod, "index_cache_dir"):
        monkeypatch.setattr(
            mod, "index_cache_dir",
            lambda **kw: tmp_path / "idx" / str(abs(hash(frozenset(
                (k, str(v)) for k, v in kw.items()))))
        )
    if hasattr(mod, "cuda_oom_warning"):
        import contextlib
        monkeypatch.setattr(mod, "cuda_oom_warning", lambda *a, **kw: contextlib.nullcontext())
    if hasattr(mod, "model_device_kwargs"):
        monkeypatch.setattr(mod, "model_device_kwargs", lambda *a, **kw: {})
    if hasattr(mod, "resolve_device"):
        monkeypatch.setattr(mod, "resolve_device", lambda *a, **kw: "cpu")
    if hasattr(mod, "build_chunker"):
        monkeypatch.setattr(mod, "build_chunker", lambda cfg: SimpleNamespace(
            chunk=lambda t, **kw: [t], chunk_documents=lambda d, **kw: list(d)))


def _load_cfg(name, tmp_path, shipped_grid=False, **over):
    cfg = yaml.safe_load((PROJECT_ROOT / "experiments/configs/full" / f"{name}.yaml").read_text())
    if not shipped_grid:
        cfg["epsilon_targets"] = list(TARGETS)
        cfg["workload_queries"] = list(WORKLOADS)
    cfg["datasets"] = [{"name": "squad", "max_index_docs": 3, "max_test_queries": 2}]
    cfg["dataset"] = {"name": "squad", "max_index_docs": 3, "max_test_queries": 2}
    cfg["max_queries"] = 2
    cfg["output_dir"] = str(tmp_path / "out")
    cfg["metrics"] = {"utility": ["match_accuracy"], "privacy": []}
    cfg["verbose"] = False
    cfg.update(over)
    return cfg


def _args(**over):
    ns = argparse.Namespace(
        config=None, verbose=False, verbose_max_examples=0, dataset=None,
        llm_name=None, llm_model=None, output=None, seed=None,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _written(tmp_path):
    return sorted(p for p in (tmp_path / "out").rglob("*.json"))


def _assert_grid(tmp_path, expected_cells=4):
    files = _written(tmp_path)
    assert files, "the sweep wrote nothing"
    payloads = [json.loads(p.read_text()) for p in files]
    root = tmp_path / "out"
    print(f"\n  {len(files)} file(s) written:")
    for p, pl in zip(files, payloads):
        print(
            f"    {p.relative_to(root)}  "
            f"target={pl.get('epsilon_target')} Q={pl.get('workload_queries')} "
            f"eps_workload={pl.get('epsilon_workload')} "
            f"err={pl.get('workload_error')}"
        )
    cells = {(pl.get("epsilon_target"), pl.get("workload_queries")) for pl in payloads}
    for pl, p in zip(payloads, files):
        assert pl.get("epsilon_target") is not None, f"{p}: no epsilon_target"
        assert pl.get("workload_queries") is not None, f"{p}: no workload_queries"
        assert "epsilon_workload" in pl, f"{p}: no epsilon_workload"
    assert len(cells) == expected_cells, f"expected {expected_cells} cells, got {sorted(cells)}"
    assert len(files) == expected_cells, (
        f"expected {expected_cells} files, got {len(files)}: cells overwrote each other"
    )
    return payloads


# ---------------------------------------------------------------------------
# Per-runner drivers
# ---------------------------------------------------------------------------

def _drive_dp_rag(monkeypatch, tmp_path):
    import experiments.run_dp_rag as m
    from dp_methods.per_query.dp_rag import DPAggregatedGenerator, DPRetriever, DPTopPRetriever

    _patch_common(monkeypatch, m, tmp_path)
    def _real_token_epsilon(stub, kw):
        # The real generator back-solves the per-token budget from the generation
        # budget by PLD bisection; the runner then reports *that* number, so the stub
        # runs the same inversion instead of a constant.
        stub.epsilon = float(kw["epsilon"])
        stub.delta = float(kw["delta"])
        stub.max_new_tokens = int(kw["max_new_tokens"])
        stub.token_epsilon = DPAggregatedGenerator._compute_token_epsilon(stub)

    monkeypatch.setattr(
        m, "DPAggregatedGenerator",
        _checked_ctor(DPAggregatedGenerator, after_init=_real_token_epsilon),
    )
    monkeypatch.setattr(m, "DPRetriever", _checked_ctor(DPRetriever))
    monkeypatch.setattr(m, "DPTopPRetriever", _checked_ctor(DPTopPRetriever))
    monkeypatch.setattr(m, "RAGPipeline", lambda **kw: SimpleNamespace(
        index=lambda d: None, query=lambda q, k=5: "stub answer"))
    monkeypatch.setattr(m, "_index_once", lambda *a, **kw: None)

    cfg = _load_cfg("dp_rag", tmp_path)
    m._run_for_llm(cfg, "stub/model", "stub__model", False)
    return _assert_grid(tmp_path)


def test_dp_rag_sweep(monkeypatch, tmp_path):
    _drive_dp_rag(monkeypatch, tmp_path)


def test_dp_rag_records_the_split_it_actually_ran_at(monkeypatch, tmp_path):
    """The configured ratio is not the split, so the split has to be measured.

    `retrieval_ratio` fixes eps_r / token_eps, a constant. What a reader needs is
    retrieval's share of the total, eps_r / (eps_r + eps_g), and eps_g is a PLD
    composition of T copies of token_eps — sublinear in the per-token budget. So the
    realised share moves across the sweep with the ratio and T both fixed, and a reader
    who divides the config's two numbers gets a figure no cell ran at.
    """
    payloads = _drive_dp_rag(monkeypatch, tmp_path)

    shares = []
    for pl in payloads:
        eps_r, eps_g = pl["epsilon_retrieval"], pl["epsilon_generation"]
        assert pl["epsilon_retrieval_share"] == pytest.approx(eps_r / (eps_r + eps_g)), (
            f"the recorded share disagrees with the budgets in its own payload: {pl}"
        )
        assert 0.0 < pl["epsilon_retrieval_share"] < 1.0
        shares.append(pl["epsilon_retrieval_share"])

    # The point of recording it per cell: one number from the config would be wrong for
    # every cell but at most one.
    assert len(set(round(s, 9) for s in shares)) > 1, (
        f"the realised split did not move across the sweep: {shares}"
    )


def test_dpvoterag_sweep(monkeypatch, tmp_path):
    import experiments.run_dpvoterag as m
    from dp_methods.per_query.dpvoterag.generator import VoteRAGGenerator

    _patch_common(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "VoteRAGGenerator", _checked_ctor(VoteRAGGenerator))
    monkeypatch.setattr(m, "DPVoteRAGPipeline", lambda **kw: SimpleNamespace(
        index=lambda d: None, query=lambda q, k=5: "stub answer"))

    cfg = _load_cfg("dpvoterag", tmp_path)
    m._run_for_llm(cfg, _args(), "stub/model", "stub__model", False)
    _assert_grid(tmp_path)


def test_dpsparsevoterag_sweep(monkeypatch, tmp_path):
    import experiments.run_dpsparsevoterag as m
    from dp_methods.per_query.dpvoterag.sparse_vote_generator import SparseVoteRAGGenerator

    _patch_common(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "SparseVoteRAGGenerator", _checked_ctor(SparseVoteRAGGenerator))
    monkeypatch.setattr(m, "DPVoteRAGPipeline", lambda **kw: SimpleNamespace(
        index=lambda d: None, query=lambda q, k=5: "stub answer"))

    cfg = _load_cfg("dpsparsevoterag", tmp_path)
    m._run_for_llm(cfg, _args(), "stub/model", "stub__model", False)
    _assert_grid(tmp_path)


def test_dpksa_sweep(monkeypatch, tmp_path):
    import experiments.run_dpksa as m
    from dp_methods.per_query.dpksa.generator import DPKSAGenerator

    _patch_common(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "DPKSAGenerator", _checked_ctor(DPKSAGenerator, after_init=_real_dpksa_calibration))
    monkeypatch.setattr(m, "DPKSAPipeline", lambda **kw: SimpleNamespace(
        index=lambda d: None, query=lambda q, k=5: "stub answer"))

    cfg = _load_cfg("dpksa", tmp_path)
    m._run_for_llm(cfg, _args(), "stub/model", "stub__model", False)
    _assert_grid(tmp_path)


def _real_dpksa_calibration(stub, kw):
    """Run DP-KSA's own calibration, the numbers the runner copies into the payload."""
    from metrics.privacy_accounting_dpksa import calibrate_dpksa

    stub.calibration = calibrate_dpksa(
        target_epsilon=float(kw["target_epsilon"]),
        target_delta=float(kw["target_delta"]),
        ptr_delta_fraction=float(kw.get("ptr_delta_fraction", 0.5)),
        ptr_noise_coupling=float(kw.get("ptr_noise_coupling", 1.0)),
    )


def test_dpksa_sweep_behind_the_typo(monkeypatch, tmp_path):
    """The rest of DP-KSA's loop, with `_budget_floor` corrected in memory.

    Separates 'DP-KSA crashes on line 292' from 'DP-KSA's sweep is otherwise broken':
    without this the first defect masks everything after it.
    """
    import math

    import experiments.run_dpksa as m
    from dp_methods.per_query.dpksa.generator import DPKSAGenerator
    from metrics.workload_accounting import METHOD_KNOB_BUILDERS

    def _fixed_budget_floor(*, ptr_noise_coupling, mechanism_delta=None,
                            mechanism_mechanism_delta=None):
        # Accepts the typo'd keyword too: the call site at run_dpksa.py:293 passes it,
        # so the shim has to tolerate it to get past the first defect.
        mechanism_delta = (
            mechanism_delta if mechanism_delta is not None else mechanism_mechanism_delta
        )
        knob = METHOD_KNOB_BUILDERS["dpksa"]
        lo, hi = 1e-9, 1.0
        for _ in range(100):
            mid = math.sqrt(lo * hi)
            try:
                knob(mid, mechanism_delta=mechanism_delta,
                     ptr_noise_coupling=ptr_noise_coupling)
            except (ValueError, RuntimeError):
                lo = mid
            else:
                hi = mid
        return hi

    _patch_common(monkeypatch, m, tmp_path)
    if hasattr(m, "_budget_floor"):
        monkeypatch.setattr(m, "_budget_floor", _fixed_budget_floor)
    monkeypatch.setattr(m, "DPKSAGenerator", _checked_ctor(DPKSAGenerator, after_init=_real_dpksa_calibration))
    monkeypatch.setattr(m, "DPKSAPipeline", lambda **kw: SimpleNamespace(
        index=lambda d: None, query=lambda q, k=5: "stub answer"))

    cfg = _load_cfg("dpksa", tmp_path)
    m._run_for_llm(cfg, _args(), "stub/model", "stub__model", False)
    _assert_grid(tmp_path)


def test_private_rag_sweep(monkeypatch, tmp_path):
    import experiments.run_private_rag as m
    from dp_methods.per_query.private_rag.generator import PrivateRAGGenerator

    _patch_common(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "PrivateRAGGenerator", _checked_ctor(PrivateRAGGenerator))
    monkeypatch.setattr(m, "_make_retriever", lambda *a, **kw: StubRetriever())
    monkeypatch.setattr(m, "DPVoteRAGPipeline", lambda **kw: SimpleNamespace(
        index=lambda d: None, query=lambda q, k=5: "stub answer"))

    cfg = _load_cfg("private_rag", tmp_path)
    m._run_for_llm(cfg, _args(), "stub/model", "stub__model", False)
    # One stream per target, not one per (target, workload): this method's budget is
    # flat in Q, so the shorter workloads are prefixes of the longest and are read off
    # `prefix_curve` rather than run again. That single stream is still FILED at every
    # workload size — a row that exists only at Q=max is absent from every panel drawn
    # at another Q, and flatness is only visible where there are rows to be flat across.
    _assert_grid(tmp_path, expected_cells=len(TARGETS) * len(WORKLOADS))


class _StubSynIndexer:
    """Stands in for DPSynRAGIndexer: an already-built synthetic corpus."""

    def __init__(self, *a, **kw):
        self.index = object()
        self.synthetic_texts = ["syn one", "syn two"]
        self.last_stats = {
            "n_original_documents": 3, "n_non_empty_subsets": 2, "n_synthetic_kept": 2,
            "synthetic_generator": {"n_tokens_total": 10, "n_noise_flipped_total": 2},
        }

    def build_from_documents(self, docs, task_prefix=""):
        return None

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def load(self, path):
        return None


def test_dp_synrag_sweep(monkeypatch, tmp_path, capsys):
    import experiments.run_dp_synrag as m

    _patch_common(monkeypatch, m, tmp_path)
    cfg = _load_cfg("dp_synrag", tmp_path)
    cfg.setdefault("synthetic_generation", {})["epsilon_targets"] = list(TARGETS)
    cfg["llms"] = [{"name": "huggingface", "model_name": "stub/model"}]
    cfg["llm"] = {"name": "huggingface", "model_name": "stub/model"}

    monkeypatch.setattr(m, "load_config", lambda p: cfg)
    monkeypatch.setattr(m, "_build_indexer", lambda *a, **kw: _StubSynIndexer())
    monkeypatch.setattr(m, "CosineSimilarityRetriever", StubRetriever)
    monkeypatch.setattr(m, "RAGPipeline", lambda **kw: SimpleNamespace(
        index=lambda d: None, query=lambda q, k=5: "stub answer"))
    monkeypatch.setattr(m, "resolve_llm_execution",
                        lambda llms, method_name=None: list(enumerate(llms)))
    monkeypatch.setattr(m, "setup_method_logging", lambda *a, **kw: None)
    monkeypatch.setattr(m, "_synth_cache_dir", lambda *a, **kw: tmp_path / "synth")

    m.main(None)
    out = capsys.readouterr().out
    assert "[WARN]" not in out, f"main() swallowed a failure:\n{out}"
    # An offline build is flat in the workload size — the budget is spent before a
    # question is asked — so every Q carries the same epsilon and the same utility. The
    # record is filed at each of them anyway: that flatness is only visible if the rows
    # exist at every Q to be flat across, and a method present at one Q alone vanishes
    # from every panel drawn at another.
    payloads = _assert_grid(tmp_path, expected_cells=len(TARGETS) * len(WORKLOADS))
    by_target = defaultdict(list)
    for payload in payloads:
        by_target[payload["epsilon_target"]].append(payload)
    for target, group in by_target.items():
        assert {p["workload_queries"] for p in group} == set(WORKLOADS)
        assert len({p["epsilon_workload"] for p in group}) == 1, (
            f"target={target}: an offline build's epsilon moved with the workload size"
        )
        # The records are copies of one inference run, so the generated corpus and the
        # per-question results are stored once and the rest name that file. Four copies
        # of a synthetic corpus per (dataset, LLM, target) buy nothing.
        holders = [p for p in group if "synthetic_texts" in p or "results" in p]
        assert len(holders) == 1, f"target={target}: the payload is stored {len(holders)}x"
        for payload in group:
            if payload is not holders[0]:
                assert payload["shared_payload"].endswith(".json")




# ---------------------------------------------------------------------------
# The grids the configs actually ship: how many cells survive a real run
# ---------------------------------------------------------------------------

def _drive_shipped(monkeypatch, tmp_path, method):
    from types import SimpleNamespace as NS

    if method == "dp_rag":
        import experiments.run_dp_rag as m
        from dp_methods.per_query.dp_rag import DPAggregatedGenerator, DPRetriever, DPTopPRetriever

        def _tok(stub, kw):
            stub.epsilon, stub.delta = float(kw["epsilon"]), float(kw["delta"])
            stub.max_new_tokens = int(kw["max_new_tokens"])
            stub.token_epsilon = DPAggregatedGenerator._compute_token_epsilon(stub)

        _patch_common(monkeypatch, m, tmp_path)
        monkeypatch.setattr(m, "DPAggregatedGenerator",
                            _checked_ctor(DPAggregatedGenerator, after_init=_tok))
        monkeypatch.setattr(m, "DPRetriever", _checked_ctor(DPRetriever))
        monkeypatch.setattr(m, "DPTopPRetriever", _checked_ctor(DPTopPRetriever))
        monkeypatch.setattr(m, "RAGPipeline", lambda **kw: NS(
            index=lambda d: None, query=lambda q, k=5: "a"))
        monkeypatch.setattr(m, "_index_once", lambda *a, **kw: None)
        m._run_for_llm(_load_cfg("dp_rag", tmp_path, shipped_grid=True), "s/m", "s__m", False)

    elif method == "dpvoterag":
        import experiments.run_dpvoterag as m
        from dp_methods.per_query.dpvoterag.generator import VoteRAGGenerator
        _patch_common(monkeypatch, m, tmp_path)
        monkeypatch.setattr(m, "VoteRAGGenerator", _checked_ctor(VoteRAGGenerator))
        monkeypatch.setattr(m, "DPVoteRAGPipeline", lambda **kw: NS(
            index=lambda d: None, query=lambda q, k=5: "a"))
        m._run_for_llm(_load_cfg("dpvoterag", tmp_path, shipped_grid=True),
                       _args(), "s/m", "s__m", False)

    elif method == "dpsparsevoterag":
        import experiments.run_dpsparsevoterag as m
        from dp_methods.per_query.dpvoterag.sparse_vote_generator import SparseVoteRAGGenerator
        _patch_common(monkeypatch, m, tmp_path)
        monkeypatch.setattr(m, "SparseVoteRAGGenerator", _checked_ctor(SparseVoteRAGGenerator))
        monkeypatch.setattr(m, "DPVoteRAGPipeline", lambda **kw: NS(
            index=lambda d: None, query=lambda q, k=5: "a"))
        m._run_for_llm(_load_cfg("dpsparsevoterag", tmp_path, shipped_grid=True),
                       _args(), "s/m", "s__m", False)

    elif method == "dpksa":
        import experiments.run_dpksa as m
        from dp_methods.per_query.dpksa.generator import DPKSAGenerator
        _patch_common(monkeypatch, m, tmp_path)
        monkeypatch.setattr(m, "DPKSAGenerator",
                            _checked_ctor(DPKSAGenerator, after_init=_real_dpksa_calibration))
        monkeypatch.setattr(m, "DPKSAPipeline", lambda **kw: NS(
            index=lambda d: None, query=lambda q, k=5: "a"))
        m._run_for_llm(_load_cfg("dpksa", tmp_path, shipped_grid=True),
                       _args(), "s/m", "s__m", False)

    elif method == "private_rag":
        import experiments.run_private_rag as m
        from dp_methods.per_query.private_rag.generator import PrivateRAGGenerator
        _patch_common(monkeypatch, m, tmp_path)
        monkeypatch.setattr(m, "PrivateRAGGenerator", _checked_ctor(PrivateRAGGenerator))
        monkeypatch.setattr(m, "_make_retriever", lambda *a, **kw: StubRetriever())
        monkeypatch.setattr(m, "DPVoteRAGPipeline", lambda **kw: NS(
            index=lambda d: None, query=lambda q, k=5: "a"))
        m._run_for_llm(_load_cfg("private_rag", tmp_path, shipped_grid=True),
                       _args(), "s/m", "s__m", False)

    elif method == "dp_synrag":
        import experiments.run_dp_synrag as m
        _patch_common(monkeypatch, m, tmp_path)
        cfg = _load_cfg("dp_synrag", tmp_path, shipped_grid=True)
        cfg["llms"] = [{"name": "huggingface", "model_name": "s/m"}]
        cfg["llm"] = cfg["llms"][0]
        monkeypatch.setattr(m, "load_config", lambda p: cfg)
        monkeypatch.setattr(m, "_build_indexer", lambda *a, **kw: _StubSynIndexer())
        monkeypatch.setattr(m, "CosineSimilarityRetriever", StubRetriever)
        monkeypatch.setattr(m, "RAGPipeline", lambda **kw: NS(
            index=lambda d: None, query=lambda q, k=5: "a"))
        monkeypatch.setattr(m, "resolve_llm_execution",
                            lambda l, method_name=None: list(enumerate(l)))
        monkeypatch.setattr(m, "setup_method_logging", lambda *a, **kw: None)
        monkeypatch.setattr(m, "_synth_cache_dir", lambda *a, **kw: tmp_path / "synth")
        m.main(None)


#: Methods that file their one flat-in-Q stream under a single workload size instead of
#: at every one. Empty, and meant to stay that way. A method whose cost is flat in Q still runs ONE
#: stream, but it files that stream at every workload size the grid declares; a row
#: written at Q=max alone vanishes from every panel drawn at another Q.
_COLLAPSES_ITS_WORKLOAD_SWEEP: set[str] = set()


@pytest.mark.parametrize("method", [
    "dp_rag", "dpvoterag", "dpsparsevoterag", "dpksa", "private_rag", "dp_synrag",
])
def test_shipped_grid_is_not_mostly_skipped(monkeypatch, tmp_path, method):
    """Every cell of the grid the config ships must produce a result.

    A target the mechanism cannot reach is logged and `continue`d, so a config whose
    points are all out of reach runs for hours and writes nothing at all. The count is
    the check: 4 targets x 4 workloads = 16 cells.

    A method flat in the workload size still runs ONE stream — its budget covers the
    whole stream however long it is — but it files the result at every Q, because a row
    that exists only at one Q disappears from every panel drawn at another. Every method
    now does, which is why ``_COLLAPSES_ITS_WORKLOAD_SWEEP`` is empty.
    """
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "experiments/configs/full" / f"{method}.yaml").read_text())
    targets = (cfg.get("epsilon_targets")
               or cfg.get("synthetic_generation", {}).get("epsilon_targets"))
    workloads = cfg.get("workload_queries")

    collapsed = method in _COLLAPSES_ITS_WORKLOAD_SWEEP
    expected = len(targets) * (1 if collapsed else len(workloads))

    _drive_shipped(monkeypatch, tmp_path, method)

    files = _written(tmp_path)
    cells = {
        (pl.get("epsilon_target"), pl.get("workload_queries"))
        for pl in (json.loads(p.read_text()) for p in files)
    }
    print(f"\n  {method}: targets={targets} workloads={workloads} "
          f"-> {len(files)}/{expected} cells written")
    missing = {(t, q) for t in map(float, targets) for q in map(int, workloads)} - cells
    if missing:
        print(f"    MISSING: {sorted(missing)}")
    assert len(files) == expected, f"{len(files)}/{expected} cells; missing {sorted(missing)}"


def test_private_rag_ada_shipped_grid(monkeypatch, tmp_path):
    """MURAG-ADA drops a cell whenever eps_q < eps_thr + eps_0, so its grid is checked
    against its own config rather than assumed to match MURAG's."""
    from types import SimpleNamespace as NS

    import experiments.run_private_rag as m
    from dp_methods.per_query.private_rag.generator import PrivateRAGGenerator

    cfg = _load_cfg("private_rag_ada", tmp_path, shipped_grid=True)
    # MuRAG-ADA's budget is flat in the workload size, so one stream per target answers
    # for every Q and the shorter workloads are prefixes of it — but that one result is
    # filed at each workload size, so a per-Q comparison finds the method at every Q.
    expected = len(cfg["epsilon_targets"]) * len(cfg["workload_queries"])

    _patch_common(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "PrivateRAGGenerator", _checked_ctor(PrivateRAGGenerator))
    monkeypatch.setattr(m, "_make_retriever", lambda *a, **kw: StubRetriever())
    monkeypatch.setattr(m, "DPVoteRAGPipeline", lambda **kw: NS(
        index=lambda d: None, query=lambda q, k=5: "a"))
    m._run_for_llm(cfg, _args(), "s/m", "s__m", False)

    files = _written(tmp_path)
    print(f"\n  private_rag_ada: {len(files)}/{expected} cells written")
    for p in files:
        print(f"    {p.relative_to(tmp_path / 'out')}")
    assert len(files) == expected


# ---------------------------------------------------------------------------
# The derived per-token delta reaches the accountant and the mechanism alike
# ---------------------------------------------------------------------------

def _recording_ctor(real_cls, seen: list):
    """A checked stub that also records the kwargs each construction was given."""
    base = _checked_ctor(real_cls)

    class _Recorder(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            seen.append(kw)

    _Recorder.__name__ = f"Recording{real_cls.__name__}"
    return _Recorder


def _drive_vote_recording(monkeypatch, tmp_path, method):
    """Run a vote method's shipped grid, recording what each side was handed.

    Returns ``(generator_kwargs, budget_kwargs)`` in cell order, alongside the payloads
    the run wrote — the three places the per-token delta has to agree.
    """
    from types import SimpleNamespace as NS

    generator_kwargs: list[dict] = []
    budget_kwargs: list[dict] = []

    if method == "dpvoterag":
        import experiments.run_dpvoterag as m
        from dp_methods.per_query.dpvoterag.generator import VoteRAGGenerator as G

        gen_name = "VoteRAGGenerator"
    else:
        import experiments.run_dpsparsevoterag as m
        from dp_methods.per_query.dpvoterag.sparse_vote_generator import (
            SparseVoteRAGGenerator as G,
        )

        gen_name = "SparseVoteRAGGenerator"

    real_budget = m.budget_for_target

    def _spy_budget(*a, **kw):
        budget_kwargs.append(dict(kw))
        return real_budget(*a, **kw)

    _patch_common(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "budget_for_target", _spy_budget)
    monkeypatch.setattr(m, gen_name, _recording_ctor(G, generator_kwargs))
    monkeypatch.setattr(m, "DPVoteRAGPipeline", lambda **kw: NS(
        index=lambda d: None, query=lambda q, k=5: "a"))
    m._run_for_llm(_load_cfg(method, tmp_path, shipped_grid=True),
                   _args(), "s/m", "s__m", False)

    payloads = [json.loads(p.read_text()) for p in _written(tmp_path)]
    return generator_kwargs, budget_kwargs, payloads


@pytest.mark.parametrize("method", ["dpvoterag", "dpsparsevoterag"])
def test_the_derived_delta_reaches_the_accountant_and_the_generator(
    monkeypatch, tmp_path, method
):
    """One `delta_token` per cell, or the reported epsilon is simply false.

    The value is not only an accounting parameter: it also sets the Limited-Domain
    Gumbel's bottom-out threshold, so the generator releases tokens under whichever
    number it was handed. Derived on one side and read from the config on the other,
    the run would still finish and still write an epsilon — for a mechanism it never
    executed. This drives the real sweep and checks the three places the value
    surfaces: the budget inversion, the constructed generator, and the record.
    """
    from metrics.workload_accounting import DEFAULT_DELTA, MECHANISM_DELTA_SHARE

    gen_kwargs, budget_kwargs, payloads = _drive_vote_recording(
        monkeypatch, tmp_path, method
    )
    assert payloads, "the sweep wrote nothing"
    assert len(gen_kwargs) == len(payloads) == len(budget_kwargs)
    # The files come back in path order; the recorded calls in cell order. Both are
    # keyed by the cell so a mismatch is a disagreement and not a sorting artefact.
    by_cell = {(p["epsilon_target"], p["workload_queries"]): p for p in payloads}

    # The divisor is the number of *charged* releases, which is not the answer
    # length: DPSparseVoteRAG emits up to `tmax_absolute` tokens but only
    # `max_private_tokens` of them go through the Limited-Domain Gumbel and carry
    # additive delta — the rest come off the budget-free public path. The two
    # coincide in the shipped config and diverge in the paper-faithful one, so
    # reading `answer_tokens` here would pass while checking the wrong quantity.
    releases_key = {
        "dpvoterag": "max_length",
        "dpsparsevoterag": "max_private_tokens",
    }[method]
    releases = int(_load_cfg(method, tmp_path, shipped_grid=True)[method][releases_key])

    for gen, budget in zip(gen_kwargs, budget_kwargs):
        payload = by_cell[(float(budget["target_epsilon"]), int(budget["queries"]))]
        q = int(payload["workload_queries"])
        expected = MECHANISM_DELTA_SHARE * DEFAULT_DELTA / (q * releases)
        assert payload["delta_token"] == pytest.approx(expected), (
            f"the record's delta_token is not the workload's at Q={q}"
        )
        assert gen["delta_token"] == pytest.approx(payload["delta_token"]), (
            f"the generator ran at delta_token={gen['delta_token']} while the record "
            f"reports {payload['delta_token']} (Q={q})"
        )
        assert budget["delta_token"] == pytest.approx(payload["delta_token"]), (
            f"the budget was inverted at delta_token={budget['delta_token']} while "
            f"the record reports {payload['delta_token']} (Q={q})"
        )


def test_dpksas_derived_delta_reaches_the_accountant_and_the_generator(
    monkeypatch, tmp_path
):
    """One `mechanism_delta` per cell, or the reported epsilon is simply false.

    DP-KSA's is the same story as the vote methods' `delta_token`, through a different
    mechanism: the PTR test spends a share of it on every query, so deriving it from
    the workload is what keeps that spend constant across the grid — and it is also what
    `calibrate_dpksa` turns into the Gumbel scale, the PTR sigma and the release
    threshold. Inverted against one value and calibrated against another, the run would
    still finish and still write an epsilon, for a mechanism it never executed.
    """
    from types import SimpleNamespace as NS

    from dp_methods.per_query.dpksa.generator import DPKSAGenerator
    import experiments.run_dpksa as m
    from metrics.workload_accounting import DEFAULT_DELTA, MECHANISM_DELTA_SHARE

    generator_kwargs: list[dict] = []
    budget_kwargs: list[dict] = []
    real_budget = m.budget_for_target

    def _spy_budget(*a, **kw):
        budget_kwargs.append(dict(kw))
        return real_budget(*a, **kw)

    def _record_and_calibrate(stub, kw):
        # Recorded from inside the calibration hook rather than by wrapping the stub:
        # a second _checked_ctor around one that already calibrates would overwrite
        # `calibration` with the stub's constants and the cells would all read alike.
        generator_kwargs.append(dict(kw))
        _real_dpksa_calibration(stub, kw)

    _patch_common(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "budget_for_target", _spy_budget)
    monkeypatch.setattr(
        m, "DPKSAGenerator",
        _checked_ctor(DPKSAGenerator, after_init=_record_and_calibrate),
    )
    monkeypatch.setattr(m, "DPKSAPipeline", lambda **kw: NS(
        index=lambda d: None, query=lambda q, k=5: "a"))
    m._run_for_llm(_load_cfg("dpksa", tmp_path, shipped_grid=True),
                   _args(), "s/m", "s__m", False)

    payloads = [json.loads(p.read_text()) for p in _written(tmp_path)]
    assert payloads, "the sweep wrote nothing"
    assert len(generator_kwargs) == len(payloads) == len(budget_kwargs)
    by_cell = {(p["epsilon_target"], p["workload_queries"]): p for p in payloads}

    for gen, budget in zip(generator_kwargs, budget_kwargs):
        payload = by_cell[(float(budget["target_epsilon"]), int(budget["queries"]))]
        q = int(payload["workload_queries"])
        fraction = float(gen["ptr_delta_fraction"])
        expected = MECHANISM_DELTA_SHARE * DEFAULT_DELTA / (q * fraction)
        assert payload["delta"] == pytest.approx(expected), (
            f"the record's mechanism delta is not the workload's at Q={q}"
        )
        assert gen["target_delta"] == pytest.approx(payload["delta"]), (
            f"the generator calibrated at target_delta={gen['target_delta']} while "
            f"the record reports {payload['delta']} (Q={q})"
        )
        assert budget["mechanism_delta"] == pytest.approx(payload["delta"]), (
            f"the budget was inverted at mechanism_delta={budget['mechanism_delta']} "
            f"while the record reports {payload['delta']} (Q={q})"
        )
        # And the split reaching the two sides has to agree as well: it decides how much
        # of that delta a query actually spends, so the invariant above rests on it.
        assert budget["ptr_delta_fraction"] == pytest.approx(fraction)
        # What the invariant is for: the same additive mass in every cell.
        assert q * float(payload["calibration"]["delta_ptr"]) == pytest.approx(
            MECHANISM_DELTA_SHARE * DEFAULT_DELTA
        ), f"Q={q} spends a different additive mass from its neighbours"


@pytest.mark.parametrize("method", ["dpvoterag", "dpsparsevoterag"])
def test_every_cell_of_the_vote_grid_is_reported_even_where_it_cannot_release(
    monkeypatch, tmp_path, method
):
    """A mechanism that cannot serve a workload is a result, not a missing row.

    Over much of this grid the bottom-out symbol sits above the number of voters, so
    no histogram can clear it and every answer comes back empty. Such a cell has a
    perfectly good epsilon and must be run and written: dropped, a reader cannot tell
    an unservable workload from one nobody got round to running.
    """
    _, _, payloads = _drive_vote_recording(monkeypatch, tmp_path, method)

    cells = {(p["epsilon_target"], p["workload_queries"]) for p in payloads}
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "experiments/configs/full" / f"{method}.yaml").read_text())
    expected = {(float(t), int(q))
                for t in cfg["epsilon_targets"] for q in cfg["workload_queries"]}
    assert cells == expected, f"missing cells: {sorted(expected - cells)}"

    unservable = [p for p in payloads if not p["ld_feasible"]]
    assert unservable, "no cell is out of the mechanism's reach; the test is vacuous"
    for p in unservable:
        assert p.get("workload_error") is None, p["workload_error"]
        assert isinstance(p["epsilon_workload"], float)


def test_an_answer_the_mechanism_never_emitted_stays_visible_as_empty():
    """The scorer separates "did not answer" from "answered wrongly".

    An unservable cell returns an empty string for every question. Averaged into the
    overlap metrics that is indistinguishable from a wrong answer, so `answer_rate` is
    in the scorer's own metric set and is what makes the difference readable.
    """
    from experiments.score_results import UTILITY_METRICS
    from metrics.utility import compute_utility_metrics

    assert "answer_rate" in UTILITY_METRICS
    scored = compute_utility_metrics(["", "", ""], ["Paris", "Rome", "Berlin"],
                                     ["answer_rate"])
    assert scored["answer_rate"] == 0.0
