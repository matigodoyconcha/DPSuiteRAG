"""Unit tests for Private-RAG (MURAG / MURAG-ADA), arXiv:2511.07637."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dp_methods.per_query.private_rag.budget_manager import BudgetManager
from dp_methods.per_query.private_rag.generator import PrivateRAGGenerator
from metrics.privacy_accounting_private_rag import (
    method_epsilon,
    subroutine_epsilon,
    subroutine_max_releases,
    summarize,
)
from dp_methods.per_query.private_rag.retriever import (
    AdaptiveBudgetAwareRetriever,
    BudgetAwareRetriever,
)


class _FakeTokenizer:
    """Just enough tokenizer surface for the ExpoMech unit tests (no model)."""

    vocab_size = 50
    eos_token_id = 0
    chat_template = None


def _expo_gen(epsilon: float = 10.0, epsilon_0: float = 1.0) -> PrivateRAGGenerator:
    gen = PrivateRAGGenerator(
        model=None, tokenizer=_FakeTokenizer(), m=5, k=1,
        epsilon=epsilon, epsilon_0=epsilon_0,
    )
    gen._np_rng = np.random.default_rng(0)
    return gen


class _FakeIndexer:
    """Minimal indexer exposing the attributes the retrievers read.

    ``embeddings_matrix`` is a (N, 1) column so that a query of ``[1.0]`` yields
    scores equal to the column entries — deterministic control for unit tests.
    """

    def __init__(self, scores: list[float]):
        self.embeddings_matrix = np.array([[s] for s in scores], dtype=np.float32)
        self.metadata = [{"doc_id": i, "text": f"d{i}"} for i in range(len(scores))]


_UNIT_QUERY = np.array([1.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# BudgetManager — individual pure-DP filter (E(z) = M * eps_q)
# ---------------------------------------------------------------------------

def test_budget_manager_starts_full() -> None:
    bm = BudgetManager(initial_budget=5.0)
    assert bm.remaining("doc0") == 5.0        # unseen -> full budget
    assert bm.can_query("doc0", cost=5.0)     # exactly enough
    assert not bm.can_query("doc0", cost=5.1)  # not quite


def test_budget_manager_consume_and_exhaust() -> None:
    bm = BudgetManager(initial_budget=6.0)
    bm.consume("z", 2.0)
    assert bm.remaining("z") == pytest.approx(4.0)
    assert bm.can_query("z", 4.0)
    bm.consume("z", 2.0)
    bm.consume("z", 2.0)
    assert bm.remaining("z") == pytest.approx(0.0)
    assert not bm.can_query("z", 2.0)  # exhausted -> dropped from active set


def test_budget_manager_floors_at_zero() -> None:
    bm = BudgetManager(initial_budget=1.0)
    bm.consume("z", 5.0)  # would go negative; floored
    assert bm.remaining("z") == 0.0


def test_budget_manager_per_document_independence() -> None:
    bm = BudgetManager(initial_budget=10.0)
    bm.consume("a", 10.0)          # exhaust a
    assert not bm.can_query("a", 1.0)
    assert bm.can_query("b", 10.0)  # b untouched


def test_budget_manager_active_ids_filters_by_cost() -> None:
    bm = BudgetManager(initial_budget=4.0)
    bm.consume("a", 4.0)   # exhausted
    bm.consume("b", 2.0)   # 2 left
    ids = ["a", "b", "c"]  # c unseen (full)
    assert bm.active_ids(cost=2.0, candidate_ids=ids) == ["b", "c"]
    assert bm.active_ids(cost=4.0, candidate_ids=ids) == ["c"]


def test_budget_manager_reset() -> None:
    bm = BudgetManager(initial_budget=3.0)
    bm.consume("a", 3.0)
    bm.reset()
    assert bm.remaining("a") == 3.0
    assert bm.can_query("a", 3.0)


def test_budget_manager_stats() -> None:
    bm = BudgetManager(initial_budget=5.0)
    bm.consume_many(["a", "b"], 5.0)  # both exhausted
    bm.consume("c", 1.0)
    s = bm.stats(cost=5.0)
    assert s["n_tracked"] == 3
    assert s["n_exhausted"] == 3  # a, b exhausted; c cannot pay 5 (has 4)
    assert s["total_consumed"] == pytest.approx(11.0)


def test_budget_manager_rejects_nonpositive_budget() -> None:
    with pytest.raises(ValueError):
        BudgetManager(initial_budget=0.0)


# ---------------------------------------------------------------------------
# BudgetAwareRetriever (MURAG) — fixed public threshold
# ---------------------------------------------------------------------------

def test_murag_returns_top_k_above_threshold() -> None:
    idx = _FakeIndexer([0.9, 0.8, 0.5, 0.1, -0.2])
    bm = BudgetManager(initial_budget=10.0)
    r = BudgetAwareRetriever(indexer=idx, budget_manager=bm, epsilon_q=2.0, threshold=0.3)
    out = r.retrieve(_UNIT_QUERY, k=2)
    assert [d["text"] for d in out] == ["d0", "d1"]          # top-2 by score
    assert all(out[i]["score"] >= out[i + 1]["score"] for i in range(len(out) - 1))


def test_score_diagnostics_summary() -> None:
    from dp_methods.per_query.private_rag.retriever import _score_diagnostics

    d = _score_diagnostics(np.array([0.1, 0.2, 0.55, 0.7, 0.9, 0.3, 0.45, 0.62]))
    assert d["score_max"] == 0.9
    assert d["n_above"]["0.5"] == 4          # 0.55, 0.62, 0.7, 0.9 (strict >)
    assert d["n_above"]["0.7"] == 1          # only 0.9
    assert _score_diagnostics(np.array([])) == {}


def test_murag_reports_score_diag() -> None:
    idx = _FakeIndexer([0.9, 0.8, 0.5, 0.1, -0.2])
    bm = BudgetManager(initial_budget=10.0)
    r = BudgetAwareRetriever(indexer=idx, budget_manager=bm, epsilon_q=2.0, threshold=0.3)
    r.retrieve(_UNIT_QUERY, k=2)
    diag = r.last_run_stats["score_diag"]
    assert diag["score_max"] == pytest.approx(0.9)
    assert diag["n_above"]["0.5"] == 2       # 0.9, 0.8 (0.5 is not > 0.5)


def test_murag_charges_all_above_threshold_not_just_top_k() -> None:
    # Docs 0,1,2 are above tau=0.3; k=2 returns only 0,1 but ALL THREE are charged
    # (paper: charge D_qt before Top-K). Docs 3,4 (below tau) are untouched.
    idx = _FakeIndexer([0.9, 0.8, 0.5, 0.1, -0.2])
    bm = BudgetManager(initial_budget=10.0)
    r = BudgetAwareRetriever(indexer=idx, budget_manager=bm, epsilon_q=2.0, threshold=0.3)
    r.retrieve(_UNIT_QUERY, k=2)
    assert bm.remaining(0) == pytest.approx(8.0)
    assert bm.remaining(1) == pytest.approx(8.0)
    assert bm.remaining(2) == pytest.approx(8.0)  # charged even though not returned
    assert bm.remaining(3) == 10.0                # below threshold: free
    assert bm.remaining(4) == 10.0
    assert r.last_run_stats["n_charged"] == 3
    assert r.last_run_stats["n_returned"] == 2


def test_murag_depletes_budget_and_drops_documents() -> None:
    idx = _FakeIndexer([0.9, 0.8, 0.5, 0.1, -0.2])
    bm = BudgetManager(initial_budget=6.0)  # allows 3 participations at eps_q=2
    r = BudgetAwareRetriever(indexer=idx, budget_manager=bm, epsilon_q=2.0, threshold=0.3)
    for _ in range(3):
        assert r.retrieve(_UNIT_QUERY, k=2)      # non-empty while budget lasts
    # Docs 0,1,2 now exhausted; nothing above threshold remains active.
    assert r.retrieve(_UNIT_QUERY, k=2) == []


def test_murag_is_deterministic() -> None:
    idx = _FakeIndexer([0.9, 0.8, 0.5, 0.1])
    out_a = BudgetAwareRetriever(
        indexer=idx, budget_manager=BudgetManager(10.0), epsilon_q=1.0, threshold=0.3
    ).retrieve(_UNIT_QUERY, k=2)
    out_b = BudgetAwareRetriever(
        indexer=idx, budget_manager=BudgetManager(10.0), epsilon_q=1.0, threshold=0.3
    ).retrieve(_UNIT_QUERY, k=2)
    assert [d["text"] for d in out_a] == [d["text"] for d in out_b]


def test_murag_empty_when_all_below_threshold() -> None:
    idx = _FakeIndexer([0.1, 0.05, -0.2])
    bm = BudgetManager(initial_budget=10.0)
    r = BudgetAwareRetriever(indexer=idx, budget_manager=bm, epsilon_q=1.0, threshold=0.5)
    assert r.retrieve(_UNIT_QUERY, k=3) == []
    assert bm.stats()["total_consumed"] == 0.0  # nothing charged


# ---------------------------------------------------------------------------
# AdaptiveBudgetAwareRetriever (MURAG-ADA) — noisy prefix-sum threshold
# ---------------------------------------------------------------------------

# Scores interior to 0.1-wide bins (avoids float bin-boundary ambiguity):
#   bin 0 [>0.9]: d0(0.95), d1(0.93)   bin 1 (0.8,0.9]: d2(0.85)
#   bin 4: d3(0.55)   bin 5: d4(0.45)   bin 8: d5(0.15)
_ADA_SCORES = [0.95, 0.93, 0.85, 0.55, 0.45, 0.15]


def _ada_retriever(bm: BudgetManager, **kw) -> AdaptiveBudgetAwareRetriever:
    # Large eps_thr -> Laplace scale 1/eps_thr is tiny -> prefix sum ~ true counts.
    params = dict(
        indexer=_FakeIndexer(_ADA_SCORES),
        budget_manager=bm,
        epsilon_q=200.0,
        epsilon_thr=100.0,
        score_min=0.0,
        score_max=1.0,
        bin_size=0.1,
    )
    params.update(kw)
    return AdaptiveBudgetAwareRetriever(**params)


def test_murag_ada_stops_when_prefix_sum_reaches_k() -> None:
    bm = BudgetManager(initial_budget=1000.0)
    r = _ada_retriever(bm)
    out = r.retrieve(_UNIT_QUERY, k=2)               # bin 0 alone has 2 docs -> stop
    assert [d["text"] for d in out] == ["d0", "d1"]
    assert r.last_run_stats["n_bins_processed"] == 1
    assert r.last_run_stats["tau_t"] == pytest.approx(0.9)  # lower edge of bin 0


def test_murag_ada_processes_more_bins_for_larger_k() -> None:
    bm = BudgetManager(initial_budget=1000.0)
    r = _ada_retriever(bm)
    out = r.retrieve(_UNIT_QUERY, k=3)               # bin0(2)+bin1(1)=3 -> stop after bin1
    assert [d["text"] for d in out] == ["d0", "d1", "d2"]
    assert r.last_run_stats["n_bins_processed"] == 2
    assert r.last_run_stats["tau_t"] == pytest.approx(0.8)


def test_murag_ada_charges_eps_thr_plus_eps_rag() -> None:
    # A retrieved doc pays eps_thr (its processed bin) + eps_RAG (active set) = eps_q.
    # A doc in an unprocessed (lower) bin pays nothing.
    bm = BudgetManager(initial_budget=1000.0)
    r = _ada_retriever(bm)
    r.retrieve(_UNIT_QUERY, k=2)                      # processes bin 0 only
    assert bm.remaining(0) == pytest.approx(1000.0 - 200.0)  # eps_thr+eps_RAG
    assert bm.remaining(1) == pytest.approx(1000.0 - 200.0)
    assert bm.remaining(2) == 1000.0                 # bin 1 never processed


def test_murag_ada_is_reproducible() -> None:
    a = _ada_retriever(BudgetManager(1000.0)).retrieve(_UNIT_QUERY, k=2)
    b = _ada_retriever(BudgetManager(1000.0)).retrieve(_UNIT_QUERY, k=2)
    assert [d["text"] for d in a] == [d["text"] for d in b]


def test_murag_ada_rejects_eps_thr_ge_eps_q() -> None:
    with pytest.raises(ValueError):
        AdaptiveBudgetAwareRetriever(
            indexer=_FakeIndexer(_ADA_SCORES),
            budget_manager=BudgetManager(10.0),
            epsilon_q=1.0,
            epsilon_thr=1.0,  # not < eps_q
        )


# ---------------------------------------------------------------------------
# PrivateRAGGenerator — pure-DP Algorithm 7 (ExpoMech + SVT + release cap)
# ---------------------------------------------------------------------------

def test_release_cap_is_floor_eps_over_eps0() -> None:
    assert _expo_gen(epsilon=10.0, epsilon_0=1.0).max_releases == 10
    assert _expo_gen(epsilon=10.0, epsilon_0=2.0).max_releases == 5
    assert _expo_gen(epsilon=3.0, epsilon_0=2.0).max_releases == 1   # floor(1.5)
    assert _expo_gen(epsilon=8.0, epsilon_0=2.0).max_releases == 4


def test_vote_threshold_default_is_half_the_voters() -> None:
    # theta compares against the agreement count s over the m voters (0..m), so the
    # default lives on the voter scale = m/2, independent of k (NOT m*k/2).
    assert _expo_gen().vote_threshold == pytest.approx(2.5)  # m=5 -> 2.5
    gen = PrivateRAGGenerator(
        model=None, tokenizer=_FakeTokenizer(), m=8, k=3, epsilon=10.0, epsilon_0=2.0
    )
    assert gen.vote_threshold == 4.0  # m/2, not m*k/2 (=12)


def test_expo_mech_returns_valid_tokens() -> None:
    gen = _expo_gen()
    for _ in range(50):
        t = gen._expo_mech({7: 3, 8: 1})
        assert 0 <= t < _FakeTokenizer.vocab_size


def test_expo_mech_empty_histogram_is_uniform_draw() -> None:
    gen = _expo_gen()
    t = gen._expo_mech({})
    assert 0 <= t < _FakeTokenizer.vocab_size


def test_expo_mech_concentrates_at_high_epsilon() -> None:
    # Large ε_Expo (=ε_0/2=50): the single voted token dominates the full-vocab draw.
    gen = _expo_gen(epsilon=100.0, epsilon_0=100.0)
    picks = [gen._expo_mech({7: 4}) for _ in range(200)]
    assert picks.count(7) >= 195


def test_expo_mech_prefers_higher_count() -> None:
    # With separable weights, the higher-count token wins far more often.
    gen = _expo_gen(epsilon=100.0, epsilon_0=100.0)
    picks = [gen._expo_mech({7: 5, 8: 4}) for _ in range(200)]
    assert picks.count(7) > picks.count(8)


def test_expo_mech_weight_ratio_matches_exponential_mechanism() -> None:
    # P(a)/P(b) must equal exp(ε_Expo*(u_a-u_b)/(2Δu)) for two observed tokens.
    gen = _expo_gen(epsilon=4.0, epsilon_0=2.0)  # ε_Expo = 1.0, Δu = 1
    picks = np.array([gen._expo_mech({1: 3, 2: 1}) for _ in range(20000)])
    n1, n2 = int((picks == 1).sum()), int((picks == 2).sum())
    expected = np.exp(1.0 * (3 - 1) / 2.0)  # = e ≈ 2.718
    assert n1 + n2 > 0
    assert (n1 / n2) == pytest.approx(expected, rel=0.15)


def test_generator_end_to_end() -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    except Exception as exc:  # pragma: no cover - network/model availability
        pytest.skip(f"Could not load tiny-gpt2: {exc}")

    gen = PrivateRAGGenerator(
        model=model, tokenizer=tokenizer, m=3, k=1,
        epsilon=8.0, epsilon_0=2.0, t_max=6, random_seed=42,
    )
    docs = [f"Paris is the capital of France. Note {i}." for i in range(3)]
    out = gen.generate_vote("What is the capital of France?", docs)
    assert isinstance(out, str)
    stats = gen.last_run_stats
    assert stats["max_releases"] == 4
    assert stats["n_private_tokens"] <= stats["max_releases"]  # release cap respected
    assert stats["tokens_generated"] == stats["n_private_tokens"] + stats["n_baseline_tokens"]
    assert stats["exit_status"] in (0, 1, 2)


def test_generator_handles_empty_docs() -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Could not load tiny-gpt2: {exc}")

    gen = PrivateRAGGenerator(
        model=model, tokenizer=tokenizer, m=3, k=1,
        epsilon=8.0, epsilon_0=2.0, t_max=4, random_seed=42,
    )
    out = gen.generate_vote("What is the capital of France?", [])  # no retrieved docs
    assert isinstance(out, str)
    assert gen.last_run_stats["n_private_tokens"] == 0  # nothing to vote on -> baseline only


def test_partition_docs_pads_to_m_with_empty_chunks() -> None:
    # Algorithm 7 splits D_x into a *fixed* number of m chunks; with fewer than m*k
    # docs available, the surplus chunks are empty rather than shrinking the voter
    # count. len(docs)=6, m=50, k=1 -> 50 partitions, 6 non-empty, 44 empty.
    gen = PrivateRAGGenerator(
        model=None, tokenizer=_FakeTokenizer(), m=50, k=1, epsilon=10.0, epsilon_0=1.0,
    )
    docs = [f"doc {i}" for i in range(6)]
    partitions = gen._partition_docs(docs)
    assert len(partitions) == 50
    assert sum(1 for p in partitions if p) == 6
    assert sum(1 for p in partitions if not p) == 44


def test_generator_always_uses_m_voters_and_prefers_baseline_on_agreement() -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    except Exception as exc:  # pragma: no cover - network/model availability
        pytest.skip(f"Could not load tiny-gpt2: {exc}")

    gen = PrivateRAGGenerator(
        model=model, tokenizer=tokenizer, m=50, k=1,
        epsilon=10.0, epsilon_0=1.0, t_max=4, random_seed=42,
    )
    docs = [f"Paris is the capital of France. Note {i}." for i in range(6)]
    out = gen.generate_vote("What is the capital of France?", docs)
    assert isinstance(out, str)

    stats = gen.last_run_stats
    # ALWAYS m=50 voters, never shrunk to len(docs)//k (=6); 44 of them are empty-chunk
    # voters whose prompt is byte-identical to the baseline prompt at every step.
    assert stats["n_voters"] == 50
    assert stats["n_nonempty_voters"] == 6
    # Those 44 baseline-identical voters are guaranteed to agree with the baseline
    # token, so the agreement score s stays far above theta = m/2 = 25 and the SVT
    # branch keeps the free baseline token at every step (no private ExpoMech release).
    assert stats["n_private_tokens"] == 0
    assert stats["tokens_generated"] == stats["n_baseline_tokens"]


# ---------------------------------------------------------------------------
# Privacy accounting — headline epsilon = M * eps_q (pure DP, delta = 0)
# ---------------------------------------------------------------------------

def test_method_epsilon_is_M_times_eps_q() -> None:
    assert method_epsilon(1, 10.0) == 10.0
    assert method_epsilon(5, 2.0) == 10.0
    with pytest.raises(ValueError):
        method_epsilon(0, 10.0)


def test_subroutine_release_cap_and_spend() -> None:
    assert subroutine_max_releases(10.0, 2.0) == 5
    assert subroutine_max_releases(3.0, 2.0) == 1
    assert subroutine_epsilon(10.0, 2.0) == 10.0
    assert subroutine_epsilon(3.0, 2.0) == 2.0        # floor(1.5)*2, <= 3
    assert subroutine_epsilon(3.0, 2.0) <= 3.0


def test_summarize_murag() -> None:
    s = summarize(variant="murag", m_retrievals=1, epsilon_q=10.0, epsilon_0=2.0)
    assert s["epsilon"] == 10.0
    assert s["delta"] == 0.0
    assert s["max_releases"] == 5


def test_summarize_murag_ada_splits_budget() -> None:
    s = summarize(
        variant="murag_ada", m_retrievals=1, epsilon_q=10.0, epsilon_0=2.0, epsilon_thr=1.0
    )
    assert s["epsilon"] == 10.0          # headline unchanged
    assert s["epsilon_thr"] == 1.0
    assert s["epsilon_rag"] == 9.0       # eps_q - eps_thr
    assert s["max_releases"] == 4        # floor(9/2)
    assert s["delta"] == 0.0


def test_summarize_murag_ada_requires_eps_thr() -> None:
    with pytest.raises(ValueError):
        summarize(variant="murag_ada", m_retrievals=1, epsilon_q=10.0, epsilon_0=2.0)
    with pytest.raises(ValueError):
        summarize(variant="bogus", m_retrievals=1, epsilon_q=10.0, epsilon_0=2.0)


# ---------------------------------------------------------------------------
# Integration: budget-aware retriever + generator via DPVoteRAGPipeline
# ---------------------------------------------------------------------------

_DOCS = [
    "Differential privacy protects sensitive data by adding calibrated noise.",
    "FAISS is an efficient library for vector similarity search.",
    "A RAG pipeline combines document retrieval and text generation.",
    "Paris is the capital of France and a major European city.",
]
_QUERY = "What does differential privacy protect?"


def _built_pipeline_parts():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vanilla.chunker import FixedSizeChunker
    from vanilla.embedder import HFEmbedder
    from vanilla.indexer import FAISSIndexer
    from vanilla.postprocessor import IdentityPostprocessor
    from vanilla.prompt_builder import SimplePromptBuilder
    from vanilla.reranker import IdentityReranker

    embedder = HFEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")

    indexer = FAISSIndexer()
    vectors = embedder.embed(_DOCS)
    metadata = [{"doc_id": i, "chunk_id": 0, "text": d} for i, d in enumerate(_DOCS)]
    indexer.build(vectors, metadata)
    return {
        "chunker": FixedSizeChunker(chunk_size=200, overlap=20),
        "embedder": embedder,
        "indexer": indexer,
        "reranker": IdentityReranker(),
        "prompt_builder": SimplePromptBuilder(),
        "postprocessor": IdentityPostprocessor(),
        "model": model,
        "tokenizer": tokenizer,
    }


# ---------------------------------------------------------------------------
# Runner: per-token budget grid (paper Appendix E) — eps_0 swept per eps_q point
# ---------------------------------------------------------------------------

from experiments.run_private_rag import (  # noqa: E402
    _ada_cell_skipped,
    _cell_output_filename,
    _epsilon_0_grid,
)


def test_epsilon_0_grid_reads_top_level_values() -> None:
    grid = _epsilon_0_grid({"epsilon_0_values": [0.5, 1, 2]}, fallback=2.0)
    assert grid == [0.5, 1.0, 2.0]
    assert all(isinstance(v, float) for v in grid)


def test_epsilon_0_grid_falls_back_to_single_value() -> None:
    # Absent epsilon_0_values -> single-cell grid = [private_rag.epsilon_0], preserving
    # the pre-grid behavior for old configs.
    assert _epsilon_0_grid({}, fallback=2.0) == [2.0]


def test_cell_output_filename_single_vs_multi_eps0() -> None:
    # One eps_0: bare eps_q filename (backward compatible with existing plot scripts).
    assert _cell_output_filename(5.0, 2.0, multi_eps0=False) == "epsilon_q_5.0.json"
    # Multiple eps_0: one file per (eps_q, eps_0) cell.
    assert _cell_output_filename(5.0, 0.5, multi_eps0=True) == "epsilon_q_5.0_eps0_0.5.json"


def test_multi_value_grid_produces_nested_cells() -> None:
    # MURAG has no per-cell skip: every (eps_q, eps_0) pair is a run.
    eps_q_values = [5.0, 10.0]
    grid = _epsilon_0_grid({"epsilon_0_values": [0.5, 1.0, 2.0]}, fallback=2.0)
    cells = [
        _cell_output_filename(q, e0, multi_eps0=len(grid) > 1)
        for q in eps_q_values
        for e0 in grid
        if not _ada_cell_skipped("murag", q, e0, epsilon_thr=1.0)
    ]
    assert cells == [
        "epsilon_q_5.0_eps0_0.5.json",
        "epsilon_q_5.0_eps0_1.0.json",
        "epsilon_q_5.0_eps0_2.0.json",
        "epsilon_q_10.0_eps0_0.5.json",
        "epsilon_q_10.0_eps0_1.0.json",
        "epsilon_q_10.0_eps0_2.0.json",
    ]


def test_ada_skip_is_evaluated_per_cell() -> None:
    # eps_thr=1.0, strict eps_q < eps_thr+eps_0: at eps_q=2 the cells eps_0 in {0.5, 1.0}
    # survive (2 >= 1+0.5, 2 >= 1+1) and only eps_0=2.0 skips (2 < 1+2).
    grid = _epsilon_0_grid({"epsilon_0_values": [0.5, 1.0, 2.0]}, fallback=2.0)
    kept = [e0 for e0 in grid if not _ada_cell_skipped("murag_ada", 2.0, e0, epsilon_thr=1.0)]
    assert kept == [0.5, 1.0]
    # eps_q=5 clears the whole grid (5 >= 1+2).
    kept5 = [e0 for e0 in grid if not _ada_cell_skipped("murag_ada", 5.0, e0, epsilon_thr=1.0)]
    assert kept5 == [0.5, 1.0, 2.0]


def test_single_value_grid_keeps_bare_filename_and_no_skip() -> None:
    grid = _epsilon_0_grid({}, fallback=2.0)  # fallback -> single cell
    assert grid == [2.0]
    assert _cell_output_filename(10.0, grid[0], multi_eps0=len(grid) > 1) == "epsilon_q_10.0.json"


@pytest.mark.parametrize("variant", ["murag", "murag_ada"])
def test_pipeline_end_to_end(variant: str) -> None:
    try:
        parts = _built_pipeline_parts()
    except Exception as exc:  # pragma: no cover - network/model availability
        pytest.skip(f"Could not load models: {exc}")

    from dp_methods.per_query.dpvoterag.pipeline import DPVoteRAGPipeline

    bm = BudgetManager(initial_budget=10.0)
    if variant == "murag":
        retriever = BudgetAwareRetriever(
            indexer=parts["indexer"], budget_manager=bm, epsilon_q=10.0, threshold=0.0
        )
    else:
        retriever = AdaptiveBudgetAwareRetriever(
            indexer=parts["indexer"], budget_manager=bm,
            epsilon_q=10.0, epsilon_thr=1.0, score_min=0.0, score_max=1.0, bin_size=0.1,
        )
    generator = PrivateRAGGenerator(
        model=parts["model"], tokenizer=parts["tokenizer"], m=3, k=1,
        epsilon=10.0, epsilon_0=2.0, t_max=4, random_seed=42,
    )
    pipeline = DPVoteRAGPipeline(
        chunker=parts["chunker"], embedder=parts["embedder"], indexer=parts["indexer"],
        retriever=retriever, reranker=parts["reranker"],
        prompt_builder=parts["prompt_builder"], generator=generator,
        postprocessor=parts["postprocessor"],
    )
    answer = pipeline.query(_QUERY)  # retrieve_k defaults to generator.m * generator.k
    assert isinstance(answer, str)
    # A second query keeps depleting the same budget without error (multi-query).
    answer2 = pipeline.query("What is FAISS used for?")
    assert isinstance(answer2, str)
    assert bm.stats()["total_consumed"] > 0.0
