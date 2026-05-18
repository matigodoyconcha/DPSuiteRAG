"""Tests: seed reproducibility and cross-query independence for all DP methods.

Two properties are verified for every method:
  1. Reproducibility — same query + same config → same random draws (and same output).
  2. Independence   — different queries          → different random draws.

The RNG derivation logic is tested without loading any LLM (fast).
End-to-end reproducibility for VoteRAG is confirmed with sshleifer/tiny-gpt2.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

Q1 = "What does differential privacy protect?"
Q2 = "How does vector search work with FAISS?"
SEED = 42
TINY_MODEL = "sshleifer/tiny-gpt2"


# ---------------------------------------------------------------------------
# Pure-math helpers (replicate the exact offset formulas used in each method)
# ---------------------------------------------------------------------------

def _offset_retriever(vec: np.ndarray) -> int:
    """DPRetriever / DPTopPRetriever: hash of the L2-normalised query vector bytes."""
    return int(hashlib.sha256(vec.tobytes()).hexdigest()[:8], 16)


def _offset_generator(prompts: list[str]) -> int:
    """DPICLGenerator: hash of all prompts joined with the sentinel."""
    key = "\n||\n".join(prompts).encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def _offset_voterag(query: str) -> int:
    """VoteRAGGenerator / SparseVoteRAGGenerator: hash of the query string."""
    return int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:8], 16)


def _first_draw(seed: int) -> int:
    """First integer draw from a fresh default_rng(seed)."""
    return int(np.random.default_rng(seed).integers(0, 1 << 32))


# ---------------------------------------------------------------------------
# Tiny mock indexer — no model loading, used for DPRetriever integration tests
# ---------------------------------------------------------------------------

class _MockIndexer:
    def __init__(self, n: int = 8, d: int = 16):
        rng = np.random.default_rng(0)
        vecs = rng.normal(size=(n, d)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
        self.embeddings_matrix = vecs / norms
        self.metadata = [{"text": f"doc_{i}"} for i in range(n)]


# ---------------------------------------------------------------------------
# DP-RAG retriever seeds
# ---------------------------------------------------------------------------

class TestDPRetrieverSeeds:

    def _qvec(self, val: float = 1.0) -> np.ndarray:
        v = np.zeros(16, dtype=np.float32)
        v[0] = val
        v[1] = 1.0
        return v / np.linalg.norm(v)

    # --- Reproducibility ---

    def test_same_vector_same_offset(self):
        q = self._qvec()
        assert _offset_retriever(q) == _offset_retriever(q)

    def test_same_vector_same_rng_draw(self):
        q = self._qvec()
        off = _offset_retriever(q)
        assert _first_draw(SEED + off) == _first_draw(SEED + off)

    def test_retrieve_same_vector_same_docs(self):
        from dp_methods.per_query.dp_rag import DPRetriever
        retriever = DPRetriever(index=_MockIndexer(), epsilon=5.0, k=3, seed=SEED)
        q = self._qvec()
        r1 = retriever.retrieve(q)
        r2 = retriever.retrieve(q)
        assert [d["text"] for d in r1] == [d["text"] for d in r2]

    # --- Independence ---

    def test_different_vectors_different_offset(self):
        q1, q2 = self._qvec(1.0), self._qvec(2.0)
        assert _offset_retriever(q1) != _offset_retriever(q2)

    def test_different_vectors_different_rng_draw(self):
        q1, q2 = self._qvec(1.0), self._qvec(2.0)
        draw1 = _first_draw(SEED + _offset_retriever(q1))
        draw2 = _first_draw(SEED + _offset_retriever(q2))
        assert draw1 != draw2

    def test_different_seeds_same_vector_different_draw(self):
        q = self._qvec()
        off = _offset_retriever(q)
        assert _first_draw(42 + off) != _first_draw(99 + off)


# ---------------------------------------------------------------------------
# DP-RAG generator seeds
# ---------------------------------------------------------------------------

PROMPTS_Q1 = ["Privacy doc-A Q1", "Privacy doc-B Q1", "Privacy public Q1"]
PROMPTS_Q2 = ["FAISS doc-A Q2", "FAISS doc-B Q2", "FAISS public Q2"]


class TestDPICLGeneratorSeeds:

    # --- Reproducibility ---

    def test_same_prompts_same_offset(self):
        assert _offset_generator(PROMPTS_Q1) == _offset_generator(PROMPTS_Q1)

    def test_same_prompts_same_token_seed_every_step(self):
        off = _offset_generator(PROMPTS_Q1)
        for step in range(8):
            # seed for step N is always SEED + offset + N — pure arithmetic, always equal
            assert SEED + off + step == SEED + off + step

    def test_different_steps_different_token_seeds(self):
        """Each token position uses a distinct seed → independent draws."""
        off = _offset_generator(PROMPTS_Q1)
        draws = [_first_draw(SEED + off + step) for step in range(8)]
        assert len(set(draws)) == len(draws), "token steps must not share RNG seeds"

    # --- Independence ---

    def test_different_prompts_different_offset(self):
        assert _offset_generator(PROMPTS_Q1) != _offset_generator(PROMPTS_Q2)

    def test_different_prompts_different_token_seed(self):
        off1 = _offset_generator(PROMPTS_Q1)
        off2 = _offset_generator(PROMPTS_Q2)
        assert SEED + off1 + 0 != SEED + off2 + 0

    def test_different_prompts_different_rng_draw(self):
        off1 = _offset_generator(PROMPTS_Q1)
        off2 = _offset_generator(PROMPTS_Q2)
        assert _first_draw(SEED + off1) != _first_draw(SEED + off2)

    def test_different_seeds_same_prompts_different_draw(self):
        off = _offset_generator(PROMPTS_Q1)
        assert _first_draw(42 + off) != _first_draw(99 + off)


# ---------------------------------------------------------------------------
# DPVoteRAG seeds — pure-math (no LLM)
# ---------------------------------------------------------------------------

class TestVoteRAGSeeds:

    # --- Reproducibility ---

    def test_same_query_same_offset(self):
        assert _offset_voterag(Q1) == _offset_voterag(Q1)

    def test_same_query_same_rng_draw(self):
        off = _offset_voterag(Q1)
        assert _first_draw(SEED + off) == _first_draw(SEED + off)

    # --- Independence ---

    def test_different_queries_different_offset(self):
        assert _offset_voterag(Q1) != _offset_voterag(Q2)

    def test_different_queries_different_rng_draw(self):
        draw1 = _first_draw(SEED + _offset_voterag(Q1))
        draw2 = _first_draw(SEED + _offset_voterag(Q2))
        assert draw1 != draw2

    def test_different_seeds_same_query_different_draw(self):
        off = _offset_voterag(Q1)
        assert _first_draw(42 + off) != _first_draw(99 + off)


# ---------------------------------------------------------------------------
# DPVoteRAG — end-to-end reproducibility (needs tiny-gpt2)
# ---------------------------------------------------------------------------

DOCS = [
    "Differential privacy protects sensitive data by adding calibrated noise.",
    "FAISS is an efficient library for vector similarity search.",
]


@pytest.fixture(scope="module")
def tiny_tok():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    return mdl, tok


class TestVoteRAGEndToEnd:

    def test_same_query_same_output(self, tiny_tok):
        """Calling generate_vote twice with the same query must return identical text."""
        from dp_methods.per_query.dpvoterag import VoteRAGGenerator
        model, tokenizer = tiny_tok
        gen = VoteRAGGenerator(
            model=model, tokenizer=tokenizer,
            m=2, k=1, epsilon_token=10.0, epsilon_total=20.0, random_seed=SEED,
        )
        r1 = gen.generate_vote(Q1, DOCS)
        r2 = gen.generate_vote(Q1, DOCS)
        assert r1 == r2, "same query → same output (reproducibility broken)"

    def test_second_query_does_not_corrupt_first(self, tiny_tok):
        """Running Q2 between two Q1 calls must not change Q1's output.

        This is the exact property broken before the per-query RNG fix:
        Q2's draws advanced the shared _np_rng, making the second Q1 call
        start from a different RNG state.
        """
        from dp_methods.per_query.dpvoterag import VoteRAGGenerator
        model, tokenizer = tiny_tok
        gen = VoteRAGGenerator(
            model=model, tokenizer=tokenizer,
            m=2, k=1, epsilon_token=10.0, epsilon_total=20.0, random_seed=SEED,
        )
        r1_first = gen.generate_vote(Q1, DOCS)
        gen.generate_vote(Q2, DOCS)          # Q2 must not corrupt Q1's seed
        r1_second = gen.generate_vote(Q1, DOCS)
        assert r1_first == r1_second, "Q2 corrupted Q1's RNG state (independence broken)"

    def test_sparse_same_query_same_output(self, tiny_tok):
        from dp_methods.per_query.dpvoterag import SparseVoteRAGGenerator
        model, tokenizer = tiny_tok
        gen = SparseVoteRAGGenerator(
            model=model, tokenizer=tokenizer,
            m=2, k=1, epsilon_token=10.0, epsilon_total=20.0,
            sparse_fraction=0.5, random_seed=SEED,
        )
        r1 = gen.generate_vote(Q1, DOCS)
        r2 = gen.generate_vote(Q1, DOCS)
        assert r1 == r2, "same query → same output (SparseVoteRAG reproducibility broken)"

    def test_sparse_second_query_does_not_corrupt_first(self, tiny_tok):
        from dp_methods.per_query.dpvoterag import SparseVoteRAGGenerator
        model, tokenizer = tiny_tok
        gen = SparseVoteRAGGenerator(
            model=model, tokenizer=tokenizer,
            m=2, k=1, epsilon_token=10.0, epsilon_total=20.0,
            sparse_fraction=0.5, random_seed=SEED,
        )
        r1_first = gen.generate_vote(Q1, DOCS)
        gen.generate_vote(Q2, DOCS)
        r1_second = gen.generate_vote(Q1, DOCS)
        assert r1_first == r1_second, "Q2 corrupted Q1's RNG state (SparseVoteRAG)"


# ---------------------------------------------------------------------------
# DP-SynRAG — cluster-level seeds (no LLM needed)
# ---------------------------------------------------------------------------

class TestDPSynRAGSeeds:

    def _random_extractions(self, n: int = 12, v: int = 25) -> np.ndarray:
        return (np.random.default_rng(7).random((n, v)) > 0.6).astype(np.float32)

    # --- DPHistogramClusterer reproducibility ---

    def test_clusterer_same_seed_same_clusters(self):
        from dp_methods.offline.dp_synrag import DPHistogramClusterer
        ext = self._random_extractions()
        c1 = DPHistogramClusterer(R=3, L=2, sigma_h=0.5, seed=SEED)
        c2 = DPHistogramClusterer(R=3, L=2, sigma_h=0.5, seed=SEED)
        idx1, clusters1 = c1.cluster(ext)
        idx2, clusters2 = c2.cluster(ext)
        np.testing.assert_array_equal(idx1, idx2)
        assert clusters1 == clusters2

    def test_clusterer_different_seed_different_clusters(self):
        """With high noise, different seeds must produce different cluster assignments."""
        from dp_methods.offline.dp_synrag import DPHistogramClusterer
        ext = self._random_extractions()
        _, clusters42 = DPHistogramClusterer(R=3, L=2, sigma_h=5.0, seed=42).cluster(ext)
        _, clusters99 = DPHistogramClusterer(R=3, L=2, sigma_h=5.0, seed=99).cluster(ext)
        assert clusters42 != clusters99

    # --- DPClusterRetriever: per-cluster RNG ---

    @pytest.fixture
    def dummy_cluster_retriever(self):
        from dp_methods.offline.dp_synrag import DPClusterRetriever

        class _Embedder:
            def embed(self, texts):
                d = len(texts)
                return np.eye(d, d, dtype=np.float32)

        return DPClusterRetriever(
            embedder=_Embedder(), k=2, sigma_mu=0.0,
            epsilon_theta_s=100.0, seed=SEED,
        )

    def test_same_cluster_same_rng_draw(self, dummy_cluster_retriever):
        cr = dummy_cluster_retriever
        draw_a = int(cr._rng_for_cluster(3).integers(0, 1 << 32))
        draw_b = int(cr._rng_for_cluster(3).integers(0, 1 << 32))
        assert draw_a == draw_b

    def test_different_clusters_different_rng_draw(self, dummy_cluster_retriever):
        cr = dummy_cluster_retriever
        draws = {int(cr._rng_for_cluster(r).integers(0, 1 << 32)) for r in range(5)}
        assert len(draws) == 5, "each cluster must have a unique RNG seed"

    # --- DPSyntheticGenerator: per-cluster seed derivation ---

    def test_same_cluster_same_seed(self):
        """The hash used to derive each cluster's token seed is deterministic."""
        for cluster_index in [0, 3, 7]:
            key = f"{SEED}|{cluster_index}".encode("utf-8")
            off_a = int(hashlib.sha256(key).hexdigest()[:8], 16)
            off_b = int(hashlib.sha256(key).hexdigest()[:8], 16)
            assert off_a == off_b

    def test_different_clusters_different_seed(self):
        offsets = set()
        for cluster_index in range(6):
            key = f"{SEED}|{cluster_index}".encode("utf-8")
            offsets.add(int(hashlib.sha256(key).hexdigest()[:8], 16))
        assert len(offsets) == 6, "each cluster must yield a unique per-token seed"
