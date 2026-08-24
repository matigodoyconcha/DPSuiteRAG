"""Repeat queries must embed to exactly the vector a fresh encode produces."""

from __future__ import annotations

import numpy as np
import pytest

from vanilla.embedder import HFEmbedder
from vanilla.query_embedder import HFQueryEmbedder, build_query_embedder

QUERY = "What is the capital of France?"


@pytest.fixture(scope="module")
def query_embedder() -> HFQueryEmbedder:
    try:
        return build_query_embedder(
            HFEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
    except Exception as exc:  # pragma: no cover - network/model availability
        pytest.skip(f"Could not load the embedder: {exc}")


def test_repeat_query_is_bitwise_identical(query_embedder: HFQueryEmbedder) -> None:
    query_embedder._query_cache.clear()
    cold = query_embedder.embed_query(QUERY)
    warm = query_embedder.embed_query(QUERY)
    assert np.array_equal(warm, cold)
    assert warm.dtype == cold.dtype

    query_embedder._query_cache.clear()
    recomputed = query_embedder.embed_query(QUERY)
    assert np.array_equal(recomputed, cold)


def test_caller_may_mutate_what_it_receives(query_embedder: HFQueryEmbedder) -> None:
    query_embedder._query_cache.clear()
    first = query_embedder.embed_query(QUERY)
    reference = first.copy()

    # Retrieval normalises the vector it is handed in place.
    first *= 7.0
    second = query_embedder.embed_query(QUERY)
    assert np.array_equal(second, reference)
    assert second is not first


def test_cache_is_bounded_and_evicts_oldest(query_embedder: HFQueryEmbedder) -> None:
    query_embedder._query_cache.clear()
    query_embedder._QUERY_CACHE_MAX = 3
    try:
        queries = [f"question number {i}?" for i in range(5)]
        vectors = [query_embedder.embed_query(q) for q in queries]

        assert len(query_embedder._query_cache) <= 3
        assert queries[0] not in query_embedder._query_cache
        # An evicted query is simply recomputed, to the same vector.
        assert np.array_equal(query_embedder.embed_query(queries[0]), vectors[0])
    finally:
        del query_embedder._QUERY_CACHE_MAX
        query_embedder._query_cache.clear()


def test_cache_is_per_instance(query_embedder: HFQueryEmbedder) -> None:
    other = HFQueryEmbedder.__new__(HFQueryEmbedder)
    other._query_cache = {}
    query_embedder._query_cache.clear()
    query_embedder.embed_query(QUERY)
    assert query_embedder._query_cache
    assert not other._query_cache


# ---------------------------------------------------------------------------
# What the query tower is built from
# ---------------------------------------------------------------------------

def test_the_query_tower_reuses_the_loaded_model(query_embedder: HFQueryEmbedder) -> None:
    """Two SentenceTransformers of one id would double the GPU footprint for the
    same weights, which is what forced both stages into one component before."""
    embedder = HFEmbedder.__new__(HFEmbedder)
    embedder.model_name = query_embedder.model_name
    embedder.model = query_embedder.model

    assert build_query_embedder(embedder).model is query_embedder.model


def test_an_embedder_that_lost_its_model_is_refused() -> None:
    """A silent fallback here drops arctic-embed's query prompt, and a dropped prompt
    never shows up in the answer — only as retrieval that quietly got worse."""
    embedder = HFEmbedder.__new__(HFEmbedder)
    embedder.model_name = "Snowflake/snowflake-arctic-embed-m-v1.5"
    embedder.model = None

    with pytest.raises(TypeError, match="no loaded SentenceTransformer"):
        build_query_embedder(embedder)


def test_a_tower_that_is_not_sentence_transformers_encodes_symmetrically() -> None:
    """No sentence-transformers model means no query prompt to lose, so the query
    goes through the same call as the documents."""
    class _Embedder:
        def embed(self, texts):
            return [[1.0, 2.0] for _ in texts]

    vec = build_query_embedder(_Embedder()).embed_query(QUERY)
    assert vec.tolist() == [1.0, 2.0]


def test_a_model_id_cannot_be_passed_positionally() -> None:
    """`HFQueryEmbedder("Snowflake/...")` would bind the id to `model` and fail only
    at the first query's encode — after the run had started."""
    with pytest.raises(TypeError):
        HFQueryEmbedder("sentence-transformers/all-MiniLM-L6-v2")
