"""Repeat queries must embed to exactly the vector a fresh encode produces."""

from __future__ import annotations

import numpy as np
import pytest

from vanilla.embedder import HFEmbedder

QUERY = "What is the capital of France?"


@pytest.fixture(scope="module")
def embedder() -> HFEmbedder:
    try:
        return HFEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as exc:  # pragma: no cover - network/model availability
        pytest.skip(f"Could not load the embedder: {exc}")


def test_repeat_query_is_bitwise_identical(embedder: HFEmbedder) -> None:
    embedder._query_cache.clear()
    cold = embedder.embed_query(QUERY)
    warm = embedder.embed_query(QUERY)
    assert np.array_equal(warm, cold)
    assert warm.dtype == cold.dtype

    embedder._query_cache.clear()
    recomputed = embedder.embed_query(QUERY)
    assert np.array_equal(recomputed, cold)


def test_caller_may_mutate_what_it_receives(embedder: HFEmbedder) -> None:
    embedder._query_cache.clear()
    first = embedder.embed_query(QUERY)
    reference = first.copy()

    # Retrieval normalises the vector it is handed in place.
    first *= 7.0
    second = embedder.embed_query(QUERY)
    assert np.array_equal(second, reference)
    assert second is not first


def test_cache_is_bounded_and_evicts_oldest(embedder: HFEmbedder) -> None:
    embedder._query_cache.clear()
    embedder._QUERY_CACHE_MAX = 3
    try:
        queries = [f"question number {i}?" for i in range(5)]
        vectors = [embedder.embed_query(q) for q in queries]

        assert len(embedder._query_cache) <= 3
        assert queries[0] not in embedder._query_cache
        # An evicted query is simply recomputed, to the same vector.
        assert np.array_equal(embedder.embed_query(queries[0]), vectors[0])
    finally:
        del embedder._QUERY_CACHE_MAX
        embedder._query_cache.clear()


def test_cache_is_per_instance(embedder: HFEmbedder) -> None:
    other = HFEmbedder.__new__(HFEmbedder)
    other._query_cache = {}
    embedder._query_cache.clear()
    embedder.embed_query(QUERY)
    assert embedder._query_cache
    assert not other._query_cache
