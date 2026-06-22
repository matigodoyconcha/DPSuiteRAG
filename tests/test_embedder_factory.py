"""Unit tests for the C4 DPR dual-encoder factory helpers (no model downloads)."""
from __future__ import annotations

import pytest

from vanilla.embedder import build_embedder, derive_ctx_encoder, embedder_identity


def test_derive_ctx_encoder_replaces_tower():
    q = "sentence-transformers/facebook-dpr-question_encoder-single-nq-base"
    assert derive_ctx_encoder(q) == (
        "sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base"
    )


def test_derive_ctx_encoder_none_when_not_dpr():
    assert derive_ctx_encoder("sentence-transformers/all-MiniLM-L6-v2") is None


def test_identity_single_tower_is_plain_name():
    assert embedder_identity("foo/bar") == "foo/bar"


def test_identity_dual_distinguishes_from_single():
    q = "sentence-transformers/facebook-dpr-question_encoder-multiset-base"
    ident = embedder_identity(q, dual_encoder=True)
    assert ident != q
    assert "dual::" in ident
    assert "ctx_encoder" in ident  # derived ctx tower is encoded in the key


def test_build_embedder_dual_requires_derivable_or_explicit_ctx():
    with pytest.raises(ValueError, match="ctx_encoder"):
        build_embedder("some/symmetric-model", dual_encoder=True)
