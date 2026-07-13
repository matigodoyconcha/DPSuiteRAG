"""Unit tests for the embedder factory helpers (no model downloads)."""
from __future__ import annotations

from vanilla.embedder import embedder_identity


def test_identity_is_plain_name():
    assert embedder_identity("foo/bar") == "foo/bar"
