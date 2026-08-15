"""The benchmark indexes whole documents, and every config agrees on that."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import build_chunker, chunking_fingerprint
from metrics.workload_accounting import group_privacy
from vanilla.chunker import FixedSizeChunker, NoChunker

CONFIG_DIRS = [
    PROJECT_ROOT / "experiments" / "configs" / "full",
    PROJECT_ROOT / "experiments" / "configs" / "paper_faithful",
]
CONFIGS = sorted(p for d in CONFIG_DIRS if d.is_dir() for p in d.glob("*.yaml"))

#: The zero-shot condition puts no document in the prompt, so it has no privacy unit to
#: state a policy about. A chunking block there would be a knob that decides nothing.
NON_INDEXING = {"no_rag"}


def test_configs_were_found():
    assert len(CONFIGS) >= 13


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_config_declares_whole_document_indexing(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if path.stem in NON_INDEXING:
        pytest.skip(f"{path.stem} indexes nothing")
    assert "chunking" in config, f"{path} does not state a chunking policy"
    assert config["chunking"]["enabled"] is False, (
        f"{path} splits documents, so its reported epsilon is a per-chunk number "
        "while the thesis states a per-document guarantee"
    )


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_config_builds_a_whole_document_chunker(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assert isinstance(build_chunker(config), NoChunker)


def test_the_fingerprint_follows_the_chunker():
    # A runner must not build with one chunker and cache under another's key.
    whole = {"chunking": {"enabled": False}}
    split = {"chunking": {"enabled": True, "chunk_size": 200, "overlap": 20}}
    assert chunking_fingerprint(whole)["no_chunk"] is True
    assert chunking_fingerprint(split)["no_chunk"] is False
    assert chunking_fingerprint(split)["chunk_size"] == 200
    assert chunking_fingerprint(whole) != chunking_fingerprint(split)


def test_an_absent_section_indexes_whole_documents():
    assert isinstance(build_chunker({}), NoChunker)
    assert chunking_fingerprint({})["no_chunk"] is True


def test_chunking_can_still_be_asked_for_explicitly():
    chunker = build_chunker({"chunking": {"enabled": True, "chunk_size": 50, "overlap": 5}})
    assert isinstance(chunker, FixedSizeChunker)


def test_the_cost_the_policy_avoids():
    # Why the default is what it is: a chunk-level guarantee lifted to the document
    # through group privacy destroys the delta well before it destroys the epsilon.
    eps_doc, delta_doc = group_privacy(1.0, 1e-5, 5)
    assert eps_doc == pytest.approx(5.0)
    assert delta_doc > 1e-3
