"""Unit tests for the dataset-major orchestrator: index cache scoping/cleanup and
dataset scheduling helpers. No model downloads."""
from __future__ import annotations

import os
from pathlib import Path

import experiments.runner as runner
from experiments.common import dataset_names, select_datasets
from experiments.index_cache import index_cache_dir, load_or_build_index


def _fingerprint_kwargs(**over):
    base = dict(
        dataset_name="triviaqa",
        max_docs=100,
        embedder_model="Snowflake/snowflake-arctic-embed-m-v1.5",
        chunk_size=200,
        chunk_overlap=20,
        seed=42,
    )
    base.update(over)
    return base


def test_index_cache_dir_scoped_to_run(monkeypatch):
    # Same settings + same run id -> same dir (sibling processes of one run share).
    monkeypatch.setenv("BENCHDP_RUN_ID", "run_A")
    a1 = index_cache_dir(**_fingerprint_kwargs())
    a2 = index_cache_dir(**_fingerprint_kwargs())
    assert a1 == a2

    # Different run id -> different dir (concurrent run cannot delete our index).
    monkeypatch.setenv("BENCHDP_RUN_ID", "run_B")
    b1 = index_cache_dir(**_fingerprint_kwargs())
    assert b1 != a1

    # Unset -> deterministic standalone dir, distinct from the scoped ones.
    monkeypatch.delenv("BENCHDP_RUN_ID", raising=False)
    s1 = index_cache_dir(**_fingerprint_kwargs())
    s2 = index_cache_dir(**_fingerprint_kwargs())
    assert s1 == s2 and s1 not in (a1, b1)


def test_index_cache_dir_still_keys_on_settings(monkeypatch):
    monkeypatch.setenv("BENCHDP_RUN_ID", "run_A")
    assert index_cache_dir(**_fingerprint_kwargs()) != index_cache_dir(
        **_fingerprint_kwargs(dataset_name="searchqa")
    )
    assert index_cache_dir(**_fingerprint_kwargs(no_chunk=False)) != index_cache_dir(
        **_fingerprint_kwargs(no_chunk=True)
    )


class _FakeIndexer:
    def load(self, d):
        self.loaded = d

    def save(self, d):
        Path(d).mkdir(parents=True, exist_ok=True)


def test_load_or_build_records_before_building(tmp_path, monkeypatch):
    # A build that crashes must still leave a cleanup entry in the manifest.
    manifest = tmp_path / "m.manifest"
    monkeypatch.setenv("BENCHDP_INDEX_MANIFEST", str(manifest))
    cache = tmp_path / "idx_crash"

    def boom():
        raise RuntimeError("build failed")

    with __import__("pytest").raises(RuntimeError):
        load_or_build_index(_FakeIndexer(), build_fn=boom, cache_dir=cache)
    assert str(cache) in manifest.read_text().split()


def test_cleanup_indices_removes_dirs_and_manifest(tmp_path):
    manifest = tmp_path / "m.manifest"
    d1, d2 = tmp_path / "idx_1", tmp_path / "idx_2"
    d1.mkdir()
    d2.mkdir()
    manifest.write_text(f"{d1}\n{d2}\n{d1}\n")  # duplicate d1
    runner._cleanup_indices(str(manifest))
    assert not d1.exists() and not d2.exists()
    assert not manifest.exists()


def test_cleanup_indices_tolerates_missing(tmp_path):
    manifest = tmp_path / "m.manifest"
    manifest.write_text(f"{tmp_path / 'never_built'}\n")
    runner._cleanup_indices(str(manifest))  # must not raise
    assert not manifest.exists()


def test_select_and_dataset_names(monkeypatch):
    cfgs = [{"name": "triviaqa"}, {"name": "searchqa"}, {"name": "medical"}]
    monkeypatch.delenv("BENCHDP_ONLY_DATASET", raising=False)
    assert select_datasets(cfgs) == cfgs
    monkeypatch.setenv("BENCHDP_ONLY_DATASET", "searchqa")
    assert [d["name"] for d in select_datasets(cfgs)] == ["searchqa"]
    assert dataset_names({"datasets": [{"name": "a"}, {"name": "b"}]}) == ["a", "b"]
    assert dataset_names({"dataset": {"name": "x"}}) == ["x"]
