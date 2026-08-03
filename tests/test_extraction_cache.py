"""The keyword cache may cross privacy configurations; the synthetic one may not."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.extraction_cache import CachedKeywordExtractor
from experiments.run_dp_synrag import _synth_cache_dir

VOCAB = ["alpha", "beta", "gamma", "delta"]
DOCS = ["a document about alpha", "another about beta and gamma"]


class FakeExtractor:
    """Counts extractions so a cache hit is observable."""

    def __init__(self, vocab=VOCAB, K=2, max_new_tokens=8, model_name="fake/model"):
        self.vocab = list(vocab)
        self.K = K
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.prompt_template = "extract {K} from {document}"
        self.substring_fallback = False
        self.calls = 0
        self.unloaded = 0

    @property
    def vocab_size(self):
        return len(self.vocab)

    def extract(self, documents):
        self.calls += 1
        out = np.zeros((len(documents), self.vocab_size), dtype=np.float32)
        for i, doc in enumerate(documents):
            for j, w in enumerate(self.vocab):
                if w in doc:
                    out[i, j] = 1.0
        return out

    def unload(self):
        self.unloaded += 1


def _cached(tmp_path, **kw):
    return CachedKeywordExtractor(FakeExtractor(**kw), root=tmp_path)


def test_a_second_extraction_is_served_from_disk(tmp_path):
    first = _cached(tmp_path)
    a = first.extract(DOCS)
    second = _cached(tmp_path)
    b = second.extract(DOCS)
    assert second._inner.calls == 0, "the cached run still called the model"
    np.testing.assert_array_equal(a, b)


def test_the_rebuilt_matrix_is_the_one_the_extractor_produced(tmp_path):
    direct = FakeExtractor().extract(DOCS)
    _cached(tmp_path).extract(DOCS)          # populate
    served = _cached(tmp_path).extract(DOCS)  # rebuild from coordinates
    np.testing.assert_array_equal(direct, served)
    assert served.dtype == direct.dtype
    assert served.shape == direct.shape


def test_different_documents_do_not_share_an_entry(tmp_path):
    _cached(tmp_path).extract(DOCS)
    other = _cached(tmp_path)
    other.extract(["a completely different document"])
    assert other._inner.calls == 1


@pytest.mark.parametrize("field,value", [
    ("K", 3),
    ("max_new_tokens", 16),
    ("model_name", "other/model"),
    ("vocab", ["alpha", "beta"]),
])
def test_anything_that_changes_the_extraction_changes_the_key(tmp_path, field, value):
    _cached(tmp_path).extract(DOCS)
    changed = _cached(tmp_path, **{field: value})
    changed.extract(DOCS)
    assert changed._inner.calls == 1, f"{field} was not part of the key"


def test_an_interrupted_write_is_not_served(tmp_path):
    first = _cached(tmp_path)
    first.extract(DOCS)
    # Drop the marker, leaving the arrays behind: exactly what a killed build leaves.
    for marker in tmp_path.rglob(".extraction_complete"):
        marker.unlink()
    second = _cached(tmp_path)
    second.extract(DOCS)
    assert second._inner.calls == 1


def test_the_wrapper_is_transparent(tmp_path):
    wrapped = _cached(tmp_path)
    wrapped.unload()
    assert wrapped._inner.unloaded == 1
    assert wrapped.vocab_size == len(VOCAB)


# --- the layer below: the synthetic corpus IS the mechanism's output ----------------

def _cfg(**over):
    base = {
        "keyword_extractor": {"K": 10, "vocab_min_word_len": 1, "exclude_stopwords": True,
                              "max_new_tokens": 50},
        "clustering": {"R": 50, "L": 5, "sigma_h": 7.071},
        "cluster_retrieval": {"k": 80, "sigma_mu": 7.454, "epsilon_theta_s": 0.4},
        "synthetic_generation": {"T": 70, "c": 0.25, "tau": 1.0},
        "self_filter": {"enabled": False},
        "models": {"embedder": "e"},
        "llm": {"model_name": "m"},
        "chunking": {"max_document_words": 400},
        "random_seed": 42,
    }
    for section, values in over.items():
        base[section] = {**base.get(section, {}), **values}
    return base


DS = {"name": "chatdoctor", "max_index_docs": 20000}


@pytest.mark.parametrize("over", [
    {"synthetic_generation": {"c": 1.0}},      # the budget knob
    {"synthetic_generation": {"tau": 0.5}},
    {"clustering": {"sigma_h": 1.0}},          # the noise
    {"cluster_retrieval": {"sigma_mu": 1.0}},
    {"cluster_retrieval": {"epsilon_theta_s": 2.0}},
    {"keyword_extractor": {"vocab_min_word_len": 3}},
    {"keyword_extractor": {"exclude_stopwords": False}},
    {"keyword_extractor": {"max_new_tokens": 20}},
])
def test_a_different_privacy_configuration_gets_a_different_corpus(over):
    """Sharing one of these across configurations reports one epsilon over another's output."""
    assert _synth_cache_dir(_cfg(), DS, 1) != _synth_cache_dir(_cfg(**over), DS, 1)


def test_a_different_noise_seed_gets_a_different_corpus():
    # The corpus is the mechanism's output, so repetitions must not share one — a
    # sweep over seeds would otherwise report a mechanism variance of zero.
    assert _synth_cache_dir(_cfg(), DS, 1) != _synth_cache_dir(_cfg(), DS, 2)


def test_the_same_configuration_reaches_the_same_corpus_across_runs(monkeypatch):
    # What the run id used to prevent: an identical cell rebuilding from scratch.
    monkeypatch.setenv("BENCHDP_RUN_ID", "20260101_000000")
    first = _synth_cache_dir(_cfg(), DS, 1)
    monkeypatch.setenv("BENCHDP_RUN_ID", "20260202_111111")
    assert _synth_cache_dir(_cfg(), DS, 1) == first
