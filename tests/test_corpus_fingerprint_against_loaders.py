"""Check the corpus/question split against what the loaders actually build.

The index cache exempts a set of keys for the two datasets whose corpus is drawn
independently of which questions are kept. That claim was wrong once already —
``max_test_queries``, ``per_bin``, ``top_k_favorites`` and ``seed_size`` were exempt
while genuinely moving the documents of SearchQA, MovieLens and SQuAD — and the
consequence of being wrong is silent: two runs share an index and one is scored
against a corpus it never selected.

A grep-style test cannot catch that. These build the loader twice, vary one key, and
compare the documents. Datasets whose data is not present are skipped rather than
failing, so the suite still runs on a machine without the downloads.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (
    _QUERY_ONLY_KEYS,
    _QUESTION_INDEPENDENT_CORPORA,
    corpus_fingerprint,
)

# Building each loader twice downloads and parses real corpora, which takes minutes.
# Opt in with BENCHDP_SLOW_TESTS=1; the cheap structural checks below always run.
_needs_data = pytest.mark.skipif(
    not os.environ.get("BENCHDP_SLOW_TESTS"),
    reason="loads real datasets; set BENCHDP_SLOW_TESTS=1 to run",
)


def _documents(name: str, **kwargs) -> list[str] | None:
    from experiments.common import _dataset_classes

    cls = _dataset_classes()[name]
    try:
        dataset = cls(random_seed=42, **kwargs)
        dataset.load()
        return dataset.get_documents()
    except Exception as exc:  # data not downloaded on this machine
        pytest.skip(f"{name} unavailable: {type(exc).__name__}: {exc}")


# (dataset, key, value_a, value_b) — keys that must be treated as corpus-affecting.
CORPUS_AFFECTING = [
    ("movielens", "max_test_queries", 40, 200),
    ("movielens", "top_k_favorites", 10, 4),
    ("movielens", "seed_size", 3, 6),
    ("searchqa", "max_test_queries", 40, 80),
]


@_needs_data
@pytest.mark.parametrize("name,key,a,b", CORPUS_AFFECTING, ids=lambda v: str(v))
def test_a_key_that_moves_the_documents_is_in_the_fingerprint(name, key, a, b):
    docs_a = _documents(name, max_index_docs=400, **{key: a})
    docs_b = _documents(name, max_index_docs=400, **{key: b})
    if docs_a == docs_b:
        pytest.skip(f"{name}.{key} does not move the corpus at this size")

    # The documents differ, so the fingerprints must differ too — otherwise both runs
    # share one cached index.
    fp_a = corpus_fingerprint({"name": name, "max_index_docs": 400, key: a})
    fp_b = corpus_fingerprint({"name": name, "max_index_docs": 400, key: b})
    assert fp_a != fp_b, (
        f"{name}: changing {key} changes the indexed documents, but the cache key is "
        "identical, so the second run would be scored against the first one's corpus"
    )


@_needs_data
@pytest.mark.parametrize("name", sorted(_QUESTION_INDEPENDENT_CORPORA))
@pytest.mark.parametrize("key,a,b", [("max_test_queries", 20, 60)])
def test_the_exempt_datasets_really_are_question_independent(name, key, a, b):
    docs_a = _documents(name, max_index_docs=300, **{key: a})
    docs_b = _documents(name, max_index_docs=300, **{key: b})
    assert docs_a == docs_b, (
        f"{name} is exempt from {key} in the index fingerprint, but changing it moves "
        "the corpus — two runs would wrongly share an index"
    )


def test_exemptions_apply_only_to_the_question_independent_corpora():
    for key in _QUERY_ONLY_KEYS:
        exempt = corpus_fingerprint({"name": "natural_questions", key: 1})
        counted = corpus_fingerprint({"name": "searchqa", key: 1})
        assert key not in exempt
        assert key in counted or key in {"split"}, (
            f"{key} is exempt for searchqa, whose corpus is built from the questions "
            "it keeps"
        )
