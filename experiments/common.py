"""Shared helpers for the experiment runners and the orchestrator.

The orchestrator evaluates one dataset at a time so it can build that dataset's
index once, share it across every method and LLM, and delete it before moving on
to the next dataset. To make that possible each method run can be restricted to a
single dataset via the ``BENCHDP_ONLY_DATASET`` environment variable; when the
variable is unset a method still sweeps every dataset in its own config.
"""

from __future__ import annotations

import inspect
import os
import warnings
from typing import Any


def only_dataset() -> str | None:
    """Name of the single dataset a run is restricted to, or ``None``."""
    return os.environ.get("BENCHDP_ONLY_DATASET") or None


def select_datasets(dataset_cfgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the dataset named by :func:`only_dataset`.

    Returned unchanged when no restriction is set, so a standalone method run
    still evaluates every dataset in its config.
    """
    name = only_dataset()
    if name is None:
        return list(dataset_cfgs)
    return [d for d in dataset_cfgs if str(d.get("name")) == name]


def iter_dataset_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """One config dict per dataset to evaluate.

    A top-level ``datasets:`` list wins, with each entry merged over the singular
    ``dataset:`` block that supplies the defaults; without the list the singular block
    is used on its own. That lets one YAML sweep several datasets, each with its own
    corpus size and query count.
    """
    base = config.get("dataset", {}) or {}
    datasets = config.get("datasets")
    if isinstance(datasets, list) and datasets:
        return [{**base, **(entry or {})} for entry in datasets]
    return [dict(base)]


def iter_llm_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """One llm-config dict per LLM to evaluate, mirroring :func:`iter_dataset_configs`."""
    base = config.get("llm", {}) or {}
    llms = config.get("llms")
    if isinstance(llms, list) and llms:
        return [{**base, **(entry or {})} for entry in llms]
    return [dict(base)]


def llm_label(llm_cfg: dict[str, Any]) -> str:
    """Filesystem-safe label for an LLM, derived from its ``model_name``."""
    return str(llm_cfg.get("model_name", "model")).replace("/", "__")


#: Config keys that describe where a run writes or how it is orchestrated, never
#: something a dataset loader should receive.
_NON_LOADER_KEYS = frozenset({"name", "docs_path", "queries_path"})


def dataset_loader_kwargs(loader_cls: type, ds_cfg: dict[str, Any]) -> dict[str, Any]:
    """Keys of ``ds_cfg`` the loader's constructor actually accepts.

    Forwarding by introspection rather than by a hardcoded list is what lets a filter
    declared in a YAML entry — a relevance band, a catalogue cap, a support binning —
    reach the loader of whichever method is running. A runner that names its kwargs
    explicitly drops any filter it was not written to know about, and drops it without
    a word, so the run looks like it applied a filter it never saw.
    """
    accepted = set(inspect.signature(loader_cls.__init__).parameters) - {"self"}
    forwarded = {}
    dropped = []
    for key, value in ds_cfg.items():
        if key in _NON_LOADER_KEYS:
            continue
        if key in accepted:
            forwarded[key] = value
        else:
            dropped.append(key)
    if dropped:
        # Silence here reproduces the failure this function exists to remove: a run
        # that looks like it applied a filter, or a corpus cap under a key the loader
        # does not know, and reports nothing either way.
        warnings.warn(
            f"{loader_cls.__name__} does not accept {sorted(dropped)}; these keys were "
            "dropped and have no effect on the run.",
            RuntimeWarning,
            stacklevel=2,
        )
    return forwarded


#: Datasets whose corpus is drawn independently of which test questions are kept.
#: Natural Questions and TriviaQA both sample the DPR ``psgs_w100`` passages, so their
#: question-side filters change nothing about the index — and holding those filters out
#: of the key is what lets the two share one ~34 GB build instead of two.
#:
#: No other dataset qualifies. SearchQA builds its corpus out of the very questions it
#: keeps; MovieLens reserves the query users and indexes the rest, so anything moving
#: that partition moves the corpus; SQuAD seeds the index with each test question's
#: gold context. For all of them every loader key is treated as corpus-affecting, which
#: is the safe direction: an unnecessary rebuild costs time, whereas wrongly sharing an
#: index means scoring a run against documents it never selected, silently.
_QUESTION_INDEPENDENT_CORPORA = frozenset({"natural_questions", "triviaqa"})

#: Query-side keys, exempt only for the datasets above.
_QUERY_ONLY_KEYS = frozenset({
    "max_test_queries",
    "test_split",
    "split",
    "relevance_min",
    "relevance_max",
    "n_relevance_bins",
    "per_bin",
    "relevance_path",
})

#: Config keys naming the same quantity in different runners' YAMLs. Normalised so two
#: methods asking for the same corpus land on one cached index rather than building it
#: twice — for the DPR corpus that is ~20 h and ~34 GB per duplicate.
_KEY_ALIASES = {"max_samples": "max_index_docs"}


def corpus_fingerprint(ds_cfg: dict[str, Any]) -> dict[str, Any]:
    """The parts of a dataset entry that determine which documents get indexed.

    Feeds :func:`experiments.index_cache.index_cache_dir` so that two entries naming
    the same dataset with different filters — a catalogue cap, a query count that
    shifts the document/question split, a sampling seed — do not collide on one cached
    index and end up scored against each other's corpus.
    """
    name = str(ds_cfg.get("name", "")).lower()
    exempt = _QUERY_ONLY_KEYS if name in _QUESTION_INDEPENDENT_CORPORA else frozenset()
    fingerprint = {}
    for key, value in ds_cfg.items():
        if key in exempt or key in _NON_LOADER_KEYS:
            continue
        fingerprint[_KEY_ALIASES.get(key, key)] = value
    return fingerprint


def build_chunker(config: dict[str, Any]):
    """Return the chunker the ``chunking`` section asks for; whole documents by default.

    The benchmark's privacy unit is one corpus document, and every method reports its
    epsilon against that unit. Splitting a document into ``c`` chunks and protecting a
    single chunk only yields a document-level guarantee through group privacy, which
    costs ``(c*eps, c*exp((c-1)*eps)*delta)`` — the delta grows exponentially in ``c``,
    so at five chunks and eps=1 a delta of 1e-5 has already degraded past 1e-3 and the
    document-level statement is worth nothing. Indexing whole documents is what makes
    the reported number mean what the thesis says it means.

    ``chunking.enabled: true`` is available for a corpus whose documents are already
    short enough to be one privacy unit each, or for an experiment that deliberately
    accepts the degradation; :func:`metrics.workload_accounting.group_privacy` gives
    the correction.
    """
    from vanilla.chunker import FixedSizeChunker, NoChunker

    chunk_cfg = config.get("chunking", {}) or {}
    if not chunk_cfg.get("enabled", False):
        max_words = chunk_cfg.get("max_document_words")
        return NoChunker(max_words=int(max_words) if max_words else None)
    return FixedSizeChunker(
        chunk_size=int(chunk_cfg.get("chunk_size", 256)),
        overlap=int(chunk_cfg.get("overlap", 32)),
    )


def chunking_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    """Chunking settings as the index cache key needs them.

    Keeps the cache arguments derived from the same config section the chunker is,
    so a runner cannot build with one chunker and cache under another's key.
    """
    chunk_cfg = config.get("chunking", {}) or {}
    enabled = bool(chunk_cfg.get("enabled", False))
    max_words = chunk_cfg.get("max_document_words")
    return {
        "no_chunk": not enabled,
        "chunk_size": int(chunk_cfg.get("chunk_size", 256)),
        "chunk_overlap": int(chunk_cfg.get("overlap", 32)),
        # Its own key: a document cap changes the indexed text while leaving the number
        # of entries alone, and chunk_size drops out of the key entirely when documents
        # are indexed whole.
        "max_document_words": int(max_words) if max_words else None,
    }


def dataset_names(config: dict[str, Any]) -> list[str]:
    """Ordered dataset names a method config will evaluate.

    Mirrors the runners' selection: a top-level ``datasets:`` list, otherwise the
    singular ``dataset:`` block.
    """
    datasets = config.get("datasets")
    if isinstance(datasets, list) and datasets:
        return [str((e or {}).get("name")) for e in datasets if (e or {}).get("name")]
    base = config.get("dataset") or {}
    name = base.get("name")
    return [str(name)] if name else []
