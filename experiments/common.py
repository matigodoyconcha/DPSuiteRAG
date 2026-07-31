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
    return {
        key: value
        for key, value in ds_cfg.items()
        if key in accepted and key not in _NON_LOADER_KEYS
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
