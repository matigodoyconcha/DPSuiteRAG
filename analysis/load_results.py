"""Result loader and normalisation layer for BenchDP-RAG.

Reads every JSON produced by the experiment runners, normalises the
heterogeneous schemas into a single CanonicalRecord dataclass, and exposes
``load_all`` as the single entry-point for the analysis scripts.
"""
from __future__ import annotations

import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Per-run directory created by experiments/runner.py, e.g. ``run_20260618_113814``.
_RUN_STAMP_RE = re.compile(r"^run_\d{8}_\d{6}$")

# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

@dataclass
class CanonicalRecord:
    method: str
    dataset: str
    variant: str                    # "quick" | "full"
    source_file: str
    model: Optional[str]

    # Primary epsilon axis (chosen per-method, see _normalise_*)
    epsilon_value: Optional[float]
    epsilon_kind: str               # "input_total" | "spent" | "derived_zcdp" | "none" | "na"

    # Method-specific epsilon bookkeeping
    epsilon_composed: Optional[float]   # DP-RAG: RDP→advanced→basic composition
    epsilon_sum: Optional[float]        # DP-RAG: eps_r + eps_g
    epsilon_allocated: Optional[float]  # DPVoteRAG: epsilon_total (input)
    epsilon_spent: Optional[float]      # DPVoteRAG: dp_spent.epsilon_total_spent

    delta: Optional[float]

    utility: dict[str, float] = field(default_factory=dict)
    privacy: dict[str, float] = field(default_factory=dict)

    n_queries: Optional[int] = None
    runtime: Optional[float] = None     # seconds per query


# ---------------------------------------------------------------------------
# Helpers shared with summary_table and plot_tradeoff
# ---------------------------------------------------------------------------

_UTILITY_KEYS = {"contains_accuracy", "match_accuracy", "hit_rate", "rouge_l", "f1_score", "bert_score_f1"}
_PRIVACY_KEYS = {"s2mia_auc", "mia_auc"}


def _dataset_name(payload: dict) -> str:
    ds = payload.get("dataset", "unknown")
    if isinstance(ds, dict):
        return str(ds.get("name", "unknown"))
    return str(ds)


def _model_name(payload: dict) -> Optional[str]:
    """Return the LLM used for a run, or ``None`` if not recorded.

    Handles the three payload shapes produced by the runners: a top-level
    ``llm`` key (DP-RAG / DPVoteRAG / DPSparseVoteRAG / DP-KSA), a nested
    ``hyperparams.llm`` (DP-SynRAG), and ``models.llm`` (vanilla).
    """
    llm = payload.get("llm")
    if not llm:
        hp = payload.get("hyperparams")
        if isinstance(hp, dict):
            llm = hp.get("llm")
    if not llm:
        models = payload.get("models")
        if isinstance(models, dict):
            llm = models.get("llm")
    return str(llm) if llm else None


def _get_metrics(payload: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Return (utility_dict, privacy_dict) from a result payload.

    Handles two layouts:
    - Nested: ``payload["metrics"]`` is a dict.
    - Flat: utility/privacy keys scattered at the top level.

    The flat ``mia_auc`` alias is normalised to ``s2mia_auc`` throughout.
    """
    nan = float("nan")

    def _try_float(v) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return nan

    raw: dict[str, float] = {}

    nested = payload.get("metrics")
    if isinstance(nested, dict):
        for k, v in nested.items():
            raw[k] = _try_float(v)

    # Also collect whitelisted top-level keys (flat layout used by DPVoteRAG)
    for k in list(_UTILITY_KEYS) + list(_PRIVACY_KEYS):
        if k in payload and k not in raw:
            raw[k] = _try_float(payload[k])

    # Normalise alias: ``mia_auc`` is the flat top-level key score_results.py
    # writes for the primary AUC-MIA; fold it into ``s2mia_auc`` when the
    # canonical key is absent.
    if "s2mia_auc" not in raw and "mia_auc" in raw:
        raw["s2mia_auc"] = raw["mia_auc"]
    raw.pop("mia_auc", None)

    utility: dict[str, float] = {}
    privacy: dict[str, float] = {}
    for k, v in raw.items():
        if k in _UTILITY_KEYS:
            utility[k] = v
        elif k == "s2mia_auc":
            privacy[k] = v

    return utility, privacy


def _method_from_path(rel: Path) -> tuple[str, str]:
    """Return (method_name, variant) from the path relative to results/.

    Results may be nested under a per-run ``run_<YYYYmmdd_HHMMSS>/`` directory
    created by experiments/runner.py; that timestamp segment is skipped so the
    method name is taken from the segment that actually names the method (this
    also tolerates pointing the loader straight at ``results/`` or at a single
    ``run_<stamp>/`` directory). The first remaining segment is the canonical
    method name; a ``_paper_faithful`` or ``_quick`` suffix is stripped and
    returned as the variant (``_quick`` is kept for older result directories).
    """
    parts = [p for p in rel.parts if not _RUN_STAMP_RE.match(p)]
    segment = parts[0] if parts else "unknown"
    for suffix, variant in (("_paper_faithful", "paper_faithful"), ("_quick", "quick")):
        if segment.endswith(suffix):
            return segment[: -len(suffix)], variant
    return segment, "full"


def pareto_front(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return the Pareto-optimal subset of (mia_auc, utility) points.

    A point dominates another when it has *lower* MIA-AUC (better privacy)
    AND *higher* utility.  NaN coordinates are excluded.
    """
    valid = [(x, y) for x, y in points if not (math.isnan(x) or math.isnan(y))]
    front: list[tuple[float, float]] = []
    for candidate in valid:
        dominated = False
        for other in valid:
            if other == candidate:
                continue
            if other[0] <= candidate[0] and other[1] >= candidate[1]:
                if other[0] < candidate[0] or other[1] > candidate[1]:
                    dominated = True
                    break
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda p: p[0])


# ---------------------------------------------------------------------------
# Per-method normalisers
# ---------------------------------------------------------------------------

def _normalise_vanilla(
    payload: dict,
    source_file: str,
    variant: str,
    *,
    recompute: bool,
    bertscore: bool,
) -> CanonicalRecord:
    model = None
    models_block = payload.get("models", {})
    if isinstance(models_block, dict):
        model = models_block.get("llm")

    utility: dict[str, float] = {}
    privacy: dict[str, float] = {}

    if recompute:
        results = payload.get("results", [])
        predictions = [r.get("answer", "") for r in results]
        references = [r.get("reference", "") for r in results]

        metric_names = ["contains_accuracy", "match_accuracy", "rouge_l", "f1_score"]
        if bertscore:
            metric_names.append("bert_score_f1")

        if predictions and references:
            from metrics import compute_utility_metrics
            utility = compute_utility_metrics(predictions, references, metric_names)
            # Privacy (AUC-MIA) is produced by the separate --mia-only run, not
            # recomputed here; utility runs simply leave the privacy dict empty.

    return CanonicalRecord(
        method="vanilla",
        dataset=_dataset_name(payload),
        variant=variant,
        source_file=source_file,
        model=model,
        epsilon_value=None,
        epsilon_kind="none",
        epsilon_composed=None,
        epsilon_sum=None,
        epsilon_allocated=None,
        epsilon_spent=None,
        delta=None,
        utility=utility,
        privacy=privacy,
        n_queries=payload.get("num_test_queries"),
        runtime=None,
    )


def _normalise_dp_rag(payload: dict, source_file: str, variant: str) -> CanonicalRecord:
    eps_r = payload.get("epsilon_retrieval")
    eps_g = payload.get("epsilon_generation")

    epsilon_sum: Optional[float] = None
    if eps_r is not None and eps_g is not None:
        try:
            epsilon_sum = float(eps_r) + float(eps_g)
        except (TypeError, ValueError):
            pass

    epsilon_composed: Optional[float] = None
    accounting = payload.get("privacy_accounting")
    if isinstance(accounting, dict):
        # Prefer advanced composition, fall back to rdp then basic
        for key in ("advanced", "rdp", "basic"):
            entry = accounting.get(key)
            if isinstance(entry, dict) and entry.get("epsilon") is not None:
                try:
                    epsilon_composed = float(entry["epsilon"])
                    break
                except (TypeError, ValueError):
                    pass

    # Primary epsilon: composed if available, otherwise sum (quick runs lack accounting)
    if epsilon_composed is not None:
        epsilon_value = epsilon_composed
    else:
        epsilon_value = epsilon_sum

    delta_val: Optional[float] = None
    try:
        delta_val = float(payload["delta"])
    except (KeyError, TypeError, ValueError):
        pass

    utility, privacy = _get_metrics(payload)

    return CanonicalRecord(
        method="dp_rag",
        dataset=_dataset_name(payload),
        variant=variant,
        source_file=source_file,
        model=_model_name(payload),
        epsilon_value=epsilon_value,
        epsilon_kind="input_total",
        epsilon_composed=epsilon_composed,
        epsilon_sum=epsilon_sum,
        epsilon_allocated=None,
        epsilon_spent=None,
        delta=delta_val,
        utility=utility,
        privacy=privacy,
        n_queries=payload.get("n_queries"),
        runtime=None,
    )


def _normalise_dpvoterag(
    payload: dict, source_file: str, variant: str, method: str
) -> CanonicalRecord:
    epsilon_allocated: Optional[float] = None
    try:
        epsilon_allocated = float(payload["epsilon_total"])
    except (KeyError, TypeError, ValueError):
        pass

    epsilon_spent: Optional[float] = None
    dp_spent = payload.get("dp_spent")
    if isinstance(dp_spent, dict):
        try:
            epsilon_spent = float(dp_spent["epsilon_total_spent"])
        except (KeyError, TypeError, ValueError):
            pass

    delta_val: Optional[float] = None
    try:
        delta_val = float(payload["delta_total"])
    except (KeyError, TypeError, ValueError):
        pass

    utility, privacy = _get_metrics(payload)

    runtime: Optional[float] = None
    try:
        runtime = float(payload["time_per_query_sec"])
    except (KeyError, TypeError, ValueError):
        pass

    predictions = payload.get("predictions", [])
    n_queries: Optional[int] = len(predictions) if predictions else payload.get("n_queries")

    return CanonicalRecord(
        method=method,
        dataset=_dataset_name(payload),
        variant=variant,
        source_file=source_file,
        model=_model_name(payload),
        epsilon_value=epsilon_allocated,
        epsilon_kind="input_total",
        epsilon_composed=None,
        epsilon_sum=None,
        epsilon_allocated=epsilon_allocated,
        epsilon_spent=epsilon_spent,
        delta=delta_val,
        utility=utility,
        privacy=privacy,
        n_queries=n_queries,
        runtime=runtime,
    )


def _normalise_dp_synrag(payload: dict, source_file: str, variant: str) -> CanonicalRecord:
    epsilon_value: Optional[float] = None
    delta_val: Optional[float] = None
    accounting = payload.get("privacy_accounting", {})
    if isinstance(accounting, dict):
        zcdp = accounting.get("zcdp", {})
        if isinstance(zcdp, dict):
            try:
                epsilon_value = float(zcdp["epsilon"])
            except (KeyError, TypeError, ValueError):
                pass
            try:
                delta_val = float(zcdp["delta"])
            except (KeyError, TypeError, ValueError):
                pass

    utility, privacy = _get_metrics(payload)

    return CanonicalRecord(
        method="dp_synrag",
        dataset=_dataset_name(payload),
        variant=variant,
        source_file=source_file,
        model=_model_name(payload),
        epsilon_value=epsilon_value,
        epsilon_kind="derived_zcdp",
        epsilon_composed=None,
        epsilon_sum=None,
        epsilon_allocated=None,
        epsilon_spent=None,
        delta=delta_val,
        utility=utility,
        privacy=privacy,
        n_queries=payload.get("n_queries"),
        runtime=None,
    )


def _normalise_generic(
    payload: dict, source_file: str, variant: str, method: str
) -> CanonicalRecord:
    """Fallback normaliser for dpksa, private_rag (epsilon is an input budget)."""
    epsilon_value: Optional[float] = None
    epsilon_kind = "na"

    for key in ("epsilon_total", "epsilon", "eps"):
        if key in payload:
            try:
                epsilon_value = float(payload[key])
                epsilon_kind = "input_total"
                break
            except (TypeError, ValueError):
                pass

    utility, privacy = _get_metrics(payload)

    return CanonicalRecord(
        method=method,
        dataset=_dataset_name(payload),
        variant=variant,
        source_file=source_file,
        model=_model_name(payload),
        epsilon_value=epsilon_value,
        epsilon_kind=epsilon_kind,
        epsilon_composed=None,
        epsilon_sum=None,
        epsilon_allocated=None,
        epsilon_spent=None,
        delta=None,
        utility=utility,
        privacy=privacy,
        n_queries=payload.get("n_queries"),
        runtime=None,
    )


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

_DPVOTE_METHODS = {"dpvoterag", "dpsparsevoterag"}
_VALID_EPSILON_KINDS = {"input_total", "spent", "derived_zcdp", "none", "na"}


def load_all(
    results_dir: str | Path = "results",
    datasets: Optional[list[str]] = None,
    methods: Optional[list[str]] = None,
    recompute_vanilla: bool = True,
    bertscore: bool = False,
) -> list[CanonicalRecord]:
    """Load and normalise all experiment result JSON files.

    Args:
        results_dir: Root of the results directory tree.
        datasets: If given, only load records whose dataset matches one of these names.
        methods: If given, only load records whose method matches one of these names.
        recompute_vanilla: Recompute utility/privacy metrics from vanilla ``results[]``
            instead of reading pre-computed values (there are none in the vanilla schema).
        bertscore: Include BERTScore F1 when recomputing vanilla metrics.

    Returns:
        List of CanonicalRecord, one per successfully parsed JSON file.
        Files that cannot be parsed are skipped with a warning.
    """
    root = Path(results_dir)
    if not root.exists():
        logger.warning("results_dir %s does not exist", root)
        return []

    records: list[CanonicalRecord] = []
    json_files = sorted(root.glob("*/**/*.json"))

    for path in json_files:
        rel = path.relative_to(root)
        method, variant = _method_from_path(rel)

        if methods is not None and method not in methods:
            continue

        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            logger.warning("Skipping %s — could not parse JSON: %s", path, exc)
            continue

        try:
            record = _dispatch(
                payload,
                source_file=str(path),
                variant=variant,
                method=method,
                recompute_vanilla=recompute_vanilla,
                bertscore=bertscore,
            )
        except Exception as exc:
            logger.warning("Skipping %s — normalisation error: %s", path, exc)
            continue

        if datasets is not None and record.dataset not in datasets:
            continue

        records.append(record)

    return records


def _dispatch(
    payload: dict,
    source_file: str,
    variant: str,
    method: str,
    recompute_vanilla: bool,
    bertscore: bool,
) -> CanonicalRecord:
    if method == "vanilla":
        return _normalise_vanilla(
            payload,
            source_file,
            variant,
            recompute=recompute_vanilla,
            bertscore=bertscore,
        )
    if method == "dp_rag":
        return _normalise_dp_rag(payload, source_file, variant)
    if method in _DPVOTE_METHODS:
        return _normalise_dpvoterag(payload, source_file, variant, method)
    if method == "dp_synrag":
        return _normalise_dp_synrag(payload, source_file, variant)
    return _normalise_generic(payload, source_file, variant, method)
