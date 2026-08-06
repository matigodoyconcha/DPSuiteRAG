"""Shared helpers for the experiment runners and the orchestrator.

The orchestrator evaluates one dataset at a time so it can build that dataset's
index once, share it across every method and LLM, and delete it before moving on
to the next dataset. To make that possible each method run can be restricted to a
single dataset via the ``BENCHDP_ONLY_DATASET`` environment variable; when the
variable is unset a method still sweeps every dataset in its own config.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import warnings
from collections.abc import Sequence
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
        if key in _NON_LOADER_KEYS or key in _RUNNER_ONLY_KEYS:
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
#: ``split`` is what two of the YAMLs call the evaluation split; no loader has a
#: parameter by that name, so left unmapped it is dropped and the split silently falls
#: back to the loader's default.
_KEY_ALIASES = {"max_samples": "max_index_docs", "split": "test_split"}

#: Consumed by the runner rather than by a loader, so it is not a dropped filter.
_RUNNER_ONLY_KEYS = frozenset({"overrides"})


def corpus_fingerprint(
    ds_cfg: dict[str, Any], *, default_max_queries: int | None = None
) -> dict[str, Any]:
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
        # `overrides` is a runner-side block that never reaches a loader, so keying on
        # it would split the cache and rebuild an identical corpus.
        if key in exempt or key in _NON_LOADER_KEYS or key in _RUNNER_ONLY_KEYS:
            continue
        fingerprint[_KEY_ALIASES.get(key, key)] = _coerce(key, value)
    # The loader is handed the run's query count when the entry omits one, and for a
    # dataset that partitions its data between corpus and questions that changes the
    # documents. Keying on the entry alone would miss it.
    if (
        default_max_queries is not None
        and "max_test_queries" not in fingerprint
        and "max_test_queries" not in exempt
    ):
        fingerprint["max_test_queries"] = int(default_max_queries)
    return fingerprint


#: Config blocks a method keeps its own settings in. A seed buried in one of these is
#: still the run's seed; the runners differ only in where their YAML happens to put it.
_METHOD_BLOCKS = (
    "dp_rag", "dpvoterag", "dpsparsevoterag", "dpksa", "private_rag", "dp_synrag",
)


def _nested_seed(config: dict[str, Any], key: str) -> int | None:
    """Seed declared inside the running method's own block, if any.

    ``method`` decides which block to read. Scanning the blocks in a fixed order would
    pick up a stale one left behind by copy-paste — a config for one method that still
    carries another's block would silently run on the wrong seed.
    """
    method = str(config.get("method", "")).lower()
    if method not in _METHOD_BLOCKS:
        # vanilla and the MIA configs have no method block; scanning the six anyway
        # would adopt whichever one a copy-paste left behind.
        return None
    for block in [method]:
        section = config.get(block)
        if isinstance(section, dict) and section.get(key) is not None:
            return int(section[key])
    return None


def resolve_seeds(config: dict[str, Any]) -> tuple[int, int]:
    """Return ``(data_seed, run_seed)``.

    ``data_seed`` selects the corpus and the test questions, and feeds the index cache
    key. ``run_seed`` drives the DP mechanisms' noise.

    They have to be separate to repeat a measurement. A single seed used for both means
    a repetition either changes nothing — the same seed reproduces the same run exactly,
    so the "repeat" measures no variance at all — or, if varied, resamples the corpus and
    the questions too, which rebuilds the index (~20 h for the DPR passages) and mixes
    sampling variance into what was meant to isolate the mechanism's own randomness.

    ``run_seed`` comes from ``BENCHDP_RUN_SEED`` when the orchestrator is sweeping
    repetitions; both fall back to the config's ``random_seed``, wherever it sits.
    """
    fallback = config.get("random_seed")
    if fallback is None:
        fallback = _nested_seed(config, "random_seed")
    fallback = 42 if fallback is None else int(fallback)

    data_seed = config.get("data_seed")
    data_seed = fallback if data_seed is None else int(data_seed)

    env_seed = os.environ.get("BENCHDP_RUN_SEED")
    if env_seed:
        run_seed = int(env_seed)
    else:
        run_seed = config.get("run_seed")
        run_seed = fallback if run_seed is None else int(run_seed)
    return data_seed, run_seed


def seed_label() -> str | None:
    """Sub-directory naming the current repetition, or ``None`` outside a sweep."""
    env_seed = os.environ.get("BENCHDP_RUN_SEED")
    return f"seed_{env_seed}" if env_seed else None


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
        "max_document_words": (int(max_words) if max_words and not enabled else None),
    }


def _dataset_classes() -> dict[str, type]:
    """Loader class per dataset name, imported on demand.

    Kept lazy so importing this module — which the orchestrator does just to read a
    config — does not pull in the whole dataset layer.
    """
    from bench_datasets.chatdoctor import ChatDoctorDataset
    from bench_datasets.hotpotqa import HotpotQADataset
    from bench_datasets.medical_dirichlet_phi3 import MedicalDirichletPhi3Dataset
    from bench_datasets.movielens import MovielensDataset
    from bench_datasets.natural_questions import NaturalQuestionsDataset
    from bench_datasets.searchqa import SearchQADataset
    from bench_datasets.squad import SQuADDataset
    from bench_datasets.triviaqa import TriviaQADataset

    return {
        "chatdoctor": ChatDoctorDataset,
        "hotpotqa": HotpotQADataset,
        "medical_dirichlet_phi3": MedicalDirichletPhi3Dataset,
        "movielens": MovielensDataset,
        "natural_questions": NaturalQuestionsDataset,
        "searchqa": SearchQADataset,
        "squad": SQuADDataset,
        "triviaqa": TriviaQADataset,
    }


#: Loader arguments that must be whole numbers. A YAML author who quotes one — or a
#: templating step that stringifies it — otherwise reaches the loader as a string and
#: fails deep inside a slice, where every runner's blanket handler turns it into a
#: skipped dataset rather than an error.
_INTEGER_KEYS = frozenset({
    "max_index_docs", "max_test_queries", "max_samples", "top_movies",
    "min_high_ratings", "top_k_favorites", "seed_size", "random_seed",
    "relevance_min", "relevance_max", "n_relevance_bins", "per_bin",
    "min_answer_snippets", "n_support_bins", "max_scan_questions",
    "relevant_docs_per_query",
})


def _coerce(key: str, value: Any) -> Any:
    if key in _INTEGER_KEYS and value is not None and not isinstance(value, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"dataset entry key {key!r} must be a whole number, got {value!r}"
            ) from None
    return value


def normalise_dataset_config(ds_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve key aliases and decouple the corpus size from the question count.

    Two runners spell the corpus cap ``max_samples`` and the rest ``max_index_docs``;
    the alias is resolved so one dataset entry reads the same everywhere. And where
    ``max_test_queries`` is missing it used to fall back to the corpus size, so a
    20000-document corpus silently asked for 20000 questions — the two are independent
    knobs and are kept that way.
    """
    aliased = {k for k in ds_cfg if k in _KEY_ALIASES}
    for key in aliased:
        target = _KEY_ALIASES[key]
        if target in ds_cfg:
            # Guarding only when the alias happens to come second would leave the
            # effective value depending on the order the keys were typed in.
            raise ValueError(
                f"dataset entry declares both {key!r} and {target!r}, which name the "
                "same quantity; keep one."
            )
    resolved: dict[str, Any] = {}
    for key, value in ds_cfg.items():
        resolved[_KEY_ALIASES.get(key, key)] = _coerce(key, value)
    return resolved


def load_dataset_from_config(
    ds_cfg: dict[str, Any], data_seed: int, *, default_max_queries: int | None = None
):
    """Build and load the dataset a config entry describes.

    Every filter the entry declares and the loader accepts is forwarded, so a filter
    applies to whichever method is running rather than only to the runners whose
    hardcoded kwarg list happened to name it.

    ``default_max_queries`` supplies the run's question count for entries that do not
    state their own. Without it those entries fall through to each loader's built-in
    default, and a loader that takes its questions from the tail of the data then hands
    the method a different set than the one every other method sees — a divergence a
    later prefix cut cannot undo.
    """
    resolved = normalise_dataset_config(ds_cfg)
    if resolved.get("max_test_queries") is None and default_max_queries is not None:
        resolved["max_test_queries"] = int(default_max_queries)
    name = str(resolved.get("name", "")).lower()
    classes = _dataset_classes()
    if name not in classes:
        raise ValueError(
            f"Unsupported dataset {name!r}; known datasets: {sorted(classes)}"
        )
    loader_cls = classes[name]
    kwargs = dataset_loader_kwargs(loader_cls, resolved)
    kwargs.setdefault("random_seed", int(data_seed))
    dataset = loader_cls(**kwargs)
    dataset.load()
    return dataset


def nonprivate_ranking(indexer, query_vector, k: int = 50) -> list[int]:
    """Document ids the same index returns with no privacy mechanism applied.

    The gold set for ``retrieval_precision_at_k``: how much of the retrieval a method
    gives up to its own noise, measured against what the identical embeddings and the
    identical corpus would have returned. Using human relevance labels instead would
    fold the retriever's quality into a number meant to isolate the mechanism's cost.

    Scored straight off the embedding matrix when it is kept, otherwise through the
    FAISS index. Returns an empty list when neither is available rather than raising —
    a run without this diagnostic should still produce its utility numbers.
    """
    import numpy as np

    vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm

    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    matrix = getattr(indexer, "embeddings_matrix", None)
    metadata = getattr(indexer, "metadata", None) or []
    if matrix is not None and len(metadata) not in (0, len(matrix)):
        # Truncating here would hand back a short gold set and quietly deflate the
        # precision of whichever method is being measured.
        raise ValueError(
            f"index metadata has {len(metadata)} entries for {len(matrix)} vectors; "
            "the non-private ranking cannot be built from a mismatched index."
        )
    if matrix is not None and len(matrix):
        scores = matrix @ vector
        k = min(int(k), len(scores))
        # Partition first: a full sort of a 21M-document corpus costs about ten times
        # as much and only the top k is ever read.
        candidates = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        top = candidates[np.argsort(-scores[candidates])]
        return [int(metadata[i].get("doc_id", i)) for i in top if i < len(metadata)]

    index = getattr(indexer, "index", None)
    if index is None or not metadata:
        return []
    _, indices = index.search(vector.reshape(1, -1), int(k))
    return [
        int(metadata[i].get("doc_id", i))
        for i in indices[0]
        if 0 <= i < len(metadata)
    ]


#: Workload sizes every record is priced at, alongside the run's own query count.
#: Q=1 is the regime the source papers publish in — one question, the full budget,
#: nothing composed — and it is the only point where a per-query method's headline
#: number is its real guarantee. The rest trace how each method degrades as the same
#: corpus keeps answering questions, which is where they stop being comparable.
WORKLOAD_SIZES = (1, 5, 10, 25, 50, 100)


def workload_accounting_block(
    method: str,
    *,
    queries: int,
    run_seed: int,
    realised: dict[str, Any] | None = None,
    answer_tokens: int | None = None,
    workload_derived_deltas: Sequence[str] = (),
    **params,
) -> dict[str, Any]:
    """The record's position on the shared axis, ready to drop into a result payload.

    Every method reports its own epsilon in its own units — per query, per corpus, per
    offline build — and those numbers are not comparable. This adds the one figure that
    is: the privacy loss over a workload of ``queries`` questions at the common delta.

    The workload size is a property of the *guarantee*, not of the measurement: epsilon
    at Q=1 and at Q=100 both describe the same run, whose utility is still the mean over
    every question asked. So the whole curve is priced here, from mechanisms already in
    hand, and no point on it costs another query. Reporting only the run's own Q would
    have tied the privacy axis to how many questions happened to be sampled.

    ``answer_tokens`` is the answer length the cell was configured to produce. Every
    text metric is monotone in it, so the protocol holds it at one value across all
    eight methods and the record carries it: a reader comparing two methods can see
    that the number they are compared on was measured over answers of the same length,
    instead of taking it on trust from a pair of YAML files.

    ``workload_derived_deltas`` names the parameters in ``params`` whose value was
    derived from the workload size — the per-release additive deltas that
    :func:`metrics.workload_accounting.delta_token_for_workload` fixes so that the mass
    spent over the stream is the same in every cell. They are inversely proportional to
    Q by construction, so pricing the *other* workload sizes below rescales them by
    ``queries / q``; leaving them at the running cell's value charged ``q/Q`` times the
    mass the axis allows and the curve went unreportable for every ``q >= 2Q`` (measured
    at Q=1: only q=1 survived). The names are declared rather than guessed, because a
    delta that genuinely does not move with Q — a mechanism's own constant — has to stay
    where it is, and only the caller that derived the value knows which it is.

    Note what the curve is: this cell's mechanism priced over q questions. It is *not*
    the Q=q cell of the grid, whose budget knob is recalibrated so that cell lands on
    its own target; ``epsilon_workload_q1`` of a Q=100 cell is therefore far below the
    headline of the Q=1 cell, and both are correct.

    ``realised`` overrides the count-bearing parameters with what the run actually
    released, producing ``epsilon_workload_spent`` beside the budget. That number is
    **not a guarantee**: how many releases a run makes depends on the corpus, so a run
    that halted early would report a smaller epsilon precisely because of what its data
    turned out to be, and no such number bounds a neighbouring corpus. It is reported
    because the gap between the two is worth seeing — a method that spends a fifth of
    what it is charged is paying for an answer length it never produces.

    Returns the failure as data rather than raising, because a diagnostic that aborts a
    finished run would cost the generation that produced it. A method whose additive
    delta overruns the target lands here with ``epsilon_workload = None`` and a reason.
    """
    from metrics.workload_accounting import DEFAULT_DELTA, method_epsilon_total

    block: dict[str, Any] = {
        "workload_queries": int(queries),
        "workload_delta": DEFAULT_DELTA,
        "run_seed": int(run_seed),
    }
    # Written even when there is none: omitting the key made a method that answers at no
    # configured length and a runner that forgot to pass the one it uses produce the same
    # payload, and only the first of those is a fact about the method.
    block["answer_tokens"] = None if answer_tokens is None else int(answer_tokens)
    derived_deltas = tuple(str(name) for name in workload_derived_deltas)
    # Which of the mechanism's deltas move with the workload, and are therefore rescaled
    # when the curve below prices the other workload sizes. A reader comparing two cells
    # of one method needs to know whether their deltas were the same number or the same
    # spend, and the empty list says the mechanism has neither kind.
    block["workload_derived_deltas"] = list(derived_deltas)
    unknown = [name for name in derived_deltas if name not in params]
    if unknown:
        # A mistyped name would otherwise leave the delta silently unscaled, i.e. the bug
        # this parameter exists to fix, reported as if it had been fixed.
        block["workload_derived_delta_error"] = (
            f"not mechanism parameters: {unknown}; the curve below leaves them unscaled"
        )
    if "delta_token" in params:
        # The mechanism's own additive delta per release, derived from Q and the answer
        # length so that Q * T * delta_token is the same mass in every cell. Recorded
        # because it is the one parameter of the accounting that the run also feeds
        # back into the mechanism (it sets the Limited-Domain bottom-out threshold),
        # and a reader has no other way to check the two were the same number.
        block["delta_token"] = float(params["delta_token"])
    try:
        result = method_epsilon_total(method, queries=queries, **params)
    except (ValueError, RuntimeError, TypeError) as exc:
        # TypeError included on purpose: a renamed or mistyped mechanism parameter
        # surfaces as one, and losing a finished run to a diagnostic is the worst
        # possible trade.
        block["epsilon_workload"] = None
        block["workload_error"] = f"{type(exc).__name__}: {exc}"
        return block
    block.update({
        "epsilon_workload": result.epsilon,
        "epsilon_workload_basic": result.epsilon_basic,
        "workload_regime": result.regime,
        "workload_delta_additive": result.delta_additive,
        "workload_best_alpha": result.best_alpha,
    })

    by_q: dict[str, float | None] = {}
    for q in sorted({*WORKLOAD_SIZES, int(queries)}):
        if q == int(queries):
            by_q[str(q)] = result.epsilon
            continue
        # A workload-derived delta is `share * delta / (q * releases)`, so pricing the
        # workload q means charging the delta that workload was allowed — the same total
        # additive mass, spread over q queries instead of Q. Scaling by Q/q reproduces
        # that derivation exactly whatever the release count behind it was, so the block
        # never has to know how each method arrived at its own delta.
        scaled = {
            **params,
            **{
                name: float(params[name]) * int(queries) / q
                for name in derived_deltas
                if name in params
            },
        }
        try:
            by_q[str(q)] = method_epsilon_total(method, queries=q, **scaled).epsilon
        except (ValueError, RuntimeError, TypeError):
            # A workload this method cannot reach is a real answer about the method,
            # not a failure of the run that asked.
            by_q[str(q)] = None
    block["epsilon_workload_by_q"] = by_q
    block["epsilon_workload_q1"] = by_q.get("1")

    if realised:
        # The realised counts are totals over the whole run, so they compose once
        # rather than being repeated per query.
        try:
            spent = method_epsilon_total(
                method, queries=1, **{**params, **realised}
            ).epsilon
        except (ValueError, RuntimeError, TypeError) as exc:
            block["epsilon_workload_spent"] = None
            block["spent_error"] = f"{type(exc).__name__}: {exc}"
        else:
            block["epsilon_workload_spent"] = spent
            budget = block.get("epsilon_workload")
            block["budget_utilisation"] = (
                spent / budget if budget else None
            )
            block["realised_counts"] = dict(realised)
    return block


class RecordingRetriever:
    """Delegates to a retriever and remembers which documents it last returned.

    The pipelines call the retriever internally, so a runner never sees what came back
    and cannot report it. Wrapping is what makes the private side of
    ``retrieval_precision_at_k`` observable without each mechanism growing its own
    reporting code — and the mechanism itself is untouched, so the wrapper cannot
    change what a run computes.

    Records document ids rather than index positions because the privacy unit is the
    document; with chunking enabled several index rows belong to one.
    """

    #: Attributes the wrapper owns; everything else belongs to the wrapped retriever.
    _OWN = frozenset({"_inner", "last_retrieved_ids"})

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "last_retrieved_ids", [])

    def retrieve(self, query_vector, k=None, **kwargs):
        out = (
            self._inner.retrieve(query_vector, **kwargs)
            if k is None
            else self._inner.retrieve(query_vector, k, **kwargs)
        )
        ids = []
        for doc in out or []:
            if not isinstance(doc, dict):
                continue
            # Two shapes in the wild: the vanilla retriever returns the chunk metadata
            # flat, while the DP-RAG and Private-RAG retrievers wrap it under
            # "metadata". Reading only the flat one left those two methods recording an
            # empty list, and an empty list still reaches the scorer — so the precision
            # came out exactly 0.0 for every cell, reading as a total loss of retrieval
            # rather than as a missing measurement.
            doc_id = doc.get("doc_id")
            if doc_id is None and isinstance(doc.get("metadata"), dict):
                doc_id = doc["metadata"].get("doc_id")
            if doc_id is not None:
                ids.append(int(doc_id))
        self.last_retrieved_ids = ids
        return out

    def __getattr__(self, name):
        # Everything else — thresholds, budget state, per-query diagnostics the
        # runners read off the retriever — passes straight through.
        #
        # Reached only for attributes the wrapper does not have, which includes
        # `_inner` itself before __init__ has run (copy and pickle probe an
        # uninitialised instance). Going through the instance dict directly stops that
        # from recursing until the stack runs out.
        inner = object.__getattribute__(self, "__dict__").get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def __setattr__(self, name, value):
        # Writes have to reach the mechanism too: a runner toggling `retriever.verbose`
        # per query would otherwise set it on the wrapper, where nothing reads it, and
        # the diagnostic it was turning on would never appear.
        if name in RecordingRetriever._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)


def query_id(query: dict[str, Any]) -> str:
    """Stable identifier for a test question, independent of its position.

    Derived from the question text so the same question carries the same id in every
    method's results, which is what makes "did these two runs answer the same set?"
    answerable after the fact rather than assumed.
    """
    text = str(query.get("query", "")).strip()
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def select_queries(
    queries: list[dict[str, Any]],
    *,
    max_queries: int | None,
    tag_ids: bool = True,
) -> list[dict[str, Any]]:
    """Take the first ``max_queries`` questions and stamp each with its id.

    Every method must answer the same questions in the same order, or a difference
    between two curves is partly a difference in what was asked. The loaders already
    shuffle deterministically from ``data_seed``, so the shared step here is only the
    cut and the labelling.

    This alone does not make the sets equal, and it is not meant to: it makes them
    *checkable*. Two things still pull them apart upstream, by design in one case and
    by configuration in the other — a loader handed relevance-filter arguments builds
    its question set from those instead (the paper_faithful configs of the two vote
    methods do exactly that, reproducing Koga's protocol), and any dataset whose query
    count or corpus size differs between configs yields a different set. The
    ``query_id`` each question carries is what turns "did these two runs answer the
    same questions?" into something the results can be asked rather than assumed.
    """
    selected = [q for q in queries if str(q.get("query", "")).strip()]
    if max_queries is not None:
        selected = selected[: int(max_queries)]
    if tag_ids:
        selected = [{**q, "query_id": query_id(q)} for q in selected]
    return selected


def dataset_names(config: dict[str, Any]) -> list[str]:
    """Ordered dataset names a method config will evaluate.

    Mirrors the runners' selection: a top-level ``datasets:`` list, otherwise the
    singular ``dataset:`` block.
    """
    # Built from the same merge the runners use, so an entry that inherits its name
    # from the base block is scheduled rather than silently skipped.
    return [
        str(entry["name"])
        for entry in iter_dataset_configs(config)
        if entry.get("name")
    ]


def privacy_sweep(
    config: dict[str, Any], *, paper_key: str, default_workloads: tuple[int, ...] = (100,)
) -> tuple[str, list[float], list[int]]:
    """What a config asks to be swept, and in which units.

    The two config sets express a sweep differently and both have to keep working. The
    shared protocol names points on the common axis (``epsilon_targets``) and lets the
    runner derive the method's own knob for each of them; a paper-faithful config names
    that knob directly, in the units its paper published. Reading only the first turns a
    paper-faithful run into a silent sweep of whatever the default happens to be — the
    run finishes, writes results, and reproduces nothing.

    Returns ``("target", values, workloads)`` when the values are shared-axis targets to
    invert, or ``("knob", values, workloads)`` when they are the method's own budget and
    are used as they stand.
    """
    workloads = [int(q) for q in config.get("workload_queries", default_workloads)]
    targets = config.get("epsilon_targets")
    if targets:
        return "target", [float(t) for t in targets], workloads
    values = config.get(paper_key)
    if values:
        return "knob", [float(v) for v in values], workloads
    raise KeyError(
        f"config declares neither epsilon_targets nor {paper_key}; there is nothing to "
        "sweep, and defaulting would run a grid nobody asked for."
    )
