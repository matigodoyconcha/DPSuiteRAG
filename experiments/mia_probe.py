"""Online driver for the membership inference attacks (MIA).

This runs the *standalone* MIA evaluation: it is deliberately decoupled from the
utility runs so that whether a document is indexed or held out never affects the
methods' utility numbers. A method runner enters MIA mode when
``BENCHDP_MIA_ONLY=1`` (set by ``experiments/runner.py --mia-only``); it then

  1. asks the dataset for a member / non-member split (``get_mia_split``),
  2. indexes only the member corpus (``split["index_documents"]``) through the
     method's own pipeline (same DP mechanism, same LLM),
  3. issues two fresh probe queries per target document, and
  4. saves the raw per-probe answers.

The AUCs are computed offline from those answers in ``metrics/mia.py`` (invoked by
``experiments/score_results.py``), mirroring the deferred-metric design used for
the utility metrics. See ``metrics/mia.py`` for the two attacks (S2MIA, RAG-MIA)
and the scoring.
"""

from __future__ import annotations

import os
from typing import Callable


def mia_only_enabled() -> bool:
    """Whether this process should run the standalone MIA instead of utility QA."""
    return os.environ.get("BENCHDP_MIA_ONLY", "0") == "1"


def resolve_mia_params(config: dict) -> dict:
    """Read the ``mia:`` config block, applying defaults.

    Keys: ``enabled`` (default True), ``n_per_class`` (100), ``index_size`` (2000,
    ``null`` to index every member document), ``seed`` (42) and
    ``anderson_doc_maxlen`` (512, truncation of the target document embedded in the
    Yes/No probe prompt so long documents do not overflow the context).
    """
    cfg = dict(config.get("mia", {}) or {})
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "n_per_class": int(cfg.get("n_per_class", 100)),
        "index_size": (None if cfg.get("index_size", 2000) is None else int(cfg.get("index_size", 2000))),
        "seed": int(cfg.get("seed", 42)),
        "anderson_doc_maxlen": int(cfg.get("anderson_doc_maxlen", 512)),
    }


def anderson_probe_prompt(doc: str, maxlen: int = 512) -> str:
    """The RAG-MIA (Anderson et al.) black-box Yes/No membership question.

    The target document is truncated to ``maxlen`` characters so a long document
    does not overflow the prompt; the wording follows the paper's best-performing
    prompt.
    """
    target = str(doc)[:maxlen]
    return (
        f'Does this : "{target}" appear in the context? '
        "Answer with Yes or No."
    )


def run_mia_probes(
    answer_fn: Callable[[str], str],
    split: dict,
    anderson_doc_maxlen: int = 512,
    on_probe: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Issue the S2MIA and RAG-MIA probe queries for every target in ``split``.

    Args:
        answer_fn: maps one query string to the method's answer string (a closure
            over the live pipeline for the current DP cell).
        split: as returned by ``BaseDataset.get_mia_split`` (``members`` /
            ``non_members`` lists of ``{query, gold, doc}``).
        anderson_doc_maxlen: truncation for the document in the Yes/No prompt.
        on_probe: optional ``(done, total)`` progress callback.

    Returns:
        One record per target document with the raw answers (scored offline):
        ``{is_member, query, gold, doc, s2mia_answer, anderson_answer}``.
    """
    targets: list[tuple[bool, dict]] = [
        (True, item) for item in split.get("members", [])
    ] + [
        (False, item) for item in split.get("non_members", [])
    ]
    total = len(targets)

    records: list[dict] = []
    for done, (is_member, item) in enumerate(targets, start=1):
        query = str(item.get("query", ""))
        gold = str(item.get("gold", ""))
        doc = str(item.get("doc", ""))

        # S2MIA: answer the query part, compare (offline) to the held-out part.
        s2mia_answer = answer_fn(query)
        # RAG-MIA (Anderson): ask whether the document is in the context.
        anderson_answer = answer_fn(anderson_probe_prompt(doc, anderson_doc_maxlen))

        records.append(
            {
                "is_member": is_member,
                "query": query,
                "gold": gold,
                "doc": doc,
                "s2mia_answer": str(s2mia_answer),
                "anderson_answer": str(anderson_answer),
            }
        )
        if on_probe is not None:
            on_probe(done, total)

    return records
