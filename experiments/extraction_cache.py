"""Reuse of DP-SynRAG's keyword extraction across privacy configurations.

Stage 1a runs the generator once per document — 20k greedy generations, the bulk of
an index build — and it is the last stage before any DP mechanism touches the data.
Its output depends on the model, the documents and the extractor's own settings, and
on nothing else: not the noise seed, not the budget, not the softmax sharpness. So one
extraction serves every point of a sweep over those, and the grid pays for it once per
(dataset, generator) instead of once per cell.

The boundary is the same one the privacy analysis draws. Everything cached here happens
before the noisy histogram; everything after it stays keyed on the seed and the budget,
where reusing an artefact across configurations would report one configuration's
epsilon over another's output.

The extraction is stored as the coordinates of its non-zeros. Held dense it is a
20000 x 234243 float32 array — 18.7 GB for 200k ones — so the matrix is rebuilt on load
rather than written out.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

#: Written only after the arrays are on disk, so a build killed mid-write is rebuilt
#: instead of being loaded back as a complete extraction.
_COMPLETE = ".extraction_complete"


def _documents_digest(documents: list[str]) -> str:
    h = hashlib.sha256()
    h.update(str(len(documents)).encode())
    for doc in documents:
        h.update(b"\x00")
        h.update(doc.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def extraction_cache_root() -> Path:
    """Where extractions live. Survives a run, unlike the per-run index cache."""
    env = os.environ.get("BENCHDP_EXTRACTION_CACHE")
    root = Path(env) if env else Path(tempfile.gettempdir()) / "benchdp_extractions"
    root.mkdir(parents=True, exist_ok=True)
    return root


class CachedKeywordExtractor:
    """Wraps a ``KeywordExtractor`` and remembers what it extracted.

    Delegates everything else, so the indexer cannot tell the difference and the
    stage's output is unchanged — a cache hit returns the same matrix the extractor
    would have produced.
    """

    def __init__(self, inner: Any, *, root: Path | None = None) -> None:
        self._inner = inner
        self._root = root or extraction_cache_root()

    def _fingerprint(self, documents: list[str]) -> str:
        inner = self._inner
        # The vocabulary is hashed rather than described: `min_word_len` and
        # `exclude_stopwords` reach it only through NLTK, and a different corpus
        # version behind the same settings is still a different extraction.
        vocab_digest = hashlib.sha256(
            "\x00".join(inner.vocab).encode("utf-8", "replace")
        ).hexdigest()[:16]
        key = {
            "model_name": getattr(inner, "model_name", ""),
            "K": int(inner.K),
            "max_new_tokens": int(inner.max_new_tokens),
            "prompt_template": getattr(inner, "prompt_template", ""),
            "substring_fallback": bool(getattr(inner, "substring_fallback", False)),
            "vocab": vocab_digest,
            "documents": _documents_digest(documents),
        }
        return hashlib.sha256(
            json.dumps(key, sort_keys=True).encode()
        ).hexdigest()[:16]

    def extract(self, documents: list[str]) -> np.ndarray:
        cache_dir = self._root / f"kw_{self._fingerprint(documents)}"
        marker = cache_dir / _COMPLETE
        coords = cache_dir / "coords.npz"

        if marker.exists() and coords.exists():
            data = np.load(coords)
            matrix = np.zeros((len(documents), self._inner.vocab_size), dtype=np.float32)
            matrix[data["rows"], data["cols"]] = 1.0
            print(f"[dp_synrag] keyword extraction served from {cache_dir}")
            return matrix

        matrix = self._inner.extract(documents)
        rows, cols = np.nonzero(matrix)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            coords, rows=rows.astype(np.int32), cols=cols.astype(np.int32)
        )
        marker.write_text("", encoding="utf-8")
        print(f"[dp_synrag] keyword extraction cached at {cache_dir}")
        return matrix

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)
