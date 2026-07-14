#!/usr/bin/env python
"""Download the DPR retriever dev files that back the relevance-binning filter.

Fetches the two official DPR retriever dev sets (Karpukhin et al., 2020) and
decompresses them to the locations the QA loaders read for the relevance-binning
query filter (:data:`bench_datasets.dpr_relevance.DEFAULT_NQ_RELEVANCE_PATH` and
:data:`bench_datasets.dpr_relevance.DEFAULT_TRIVIA_RELEVANCE_PATH`, i.e.
``data/dpr/biencoder-nq-dev.json`` and ``data/dpr/biencoder-trivia-dev.json``).

    python scripts/download_dpr_relevance.py

Each decompressed file is a few hundred MB (a list of records with
``question`` + ``positive_ctxs``). Re-running skips files whose output already
exists.
"""
from __future__ import annotations

import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

# Official DPR public mirror (dpr/data/download_data.py: resources
# "data.retriever.nq-dev" and "data.retriever.trivia-dev").
DPR_BASE_URL = "https://dl.fbaipublicfiles.com/dpr/data/retriever"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEST_DIR = _PROJECT_ROOT / "data" / "dpr"

# (gzip source url, decompressed target path) for each dev file.
FILES = [
    (f"{DPR_BASE_URL}/biencoder-nq-dev.json.gz", DEST_DIR / "biencoder-nq-dev.json"),
    (f"{DPR_BASE_URL}/biencoder-trivia-dev.json.gz", DEST_DIR / "biencoder-trivia-dev.json"),
]


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = 100.0 * done / total_size
    sys.stdout.write(f"\r  downloading… {done / 1e6:.1f}/{total_size / 1e6:.1f} MB ({pct:5.1f}%)")
    sys.stdout.flush()


def _fetch(url: str, target: Path) -> int:
    if target.is_file():
        print(f"[skip] already present: {target} ({target.stat().st_size / 1e6:.1f} MB)")
        return 0

    gz_path = target.with_suffix(target.suffix + ".gz")
    print(f"[download] {url}\n  -> {gz_path}")
    try:
        urllib.request.urlretrieve(url, gz_path, reporthook=_progress)
        print()
    except Exception as exc:  # network / URL failure — surface, don't half-write
        if gz_path.exists():
            gz_path.unlink()
        print(f"\n[error] download failed: {exc}", file=sys.stderr)
        return 1

    print(f"[decompress] {gz_path} -> {target}")
    with gzip.open(gz_path, "rb") as fin, target.open("wb") as fout:
        shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
    gz_path.unlink()  # the decompressed JSON is what the loaders read
    print(f"[done] {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


def main() -> int:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    rc = 0
    for url, target in FILES:
        rc |= _fetch(url, target)
    if rc == 0:
        print("Natural Questions and TriviaQA can now enable the relevance-binning filter.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
