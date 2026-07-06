#!/usr/bin/env python
"""Download the DPR ``psgs_w100`` Wikipedia passage corpus (facebookresearch/DPR).

Fetches the official public file used by Dense Passage Retrieval (Karpukhin et
al., 2020) — ~21M 100-word Wikipedia passages — and decompresses it to the
location the QA loaders read by default (:data:`bench_datasets.dpr_corpus.
DEFAULT_DPR_CORPUS_PATH`, i.e. ``data/dpr/psgs_w100.tsv``).

    python scripts/download_dpr_corpus.py

The gzipped download is ~4.7 GB and the decompressed TSV is ~13 GB, so run this
on the machine that will build the index (not committed to git). Re-running
skips steps whose output already exists.
"""
from __future__ import annotations

import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

# Official DPR public mirror (dpr/data/download_data.py:
# resource "data.wikipedia_split.psgs_w100").
DPR_URL = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEST_DIR = _PROJECT_ROOT / "data" / "dpr"
GZ_PATH = DEST_DIR / "psgs_w100.tsv.gz"
TSV_PATH = DEST_DIR / "psgs_w100.tsv"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = 100.0 * done / total_size
    sys.stdout.write(f"\r  downloading… {done / 1e9:.2f}/{total_size / 1e9:.2f} GB ({pct:5.1f}%)")
    sys.stdout.flush()


def main() -> int:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    if TSV_PATH.is_file():
        print(f"[skip] decompressed corpus already present: {TSV_PATH} "
              f"({TSV_PATH.stat().st_size / 1e9:.1f} GB)")
        return 0

    if not GZ_PATH.is_file():
        print(f"[download] {DPR_URL}\n  -> {GZ_PATH}")
        try:
            urllib.request.urlretrieve(DPR_URL, GZ_PATH, reporthook=_progress)
            print()
        except Exception as exc:  # network / URL failure — surface, don't half-write
            if GZ_PATH.exists():
                GZ_PATH.unlink()
            print(f"\n[error] download failed: {exc}", file=sys.stderr)
            return 1
    else:
        print(f"[skip] gzip already present: {GZ_PATH}")

    print(f"[decompress] {GZ_PATH} -> {TSV_PATH}")
    with gzip.open(GZ_PATH, "rb") as fin, TSV_PATH.open("wb") as fout:
        shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)

    print(f"[done] {TSV_PATH} ({TSV_PATH.stat().st_size / 1e9:.1f} GB)")
    print("Natural Questions and TriviaQA will now retrieve from this corpus by default.")
    print(f"You may delete the gzip to save space: {GZ_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
