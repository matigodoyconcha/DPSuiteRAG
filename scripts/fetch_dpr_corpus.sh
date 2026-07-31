#!/usr/bin/env bash
# Fetch the DPR passage corpus and the relevance files the question filters need.
#
# Without data/dpr/ two things happen silently: Natural Questions and TriviaQA fall
# back to a degenerate corpus built from their own question-answer pairs, and the
# relevance-binning filter the vote methods declare is skipped with a warning that
# never reaches the result files. Both make those datasets' numbers unusable while
# looking like ordinary runs.
#
# Sizes: psgs_w100.tsv.gz is ~13 GB compressed and ~21M passages uncompressed; the two
# biencoder files are a few hundred MB. The relevance files alone are enough to enable
# the question filters and the redundancy stratification, so they are fetched first and
# the corpus is opt-in.
#
# Usage:
#   scripts/fetch_dpr_corpus.sh              # relevance files only
#   scripts/fetch_dpr_corpus.sh --corpus     # also the ~13 GB passage corpus
set -euo pipefail

DATA_DIR="${BENCHDP_DPR_DIR:-data/dpr}"
BASE="https://dl.fbaipublicfiles.com/dpr"
WANT_CORPUS=0
[[ "${1:-}" == "--corpus" ]] && WANT_CORPUS=1

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

fetch() {
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then
    echo "  already present: $out"
    return
  fi
  echo "  downloading $out"
  curl -fL --retry 3 --retry-delay 5 -o "$out.part" "$url"
  mv "$out.part" "$out"
}

echo "Relevance files (question filters and redundancy bins):"
fetch "$BASE/data/retriever/biencoder-nq-dev.json.gz" biencoder-nq-dev.json.gz
fetch "$BASE/data/retriever/biencoder-trivia-dev.json.gz" biencoder-trivia-dev.json.gz
for f in biencoder-nq-dev biencoder-trivia-dev; do
  [[ -f "$f.json" ]] || { echo "  decompressing $f"; gunzip -k "$f.json.gz"; }
done

if [[ "$WANT_CORPUS" -eq 1 ]]; then
  echo "Passage corpus (~13 GB compressed, ~21M passages):"
  fetch "$BASE/wikipedia_split/psgs_w100.tsv.gz" psgs_w100.tsv.gz
  [[ -f psgs_w100.tsv ]] || { echo "  decompressing psgs_w100"; gunzip -k psgs_w100.tsv.gz; }
else
  echo
  echo "Passage corpus skipped. Re-run with --corpus to fetch it (~13 GB)."
  echo "Until then Natural Questions and TriviaQA use their degenerate fallback corpus."
fi

echo
echo "Contents of $DATA_DIR:"
ls -lh
