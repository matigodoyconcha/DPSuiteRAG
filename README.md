# DP-Benchmark-RAG

A modular, reproducible framework for benchmarking Differential Privacy (DP) methods
in Retrieval-Augmented Generation (RAG) systems — privacy–utility trade-off curves
across a range of ε values.

> The full implementation specification lives in
> [`benchdp_rag_agent_guide.md`](benchdp_rag_agent_guide.md) — the **official base**
> of this project. This README is the quick-start reference.

## Status

All six DP methods, the shared retrieval/scoring stack, and the central runner are
implemented; the project is in its experiment-running and analysis phase.

| Component | Status |
|---|---|
| Abstract interfaces `core/` + `RAGPipeline` | ✅ |
| Vanilla end-to-end pipeline (no-DP baseline) | ✅ |
| Datasets (NQ, TriviaQA, HotpotQA, SQuAD, ChatDoctor, SearchQA, MovieLens, medical\_dirichlet\_phi3) | ✅ |
| Utility / privacy metrics + privacy accounting (RDP / zCDP / autodp) | ✅ |
| **DP-RAG** (DP-ICL) | ✅ |
| **DPVoteRAG / DPSparseVoteRAG** (Limited Domain Gumbel + SVT) | ✅ |
| **DP-SynRAG** (DP synthetic corpus, zCDP) | ✅ |
| **DP-KSA** (keyword selection + PTR, RDP) | ✅ |
| **PRIVATE-RAG / MURAG + MURAG-ADA** (multi-query individual privacy filter) | ✅ |
| Central runner `experiments/runner.py` | ✅ |
| Analysis module (`load_results`, `summary_table`, `plot_tradeoff`) | ✅ |

## References (Papers & Reference Implementations)

Each method is faithful to a published paper and, where available, an official
reference repository:

| Method | Paper | Reference repo |
|---|---|---|
| **DP-RAG** | Grislain et al., *RAG with Differential Privacy*, [arXiv:2412.19291v2](https://arxiv.org/abs/2412.19291) (Sarus Technologies, 2024) | [sarus-tech/dp-rag](https://github.com/sarus-tech/dp-rag) |
| **DPVoteRAG / DPSparseVoteRAG** | Koga et al., *Privacy-Preserving Retrieval-Augmented Generation with Differential Privacy*, [arXiv:2412.04697v3](https://arxiv.org/abs/2412.04697) | [tacchan7412/DPRAG](https://github.com/tacchan7412/DPRAG) |
| **DP-SynRAG** | *DP-SynRAG: Differentially Private RAG via Synthetic Document Generation*, [arXiv:2510.06719v2](https://arxiv.org/abs/2510.06719) (2025) | — (no official code released) |
| **DP-KSA** | Tang et al., *Differentially Private Retrieval-Augmented Generation*, [arXiv:2602.14374](https://arxiv.org/abs/2602.14374) (USC, 2026) | — (no official code released) |
| **PRIVATE-RAG / MURAG** | Wu et al., *Private-RAG: Answering Multiple Queries with LLMs while Keeping Your Data Private*, [arXiv:2511.07637](https://arxiv.org/abs/2511.07637) (2025) | — (no official code released) |

Sub-mechanisms used by the methods above:

- **Limited Domain Gumbel** — Durfee & Rogers, *Practical Differentially Private
  Top-k Selection with Pay-what-you-get Composition*, NeurIPS 2019.
- **Sparse Vector Technique** — Dwork & Roth, *The Algorithmic Foundations of
  Differential Privacy* (2014).
- **(ε,δ) accounting for DPVoteRAG** — ported from
  `tacchan7412/DPRAG/utils/privacy.py` (advanced + sequential composition).
- **RDP composition for DP-KSA** — FindBestK (exponential mechanism) ⊕
  TopKWithPTR (Gaussian Propose-Test-Release), converted to (ε,δ)-DP
  (`metrics/privacy_accounting_dpksa.py`).
- **Individual privacy filter for PRIVATE-RAG** — per-document budget
  `E(z) = M·ε_q`, charged only when a document is retrieved; pure DP (δ = 0).

## Benchmark design: standardization & deviations from the papers

To keep the privacy–utility curves comparable *across* methods, a few components
are held fixed for every method instead of reproduced per paper. These are
deliberate benchmarking choices, each documented in the relevant
`dp_methods/**/README.md`:

- **Retriever / embedder — standardized.** Every method retrieves with one shared
  embedder, `Snowflake/snowflake-arctic-embed-m-v1.5` (cosine similarity via
  `CosineSimilarityRetriever`, over the same FAISS index). This is DP-RAG's native
  retriever; for methods whose papers used a different one — e.g.
  DPVoteRAG/DPSparseVoteRAG retrieve with an NQ-supervised DPR encoder — this is a
  deliberate deviation so retrieval quality is not a confound when comparing
  methods. The `paper_faithful/` configs restore each paper's **datasets and
  LLM(s)** but keep this shared embedder.
- **Scoring — standardized.** Utility uses SQuAD-normalized F1 / exact-match
  (article- and punctuation-insensitive, alias- and multi-reference-aware), so
  numbers line up in magnitude with the papers that use the same convention.
- **No paper-specific test-set pre-filtering.** Some papers evaluate on a curated
  subset — e.g. DPVoteRAG (§IV-A) keeps only NQ/TriviaQA questions backed by 20–59
  relevant corpus documents, precisely because DP degrades on thinly-supported
  questions. This benchmark does **not** apply such filters: realistic query
  streams include thinly-supported questions, and a DP method's degradation on
  them is part of what is measured. See
  [`dp_methods/per_query/dpvoterag/README.md`](dp_methods/per_query/dpvoterag/README.md)
  for the full rationale.

Consequence: results here are not directly comparable to each paper's headline
table; they are comparable *to each other* under one fixed retrieval + scoring
setup.

## Requirements

- Python 3.10+
- Active virtual environment

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Running Experiments

Each method has a runner script `experiments/run_<method>.py` and YAML configs
under `experiments/configs/`, split into two folders:

- `experiments/configs/full/<method>.yaml` — the broad sweep (≤4B-parameter LLMs ×
  the harmonized dataset set).
- `experiments/configs/paper_faithful/<method>.yaml` — only that method's paper
  LLM(s) and datasets (the shared embedder and scoring are kept).

DP runners accept only `--config`, `--verbose`, and `--verbose-max-examples`; all
experiment parameters (dataset, LLM, ε, etc.) live in the YAML.

### Central runner (all methods)

```bash
# Full sweep for every method, into a timestamped results/run_<stamp>/ directory
python experiments/runner.py

# Paper-faithful sweep (each method's paper LLM(s) + datasets)
python experiments/runner.py --paper-faithful

# A subset of methods
python experiments/runner.py --methods dp_rag dpsparsevoterag private_rag
```

### Individual runners

```bash
# Vanilla baseline (no DP)
python experiments/run_vanilla.py --config experiments/configs/full/vanilla.yaml

# DP-RAG  (sweeps the ε grid from the YAML)
python experiments/run_dp_rag.py --config experiments/configs/full/dp_rag.yaml

# DPVoteRAG / DPSparseVoteRAG  (independent scripts)
python experiments/run_dpvoterag.py       --config experiments/configs/full/dpvoterag.yaml
python experiments/run_dpsparsevoterag.py --config experiments/configs/full/dpsparsevoterag.yaml

# DP-SynRAG  (ε is an OUTPUT of the zCDP accountant, not swept)
python experiments/run_dp_synrag.py --config experiments/configs/full/dp_synrag.yaml

# DP-KSA  (sweeps the ε grid; noise calibrated to FindBestK + PTR)
python experiments/run_dpksa.py --config experiments/configs/full/dpksa.yaml

# PRIVATE-RAG / MURAG  (private_rag_ada.yaml is the MURAG-ADA variant)
python experiments/run_private_rag.py --config experiments/configs/full/private_rag.yaml
python experiments/run_private_rag.py --config experiments/configs/full/private_rag_ada.yaml
```

Swap `full/` for `paper_faithful/` to run the paper-faithful variant of any config.
Results are written to `results/<method>/...` (not committed).

## Analysis

```bash
# Build the summary table (CSV + LaTeX + console)
python -m analysis.summary_table \
    --datasets natural_questions triviaqa \
    --no-recompute-vanilla

# Generate trade-off figures (PNG + PDF in results/figures/)
python -m analysis.plot_tradeoff \
    --datasets natural_questions
```

Both scripts accept `--results-dir`, `--datasets`, `--methods`, and
`--no-recompute-vanilla`. The `--bertscore` flag activates BERTScore F1 for the
vanilla baseline recompute (off by default; requires a heavy model download).

Output files:
- `results/summary_table.csv` / `results/summary_table.tex`
- `results/figures/tradeoff_eps_vs_match_accuracy_<dataset>.{png,pdf}`
- `results/figures/tradeoff_eps_vs_rouge_l_<dataset>.{png,pdf}`
- `results/figures/tradeoff_eps_vs_mia_auc_bleu_<dataset>.{png,pdf}`
- `results/figures/pareto_privacy_utility_<dataset>.{png,pdf}`

## Dataset, LLM, and Split Configuration

Everything is controlled via the method YAML (`dataset`/`datasets`, `llm`/`llms`,
`models`, `pipeline`, `generation`/`inference` blocks):

- `dataset.name`: `squad | chatdoctor | natural_questions | triviaqa | hotpotqa |
  searchqa | movielens | medical_dirichlet_phi3`
- `dataset.index_split` / `test_split`: which portion is indexed vs evaluated.
- `dataset.max_index_docs` / `max_test_queries` (or `max_samples`): size caps
  for quick test runs.
- `dataset.docs_path` / `queries_path`: custom documents/queries
  (`.txt/.json/.jsonl`; queries formatted as `{"query": ..., "answer": ...}`).
- `llm.name` (provider, currently `huggingface`) and `llm.model_name`;
  `models.embedder`. Top-level `datasets:` / `llms:` lists sweep several of each
  in one config.

### CLI Overrides — vanilla only

```bash
python experiments/run_vanilla.py \
    --config experiments/configs/full/vanilla.yaml \
    --dataset squad \
    --llm-name huggingface \
    --llm-model sshleifer/tiny-gpt2 \
    --embedder-model sentence-transformers/all-MiniLM-L6-v2 \
    --index-split train --test-split validation \
    --max-index-docs 300 --max-test-queries 50 \
    --k 3 --max-new-tokens 64 \
    --output results/vanilla/vanilla_custom_run.json
```

## Tests

```bash
pytest tests/ -v
```
