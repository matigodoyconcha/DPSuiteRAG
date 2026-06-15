# DP-Benchmark-RAG

A modular, reproducible framework for benchmarking Differential Privacy (DP) methods
in Retrieval-Augmented Generation (RAG) systems — privacy–utility trade-off curves
across a range of ε values.

> The full implementation specification lives in
> [`benchdp_rag_agent_guide.md`](benchdp_rag_agent_guide.md) — the **official base**
> of this project. Pending components are built as described there; implemented
> components are documented there as they stand in the code. This README is the
> quick-start reference.

## Status

| Component | Status |
|---|---|
| Abstract interfaces `core/` + `RAGPipeline` | ✅ |
| Vanilla end-to-end pipeline | ✅ |
| Datasets (NQ, TriviaQA, HotpotQA, SQuAD, ChatDoctor, SearchQA, MovieLens, medical\_synth, medical\_dirichlet\_phi3) | ✅ |
| Utility / privacy metrics + privacy accounting (RDP / zCDP / autodp) | ✅ |
| **DP-RAG** (DP-ICL) | ✅ |
| **DPVoteRAG / DPSparseVoteRAG** (Limited Domain Gumbel + SVT) | ✅ |
| **DP-SynRAG** (DP synthetic corpus, zCDP) | ✅ |
| **PRIVATE-RAG** | 🚧 pending |
| Central runner `experiments/runner.py` | 🚧 pending |
| **Analysis module** (`load_results`, `summary_table`, `plot_tradeoff`) | ✅ |

## References (Papers & Reference Implementations)

Each implemented method is faithful to a published paper and, where available,
an official reference repository:

| Method | Paper | Reference repo |
|---|---|---|
| **DP-RAG** | Grislain et al., *RAG with Differential Privacy*, [arXiv:2412.19291v2](https://arxiv.org/abs/2412.19291) (Sarus Technologies, 2024) | [sarus-tech/dp-rag](https://github.com/sarus-tech/dp-rag) |
| **DPVoteRAG / DPSparseVoteRAG** | Koga et al., *Privacy-Preserving Retrieval-Augmented Generation with Differential Privacy*, [arXiv:2412.04697v3](https://arxiv.org/abs/2412.04697) | [tacchan7412/DPRAG](https://github.com/tacchan7412/DPRAG) |
| **DP-SynRAG** | *DP-SynRAG: Differentially Private RAG via Synthetic Document Generation*, [arXiv:2510.06719v2](https://arxiv.org/abs/2510.06719) (2025) | — (no official code released) |

Sub-mechanisms used by the methods above:

- **Limited Domain Gumbel** — Durfee & Rogers, *Practical Differentially Private
  Top-k Selection with Pay-what-you-get Composition*, NeurIPS 2019.
- **Sparse Vector Technique** — Dwork & Roth, *The Algorithmic Foundations of
  Differential Privacy* (2014).
- **(ε,δ) accounting for DPVoteRAG** — ported from
  `tacchan7412/DPRAG/utils/privacy.py` (advanced + sequential composition).

Pending methods (**PRIVATE-RAG**) do not yet have a
paper/repo fixed in this codebase — see [`benchdp_rag_agent_guide.md`](benchdp_rag_agent_guide.md) §4.

## Requirements

- Python 3.10+
- Active virtual environment

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Running Experiments

Each method has its own runner script `experiments/run_<method>.py` and a YAML
config in `experiments/configs/`. DP runners accept only `--config`, `--verbose`,
and `--verbose-max-examples`; all experiment parameters (dataset, LLM, ε, etc.)
live in the YAML. Use the `*_quick.yaml` variants for smoke runs.

```bash
# Vanilla baseline (no DP)
python experiments/run_vanilla.py --config experiments/configs/vanilla.yaml

# DP-RAG  (sweeps the ε grid from the YAML)
python experiments/run_dp_rag.py --config experiments/configs/dp_rag.yaml

# DPVoteRAG / DPSparseVoteRAG  (independent scripts)
python experiments/run_dpvoterag.py       --config experiments/configs/dpvoterag.yaml
python experiments/run_dpsparsevoterag.py --config experiments/configs/dpsparsevoterag.yaml

# DP-SynRAG  (ε is an OUTPUT of the zCDP accountant, not swept)
python experiments/run_dp_synrag.py --config experiments/configs/dp_synrag.yaml
```

Results are written to `results/<method>/...` (not committed). Analysis and figures:

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

Everything is controlled via the method YAML (`dataset`, `llm`, `models`,
`pipeline`, `generation`/`inference` blocks):

- `dataset.name`: `squad | chatdoctor | natural_questions | triviaqa | hotpotqa |
  searchqa | movielens | medical_synth | medical_dirichlet_phi3`
- `dataset.index_split` / `test_split`: which portion is indexed vs evaluated.
- `dataset.max_index_docs` / `max_test_queries` (or `max_samples`): size caps
  for quick test runs.
- `dataset.docs_path` / `queries_path`: custom documents/queries
  (`.txt/.json/.jsonl`; queries formatted as `{"query": ..., "answer": ...}`).
- `llm.name` (provider, currently `huggingface`) and `llm.model_name`; `models.embedder`.

### CLI Overrides — vanilla only

```bash
python experiments/run_vanilla.py \
    --config experiments/configs/vanilla.yaml \
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

## Next Steps

Implement the pending methods following
[`benchdp_rag_agent_guide.md`](benchdp_rag_agent_guide.md) §4:
PRIVATE-RAG (5.5) → central runner (6.1) →
full experiments and figures.
