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

## The common evaluation protocol

Each method's paper reports it on its own datasets, its own generator, its own corpus
size and its own ε — and those ε values are not the same quantity. One is spent per
query, one over the whole corpus, one before any question is asked. Comparing the
published numbers compares six different experiments.

`experiments/configs/full/` is the answer to that: one protocol every method runs
under, so a difference in results is a difference between mechanisms. `paper_faithful/`
keeps each paper's own settings, for reproducing its table.

### What is held fixed

| Axis | Value | Why it cannot vary |
|---|---|---|
| Datasets | Natural Questions, TriviaQA, SearchQA, ChatDoctor, medical_dirichlet_phi3, MovieLens | A method evaluated on an easier corpus is not a better method |
| Corpus size | `max_index_docs: 20000` | Retrieval difficulty scales with the corpus; so does the fraction of it a per-document budget can afford |
| Questions | `max_test_queries: 40`, chosen by `select_queries` from `data_seed` | Every method answers the *same* questions in the *same* order, so utility differences are paired |
| Generators | The same 7 models, ≤4B, swept identically | |
| Embedder | `Snowflake/snowflake-arctic-embed-m-v1.5`, cosine over L2-normalised vectors | Retrieval quality would otherwise be a confound |
| Chunking | `enabled: false`, `max_document_words: 400` | The privacy unit is one document; splitting it reaches the document only through group privacy, whose δ grows exponentially in the number of chunks |
| Answer length | 50 tokens wherever a method can express one | Every text metric is monotone in answer length |
| Query filters | none | Some papers evaluate only on well-supported questions, precisely where DP degrades least |
| δ | `DEFAULT_DELTA = 1e-2` on the shared axis | See below |

### The shared axis

The one number that is comparable across methods is `epsilon_workload`: the privacy
loss over a workload of Q questions, at the common δ, with every method's mechanisms
composed in one family (`metrics/workload_accounting.py`).

Each method declares its mechanism inventory — a pure-DP release, an exponential
mechanism, a Gaussian, a zCDP budget — and an additive δ where its mechanism has one
that no Rényi curve can absorb (the Limited-Domain Gumbel's bottom-out symbol, DP-KSA's
PTR false release). Those compose as RDP curves plus summed δ, and convert to (ε,δ)
exactly once, at the end. Nothing is ever round-tripped through a foreign family.

Every record therefore carries:

| field | what it answers |
|---|---|
| `epsilon` | the method's own knob — reproduces its paper, comparable to nothing |
| `epsilon_workload` | the shared axis at the run's Q — **the only cross-method number** |
| `epsilon_workload_basic` | the same under basic composition; a ranking that holds under both is not an artefact of the accountant |
| `epsilon_workload_by_q` | the whole curve, Q ∈ {1, 5, 10, 25, 50, 100} |
| `workload_error` | why a method has no place on the axis, when it has none |

`epsilon_workload_by_q` is computed from mechanisms already in hand, so no point on it
costs an extra query. Utility is always the mean over every question asked, at every Q.
That separation is the point: **Q is a property of the guarantee, not of the
measurement.**

Three regimes (`METHOD_REGIMES`) describe how a method's cost scales:

- `per_query` — pays again for every question (DP-RAG, both vote methods, DP-KSA)
- `corpus_total` — a per-document budget filter already covers the stream (MuRAG)
- `offline` — spent before any question is asked (DP-SynRAG)

At Q=1 all six sit within a factor of two of each other, which is why the published
numbers look comparable. At Q=100 they separate by two orders of magnitude, and the
ranking inverts. That inversion is the benchmark's central result.

### Why δ = 1e-2

Two of the mechanisms carry an additive δ per query that no composition can shrink.
Over a hundred-question workload each spends about 5e-3, so a common δ of 1e-5 is
unreachable — not tight, *unreachable*. 1e-2 is the floor at which every method can be
placed on one axis at all.

It is a weak guarantee for a 20 000-document corpus, roughly 200× the conventional
1/n, and that is itself a finding: no source paper composes δ over a query stream, so
none of them has to confront it.

### Deriving the ε grids

The grids in the configs are not each paper's operating points. They are the values
whose `epsilon_workload` lands on {1, 3, 10, 30, 100} at Q=100, obtained by inverting
the shared axis through the exact mechanism builders the runners charge:

```bash
python analysis/derive_epsilon_grids.py --targets 1 3 10 30 100 --queries 100
```

Every derived value is verified by recomposing it (the printed `ratio` must be 1.0),
and `tests/test_shared_grid.py` re-checks each shipped grid against its target, that
the grid is monotone, and that the answer length is constant across the sweep.

A budget knob that also decides the answer length would confound the two, so wherever
a method exposes both, the per-token budget is **derived** from the swept per-query
one — `epsilon_token_from_budget` for the vote methods, `epsilon_0_from_budget` for
MuRAG, `epsilon_targets` for DP-SynRAG's four zCDP stages. Every point of a sweep then
releases the same number of tokens and differs only in noise.

**Composability is not feasibility.** A grid can compose correctly, verify against the
axis, and still be one the mechanism cannot execute: the Limited-Domain Gumbel refuses
to release once its bottom-out threshold passes the vote count, and it does so without
raising. `metrics/privacy_accounting_autodp.py::ld_gumbel_feasible` reports that, and
the feasibility of every shipped point is asserted in the test suite so the state is a
decision rather than a discovery mid-run.

### Seeds

`data_seed` picks the corpus and the questions; `run_seed` drives the DP noise. A
repetition (`--seeds 1 2 3`) therefore varies the mechanism without resampling or
rebuilding the index, and the spread across seeds measures the mechanism rather than
the sample. Results land in `<method>/seed_<n>/`.

### Reading the results

`metrics_summary.csv` carries, beyond the utility metrics:

| column | for what |
|---|---|
| `answer_rate` | separates "did not answer" from "answered wrongly" — several methods return an empty string when their budget runs out, and averaged into the overlap metrics those look like ordinary mistakes |
| `retrieval_precision_at_k` | overlap with the ranking the method would have produced had it not been private. Not relevance: there is no ground truth for which documents a question needed |
| `private_retrieval` | whether that overlap means anything. Three methods protect only the generation and retrieve with a plain cosine search, so their 1.0 is an identity, not an achievement |
| `query_id` | lets you verify two methods answered the same set |
| `workload_error` | why a method is absent from the axis |

DP-SynRAG has no `retrieval_precision_at_k`: it does not retrieve from the corpus, it
replaces it, so document ids do not map. The column is blank rather than 0, which would
read as total retrieval failure.

---

## Adding a method to the protocol

A new method joins the comparison by satisfying the same contract as the other eight.
In order:

**1. Implement the mechanism** under `dp_methods/<per_query|offline>/<name>/`, using
the `core/` interfaces. Keep the mechanism free of benchmarking concerns — no config
reading, no result writing.

**2. Declare what it spends.** Add a builder to `metrics/workload_accounting.py` that
returns the mechanism inventory for **one query**:

```python
def mymethod_mechanisms(*, epsilon_token: float, max_tokens: int) -> list[Mechanism]:
    """Ceilings, never realised counts: an answer that stops early spends less, but
    the guarantee has to hold for the longest answer a neighbouring corpus could
    produce."""
    return [exponential_mechanism("vote", epsilon_token, count=max_tokens)]
```

Register it in `METHOD_MECHANISMS`, and its scaling in `METHOD_REGIMES`. Charging a
data-dependent quantity is the classic error here — if how much you charge depends on
what the corpus turned out to be, the number is not a bound.

**3. Write the runner** as `experiments/run_<name>.py`, using the shared helpers from
`experiments/common.py` rather than reimplementing them:

- `resolve_seeds(config)` → `(data_seed, run_seed)`
- `load_dataset_from_config(ds_cfg, data_seed)`
- `select_queries(queries, max_queries=...)` — this is what makes the comparison paired
- `build_chunker(config)` — the single chunking policy
- `RecordingRetriever(...)` — makes what was retrieved observable without touching the mechanism
- `nonprivate_ranking(indexer, qvec, k)` — the gold set for retrieval overlap
- `workload_accounting_block(name, queries=..., run_seed=..., **params)` — the shared axis

Each result item needs `query_id`, `prediction`, `reference`, `answers` (alias lists),
`retrieved_ids`, `nonprivate_retrieved_ids`, and `favorites` for set-valued golds. The
payload root needs the accounting block. `workload_accounting_block` returns failures
as data rather than raising, because losing a finished run to a diagnostic is the worst
possible trade.

**4. Register it** in `experiments/runner.py::METHODS` with its script and its three
config paths (`full_config`, `paper_faithful_config`, `smoke_config`), and add it to
`DEFAULT_METHODS`.

**5. Write `experiments/configs/full/<name>.yaml`** by copying the shared blocks from
any existing one — datasets, `llms`, `models.embedder`, `chunking` are identical across
all methods and are meant to stay that way. Only the method's own block differs.

**6. Derive its grid.** Add the method to `analysis/derive_epsilon_grids.py` so its
knob is inverted from the shared targets rather than chosen, then paste the result into
the config. Add it to `tests/test_shared_grid.py`, which will fail if the two drift.

**7. Document the deviations** in `dp_methods/**/<name>/README.md`: corpus size, answer
length, the grid, and anything the shared protocol overrides that the paper specified.
This is what makes a number reproducible by someone reading only the repo.

**8. Verify:**

```bash
python scripts/make_smoke_configs.py
python experiments/runner.py --smoke --seeds 1 2
python scripts/verify_smoke.py results/run_<stamp>
```

`verify_smoke.py` checks what a finishing run does not: that the new method answered
the same questions as the others, that its two seeds differ in the mechanism and agree
on the sample, that its budget knob moves it along the shared axis, that the fields the
analysis reads are present, and that an empty answer is reported as empty.

Two seeds that produce identical text are not by themselves a fault: a small model
collapses onto one phrase, every voter then proposes the same token, and no noise draw
can move a single-spike argmax. Under `BENCHDP_NOISE_FINGERPRINT=1` — which `--smoke`
sets for itself — the vote methods record a `noise_fingerprint` per query: a hash of the
first sample their mechanism drew, which depends only on the seed and public parameters.
Identical text with differing fingerprints is a note; the failure is reserved for
fingerprints that repeat across seeds, which is the real "`run_seed` never reached the
noise". Without the fingerprint the check says it cannot decide rather than blaming the
seed.

The variable is **off by default and measurement runs must leave it off**: the value is
derived from the mechanism's own noise, and at a publicly known scale a hashed sample is
narrow enough to brute-force back to its preimage. `core/noise_trace.py` spells the
argument out.

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
under `experiments/configs/`, split into three sets:

- `experiments/configs/full/<method>.yaml` — **the common protocol**: the shared
  datasets, generators, corpus size, answer length and ε grid described above. This
  is the set the cross-method comparison is computed from.
- `experiments/configs/paper_faithful/<method>.yaml` — that method's own paper
  settings, for reproducing its published table.
- `experiments/configs/smoke/<method>.yaml` — the common protocol shrunk to a few
  minutes, for verifying that a method runs end to end and honours the seeds.
  Generated from `full/` by `scripts/make_smoke_configs.py`, so the two cannot
  drift; it is deliberately too small to time the real grid or calibrate a
  threshold.

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

# Repetitions: same corpus and questions, different mechanism noise
python experiments/runner.py --seeds 1 2 3

# Smoke: every method, two seeds, a few minutes — then check the invariants
python experiments/runner.py --smoke --seeds 1 2
python scripts/verify_smoke.py results/run_<stamp>
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
- `results/figures/tradeoff_eps_vs_s2mia_auc_<dataset>.{png,pdf}`
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
