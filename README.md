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

## Paper and supplementary material

The complete privacy derivations of the five admitted mechanisms (document-level
neighbourhood, one Rényi accountant, every proof) are in
[`docs/DPSuiteRAG_supplementary.pdf`](docs/DPSuiteRAG_supplementary.pdf), the
supplementary material of *DPSuiteRAG: A Benchmarking Abstraction for Differentially
Private Retrieval-Augmented Generation* (WSDM 2027).

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
| Answer length | 30 tokens wherever a method can express one, recorded per cell as `answer_tokens` | Every text metric is monotone in answer length |
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

Reading every method at the same Q is what makes their numbers comparable: the ranking
at a single question is not the ranking over a workload, and the paper reports how it
changes along Q on each corpus.

### Why δ = 1e-2

The common δ is a choice of the protocol, not a limit of the accounting. The
derived-delta rule below splits the target over the releases of the workload, so it is
defined for any δ in (0, 1). A target of 1e-5, the order this literature reports, is
admissible, and its cost is utility: the conversion gives a larger ε for the same noise,
and the mechanisms that carry an additive δ per release release less often, since the
share of each release also sets their thresholds. Every result in this repository is
reported at 1e-2.

#### The derived-delta rule

An additive δ is charged once per *release*, so what the axis has to cover is not the
per-release figure but the spend over the whole workload. The protocol therefore fixes
the spend and derives the parameter, in `metrics/workload_accounting.py`:

```
delta_token   = MECHANISM_DELTA_SHARE * δ / (Q * T)        # one release per token
mechanism_δ   = MECHANISM_DELTA_SHARE * δ / (Q * fraction) # one PTR test per query
```

`MECHANISM_DELTA_SHARE` is **0.5**: half of the axis δ pays for the mechanism's own
bottom-out — the Limited-Domain Gumbel's null symbol, DP-KSA's PTR false release — and
half is left to the RDP → (ε, δ) conversion. The split is not a margin of comfort. The
conversion turns a Rényi curve into an (ε, δ) pair by trading one against the other, so
spending the whole axis δ additively leaves it nothing to convert on and there is no ε
to report at all.

What both forms buy is that `Q · T · delta_token` and `Q · fraction · mechanism_δ` are
the same constant, `5e-3`, in **every** cell of every grid: a Q=1 cell and a Q=100 cell
reported at one common δ are paying the same price for it. The derived value reaches
the generator as well as the accountant — `delta_token` sets the Gumbel's bottom-out
threshold `h_⊥`, `mechanism_δ` sets DP-KSA's PTR release threshold — so the mechanism
that ran and the ε that is reported are calibrated to the same number. Each payload
names the parameters it derived this way in `workload_derived_deltas`; the vote
methods' online per-query guard `delta_total` follows from the same value as
`2 · T · delta_token`, which is what one query's releases can actually spend.

`paper_faithful/` pins these deltas at each paper's own value instead, because no
source paper composes over a stream and none of them derives one.

### Deriving the ε grids

The grids in the configs are not each paper's operating points. Every config declares
`epsilon_targets: [1, 3, 5, 10, 50]` — points on the shared axis — and `workload_queries`,
and the runner inverts the axis through the exact mechanism builders it charges
(`budget_for_target`) to get the knob each (target, Q) cell runs at, so a target and
the value that implements it cannot drift apart. The same inversion is available
offline for inspecting a grid:

```bash
python analysis/derive_epsilon_grids.py --targets 1 5 10 50 --queries 100
```

Every derived value is verified by recomposing it (the printed `ratio` must be 1.0),
and `tests/test_shared_grid.py` re-checks each shipped grid against its target, that
the grid is monotone, and that the answer length is constant across the sweep.

A budget knob that also decides the answer length would confound the two, so wherever
a method exposes both, the per-token budget is **derived** from the swept per-query
one — `epsilon_token_from_budget` for the vote methods, `epsilon_0_from_budget` for
MuRAG, `epsilon_targets` for DP-SynRAG's four zCDP stages. Every point of a sweep then
releases the same number of tokens and differs only in noise.

#### DP-RAG's retrieval:generation split moves along the sweep

DP-RAG spends on two things, retrieving and generating, and the axis needs one scalar.
The config's `retrieval_ratio` supplies it: the retrieval budget per unit of *per-token*
generation budget, taken from the paper's own operating point (ε_retrieval = 0.5,
ε_generation = 5.0 over 70 tokens). It is a constant, and both parts of a cell's budget
scale together with it.

The split a reader actually interprets, `eps_r / (eps_r + eps_g)`, is not that constant.
`eps_g` is a PLD composition of `T` copies of the per-token budget and is *sublinear* in
it, so as the per-token budget shrinks toward the tight end of the axis, generation
costs proportionally less and retrieval's share of the total rises. Over the 16 shipped
cells the realised share runs **8.72 % – 24.33 %** at the benchmark's 30-token answer
(lowest at ε_workload = 50, Q = 1; highest at ε_workload = 1, Q = 100), and it already
runs **5.21 % – 16.70 %** at the paper's own 70-token answer, against the paper's
operating point of **9.09 %**.

Nothing here is unsound: every cell's total composes to its target exactly, which
`tests/test_shared_grid.py` checks. But the benchmark reproduces the paper's
retrieval:generation *proportion* only near the paper's anchor, and DP-RAG's curve is
therefore not a curve at one fixed split. Each cell records the split it ran at in
`epsilon_retrieval_share`, so the reading does not depend on trusting the constant.

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
| `private_answer_rate` | the same count restricted to answers a private release contributed to. See below: for DPSparseVoteRAG this, not `answer_rate`, is the column that says the mechanism ran |
| `public_only` / `tokens_private` | true exactly when the private mechanism never fired in a cell, and how many releases it made |
| `answer_tokens` | the answer length the cell was configured to produce, so two rows can be *seen* to have been measured over the same amount of text |
| `epsilon_retrieval_share` | DP-RAG only: `eps_r / (eps_r + eps_g)` as the cell ran it |
| `retrieval_precision_at_k` | overlap with the ranking the method would have produced had it not been private. Not relevance: there is no ground truth for which documents a question needed |
| `private_retrieval` | whether that overlap means anything. Three methods protect only the generation and retrieve with a plain cosine search, so their 1.0 is an identity, not an achievement |
| `query_id` | lets you verify two methods answered the same set |
| `workload_error` | why a method is absent from the axis |

DP-SynRAG has no `retrieval_precision_at_k`: it does not retrieve from the corpus, it
replaces it, so document ids do not map. The column is blank rather than 0, which would
read as total retrieval failure.

**`answer_rate` is not evidence that the mechanism ran.** DPSparseVoteRAG's Sparse
Vector Technique tests each step against a noisy threshold and routes everything below
it down a *public*, budget-free path — a plain language-model token that costs no ε and
touches no private histogram. A cell whose Limited-Domain vote can never fire therefore
still returns non-empty answers, and those answers still score: containment, ROUGE and
F1 over text no privacy mechanism produced. Read beside a method that pays for every
token it emits, that is free apparent utility measured exactly where the mechanism is
off. `private_answer_rate` is the column to read when deciding whether a cell was
served; `public_only` is true exactly when it was not, and `tokens_private` counts the
releases. All three are blank — not 0 — for the methods that route no public path, so
"the mechanism served nothing" stays distinguishable from "the question does not arise
here".

**Results are only comparable within one protocol.** Every text metric is monotone in
answer length and every mechanism is calibrated against the deltas it was handed, so a
result produced at a different answer length or with pinned rather than derived deltas
is not on the same scale as one produced under this protocol — the difference is in the
text the model generated, not in how it was scored. Comparing across the two requires a
re-run, not a re-score. `answer_tokens` and `workload_derived_deltas` are in every
payload so the question can be answered from the file rather than from its timestamp.

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
- `build_query_embedder(embedder)` (from `vanilla/query_embedder.py`) — the query tower
  every runner passes to its pipeline, sharing the corpus embedder's loaded weights
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
python experiments/runner.py --smoke --seeds 1 2
```

Read the payloads it writes for what a finishing run does not prove: that the new method
answered the same questions as the others, that its two seeds differ in the mechanism and
agree on the sample, that its budget knob moves it along the shared axis, that the fields
the analysis reads are present, and that an empty answer is reported as empty.

Two seeds that produce identical text are not by themselves a fault: a small model
collapses onto one phrase, every voter then proposes the same token, and no noise draw
can move a single-spike argmax. Under `BENCHDP_NOISE_FINGERPRINT=1` — which `--smoke`
sets for itself — the vote methods record a `noise_fingerprint` per query: a hash of the
first sample their mechanism drew, which depends only on the seed and public parameters.
Identical text with differing fingerprints is therefore fine; the real
"`run_seed` never reached the noise" is fingerprints that repeat across seeds. Without
the fingerprint the question cannot be decided either way, so absent is not a verdict.

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
  Derived from `full/` and kept in step with it by hand; it is deliberately too
  small to time the real grid or calibrate a threshold.

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
