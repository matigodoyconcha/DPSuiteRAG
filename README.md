# DPSuiteRAG

A modular, reproducible framework for benchmarking Differential Privacy (DP) methods
in Retrieval-Augmented Generation (RAG) systems: privacy–utility trade-off curves
across a range of ε values, with every method accounted on one shared axis.

This repository accompanies the paper *DPSuiteRAG: A Benchmarking Abstraction for
Differentially Private Retrieval-Augmented Generation* (WSDM 2027). See
[Citation](#citation) below.

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
| Generators | The same 4 models (OPT-1.3B, Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, Phi-3.5-mini-instruct), swept identically | |
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

An additive δ is charged once per *release*, so the protocol fixes the spend over the
whole workload and derives the per-release parameter (`metrics/workload_accounting.py`):

```
delta_token   = MECHANISM_DELTA_SHARE * δ / (Q * T)        # one release per token
mechanism_δ   = MECHANISM_DELTA_SHARE * δ / (Q * fraction) # one PTR test per query
```

`MECHANISM_DELTA_SHARE` is **0.5**: half of the axis δ pays for the mechanisms' own
bottom-out (the Limited-Domain Gumbel's null symbol, DP-KSA's PTR false release) and
half is left to the RDP → (ε, δ) conversion. Every cell of every grid therefore pays the
same price for the common δ, and the derived value sets both the mechanism's threshold
and the reported ε. Each payload lists what it derived in `workload_derived_deltas`.
`paper_faithful/` pins these deltas at each paper's own value instead.

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

**`answer_rate` is not evidence that the mechanism ran.** DPSparseVoteRAG routes the
steps below its noisy threshold to a public, budget-free path, so a cell where the
private vote never fires still returns scored answers. Read `private_answer_rate`,
`public_only` and `tokens_private` to see whether a cell was served by the mechanism.

**Results are only comparable within one protocol.** Text metrics move with answer
length and every mechanism is calibrated to the deltas it was handed, so comparing a
result produced under other settings requires a re-run, not a re-score.

---

## Adding a method to the protocol

A new method joins the comparison by satisfying the same contract as the others.
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

Then check, in the payloads it writes, that the new method answered the same questions
as the others, that its two seeds differ in the mechanism and agree on the sample, that
its budget knob moves it along the shared axis, and that an empty answer is reported as
empty. `--smoke` sets `BENCHDP_NOISE_FINGERPRINT=1`, which records a hash of the first
noise sample per query so two seeds can be told apart even when the text is identical.
Leave it **off** in measurement runs: `core/noise_trace.py` explains why.

## Requirements

- Python 3.10+
- Active virtual environment

### Hardware

The paper's runs used one GPU per job: a 16 GB card and 48 GB cards.

- **GPU:** 16 GB is enough for every generator of the protocol (up to 3.8B parameters).
  8B models do not fit in 16 GB. Phi-3.5-mini on the long-document corpora (ChatDoctor,
  MovieLens) can run out of memory at 16 GB; a 24 GB or larger card avoids it.
- **RAM and disk:** Natural Questions and TriviaQA retrieve from the full 21M-passage
  Wikipedia corpus (DPR): about 13 GB to download and a FAISS index of about 64 GB, built
  with about 128 GB of RAM (build one index at a time). The other corpora hold at most
  20,000 documents and need no special hardware.
- **Quick check:** `python experiments/runner.py --smoke` runs every method in a few
  minutes on one GPU.

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

Each method can also be run on its own, e.g.

```bash
python experiments/run_dp_rag.py --config experiments/configs/full/dp_rag.yaml
```

MuRAG-Ada is `run_private_rag.py` with `private_rag_ada.yaml`. Swap `full/` for
`paper_faithful/` to run a method under its own paper's settings. Results are written
under `results/` (not committed).

### Dataset and model configuration

Everything is set in the method YAML (`dataset`/`datasets`, `llm`/`llms`, `models`):

- `dataset.name`: `natural_questions | triviaqa | searchqa | chatdoctor | movielens |
  medical_dirichlet_phi3 | squad | hotpotqa`
- `dataset.max_index_docs` / `max_test_queries`: corpus and question caps.
- `dataset.docs_path` / `queries_path`: your own corpus and questions
  (`.txt/.json/.jsonl`; queries as `{"query": ..., "answer": ...}`).
- `llm.model_name` and `models.embedder`; top-level `datasets:` / `llms:` lists sweep
  several of each in one config.

## Results figures

`analysis/plot_results.py` draws the results figures of the paper for any run, and for
any subset of methods and corpora:

```bash
python -m analysis.plot_results --run-dir results/run_<stamp>
```

| Figure | What it shows |
|---|---|
| `reference_range` | the no-retrieval floor and the non-private ceiling of every corpus; the gap between them is all a private method can win there |
| `tradeoff_q1` | utility against the shared budget ε at Q = 1 (`--workload` for another Q), one panel per corpus, one line per method |
| `workload_axis` | utility against the workload size Q at fixed budgets (`--epsilons`, default the two largest) |

Only what the run holds is drawn: a method with no cells is left out of the panels and
the legend, and a corpus whose ceiling does not lie above its floor is kept off the
method figures (`--all-datasets` keeps it). Every value is a mean over generators, and
each figure is written with a CSV of what it draws. MuRAG and MuRAG-Ada answer one
stream, so their point at Q is the utility over the first Q answers of that stream.
Several `--run-dir` are merged (e.g. a separate 100-query stream run), `--group
"QA=triviaqa,natural_questions,searchqa"` averages corpora into one column of the
workload figure, and `--from-csv` redraws from the `results_cells.csv` a previous call
wrote.

The cells behind the paper's figures are in `docs/paper_results_cells.csv`:

```bash
python -m analysis.plot_results --from-csv docs/paper_results_cells.csv --out figures \
    --group "Real QA (mean)=triviaqa,natural_questions,searchqa" --epsilons 10 50
```

Other analysis tools:

- `python analysis/plot_run.py --run-dir results/run_<stamp>`: diagnostic plots of one
  run (coverage, metric heatmap, per-method curves, MIA, per-generator utility).
- `python -m analysis.summary_table`: the summary table as CSV and LaTeX.
- `python -m analysis.plot_tradeoff`: ε vs utility and MIA, and the privacy–utility
  Pareto front per corpus.

## Tests

```bash
pytest tests/ -v
```

## Citation

If you use DPSuiteRAG, please cite:

```bibtex
@inproceedings{godoyconcha2027dpsuiterag,
  author    = {Godoy-Concha, Matias and Bravo-Marquez, Felipe and Toro Ipinza, Mat\'{\i}as},
  title     = {{DPSuiteRAG}: A Benchmarking Abstraction for Differentially Private
               Retrieval-Augmented Generation},
  booktitle = {Proceedings of the Twentieth ACM International Conference on Web Search
               and Data Mining (WSDM '27)},
  year      = {2027},
  address   = {Hong Kong},
  publisher = {ACM},
  doi       = {10.1145/3862322.3863848}
}
```

## License

Released under the [MIT License](LICENSE).
