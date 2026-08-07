# Private-RAG (MURAG / MURAG-ADA)

Multi-query differentially private RAG for the BenchDP-RAG framework.

Reference paper:
- Ruihan Wu, Erchi Wang, Zhiyuan Zhang, Yu-Xiang Wang, "Private-RAG: Answering
  Multiple Queries with LLMs while Keeping Your Data Private", arXiv:2511.07637
  (2025).

Prior single-query DP-RAG spends privacy budget on **every** query. Private-RAG
targets the realistic **multi-query** setting: an **individual privacy filter**
gives each document `z` a budget `E(z) = M·eps_q` and charges it only when it is
actually retrieved, so accumulated privacy loss depends on how often each document
is used — not on the number of queries.

The paper has two algorithms that **differ only in retrieval**; both hand their
Top-K documents to the same single-query generation subroutine (paper Algorithm 7).
In this framework that means: swap the retriever, reuse everything else. The two
variants are selected with `private_rag.variant` in the config (`murag` /
`murag_ada`) — "just a different config".

## What Changes vs Vanilla RAG

1. **New component `BudgetManager`** (`budget_manager.py`) — the individual privacy
   filter. `E(z) = M·eps_q` per document; `can_query(z, cost)` / `consume(z, cost)`
   track the running per-document budget. At Rényi order `alpha = infinity` (pure DP)
   composition is additive, so the manager just sums the per-participation costs.

2. **Retriever** — budget-aware, two variants:

   - `BudgetAwareRetriever` (**MURAG**, Alg. 1): a **fixed, public** relevance
     threshold `tau` (data-independent, no DP cost). Every active document above
     `tau` is charged `eps_q` (all of `D_qt`, **before** Top-K), and the Top-K by
     relevance are returned for generation.

   - `AdaptiveBudgetAwareRetriever` (**MURAG-ADA**, Alg. 2): a **query-specific**
     threshold released via **noisy prefix sums** over score bins
     (`s_hat += |A_i| + Lap(1/eps_thr)`, high→low, stop when `s_hat ≥ k`). The
     per-query budget splits `eps_q = eps_thr + eps_RAG`: the threshold release
     charges `eps_thr` (documents in processed bins) and the generation step charges
     `eps_RAG` (the active set above the released threshold). A retrieved document
     therefore pays `eps_thr + eps_RAG = eps_q`.

3. **Generator `PrivateRAGGenerator`** (`generator.py`) — the paper's **pure-DP**
   Algorithm 7. It is a variant of the Koga et al. (2024, arXiv:2412.04697) voting
   mechanism where the Limited-Domain Gumbel step is replaced by the plain
   **exponential mechanism over the full vocabulary** (Algorithm 6), giving an
   ε-DP guarantee (Lemma 1) with **δ = 0**. Per decoding step it compares a no-RAG
   baseline token `b` against `m` voter tokens; a Sparse-Vector agreement test
   `s + Lap(4/eps_Lap) ≤ theta_hat` (with `theta_hat = theta + Lap(2/eps_Lap)`)
   decides whether to spend a private release (`ExpoMech`) or keep the free
   baseline. Budget split `eps_Lap = eps_Expo = eps_0/2`; at most `c = ⌊eps/eps_0⌋`
   private releases.

The generator exposes `.m`, `.k` and `generate_vote(query, docs)`, so it plugs into
the existing `DPVoteRAGPipeline` — **no dedicated pipeline class** is needed.

## Privacy Budget

Pure DP (**δ = 0**). The headline guarantee is the per-document budget cap:

```
epsilon = M · eps_q            # Theorem 1 (MURAG) / Theorem 2 (MURAG-ADA)
```

Accounting (`metrics/privacy_accounting_private_rag.py`) is closed-form: the
individual Rényi DP filter (Feldman & Zrnic 2021, Cor. 3.3) at `alpha = infinity`
is exactly pure DP, so there is no composition blow-up beyond the additive
per-document sum. With the paper's conservative main setting `M = 1`, a document
serves at most one query and `epsilon = eps_q` (the paper's "ε ≈ 10" is
`eps_q = 10, M = 1`). The single-query subroutine is `eps`-DP by Lemma 1
(AboveThreshold with `⌊eps/eps_0⌋` exponential-mechanism releases, each `eps_0`-DP).

## Parameters (config `private_rag` block)

- `variant`: `murag` or `murag_ada`.
- `M`: max retrievals per document (paper main: `M = 1`). Headline `eps = M·eps_q`.
- `m` / `k`: voters / documents-per-voter in Algorithm 7 (retrieved = `m·k`).
- `epsilon_0`: per-token budget (paper `eps_token ∈ {0.5, 1, 2}`). Used only when the
  budget is not derived (see `epsilon_0_from_budget` below).
- `vote_threshold`: SVT threshold `theta` (default `m/2`, "half the voters" — the
  agreement score lives on the `0..m` voter scale, independent of `k`).
- `t_max`: max decoding steps.
- `threshold` (**MURAG**): fixed public relevance threshold `tau` (cosine scale).
- `epsilon_thr` / `epsilon_thr_fraction` + `score_min`/`score_max`/`bin_size`
  (**MURAG-ADA**): threshold-release budget — as a constant, or as a fraction of the
  point's `eps_q` — and the score-bin grid for the prefix sums.
- `truncate_at_double_slash`: cut each context document at its first `//`, as the
  reference implementation does.

Sweep: `epsilon_q_values` (per-query budget). The headline `epsilon = M·eps_q` is
written to each result JSON (`epsilon` field, read by `score_results.py`).
`epsilon_0_from_budget` derives the per-token budget from the swept one instead of
sweeping it (one result file per `eps_q`); with it off, `epsilon_0_values` is a second
axis and each cell gets its own file.

## Deviations from the paper (deliberate, documented)

The shipped configuration is `experiments/configs/full/private_rag.yaml` (MURAG) and
`private_rag_ada.yaml` (MURAG-ADA), which run the benchmark's common evaluation
protocol: the same 6 datasets, 20,000 documents, 40 questions, 7 generators, one
embedder, whole documents and 30 answer tokens as every other method, with every
sweep placed on a shared privacy axis. `experiments/configs/paper_faithful/` keeps the
paper's own settings for each point listed below.

### Per-token budget derived from the swept one

The paper sweeps two budgets: the per-query `eps_q` and the per-token
`eps_token ∈ {0.5, 1, 2}` (Appendix E), and reports, per `eps_q`, the best cell of that
grid. The common protocol instead derives the per-token budget from the swept one,

```
eps_0 = (eps_q - eps_thr) / t_max      # eps_thr = 0 for MURAG, the threshold split for MURAG-ADA
```

so `epsilon_q_values` is the only knob and one result file exists per privacy point
(`epsilon_q_<value>.json`). Two reasons:

- **The release cap does not decide the answer length.** With a pinned `eps_0` the cap
  `c = ⌊eps/eps_0⌋` runs from 1 private token to ~198 across the grid, so points of the
  curve differ in how much text they may release, not only in how noisy each release
  is. Deriving `eps_0` fixes `c ≈ t_max` everywhere: every point may release the whole
  answer and spends its budget on noise instead of length — the same rule the vote
  methods follow, which is what makes their curves comparable.
- **This benchmark's analysis layer averages cells, it does not select among them.**
  The paper's grid is only sound with the paper's best-over-grid reporting. Averaged,
  the two starved cells drag the point down and MuRAG is penalised against methods
  that contribute exactly one cell per privacy level.

The cost is a much smaller per-token budget than the paper ever evaluates: at
`eps_q = 10` the derived value is `eps_0 ≈ 0.33` against the paper's `{0.5, 1, 2}`, so
`eps_Expo = eps_0/2 ≈ 0.17`. The exponential mechanism then barely separates the voted
tokens from the vocabulary — a token with 40 of 50 votes carries weight `e^3.33 ≈ 28`
against ~50,000 unvoted tokens of weight 1, i.e. a private release picks a voted token
with probability ~6·10⁻⁴. Below `eps_q ≈ 32` a *private* token is therefore close to
uniform noise, and the useful output is whatever the SVT lets through on the free
baseline path. This is not an artefact of the derivation but the honest shape of
Algorithm 7 at a small per-query budget spread over 30 tokens; the paper avoids it by
keeping `eps_0` large and `c` small (2 to 20 tokens at its operating points). Read the
two configurations side by side when interpreting low-`eps` MuRAG results.

### Threshold budget as a fraction of the query budget (MURAG-ADA)

The paper fixes `eps_thr = 1.0` for the noisy-prefix-sum threshold release. A constant
is a *floor* on the per-query budget: `eps_RAG = eps_q - eps_thr` must be positive (the
retriever refuses to construct otherwise), so at `eps_q = 1` the whole query budget buys
the threshold and nothing is left to generate with. The tightest point of the shared
axis — the one where the comparison between methods matters most — would simply have no
MURAG-ADA curve. The shipped config therefore spends a fixed *fraction*,
`epsilon_thr_fraction = 0.1`, keeping the split proportional and every point feasible.
At `eps_q = 10`, the paper's own headline operating point, the fraction reproduces the
paper's `eps_thr = 1.0` exactly.

The trade-off is that the threshold's accuracy now scales with the point. Per-bin
Laplace noise is `Lap(1/eps_thr)` on counts, so the prefix sum accumulated over the
B = 170 bins has standard deviation `sqrt(2·B)/eps_thr`: ~18 counts at `eps_thr = 1`
(the regime the paper validated in its Fig. 5, threshold error ≤ one bin), ~184 counts
at `eps_thr = 0.1`, against a target of `k = 50`. At the tightest points the released
`tau_t` is dominated by noise and the walk stops after a few bins or runs to the bottom;
`tau_t` and `n_bins_processed` are recorded per query so this is visible rather than
inferred.

### One global public relevance threshold (MURAG)

`tau` must be **public and data-independent** — that is what lets MuRAG charge nothing
for it. A `tau` fitted per dataset is a function of the corpus, and charging nothing for
that function would understate the guarantee. So a single global value, `tau = 0.25`, is
used for all six datasets, even though no single value suits them all: on the
arctic-embed cosine scale the corpora sit two and a half standard deviations apart
(measured over 20,000 documents × 100 questions each, `scripts/calibrate_tau.py`).

Because MuRAG charges `eps_q` to *every* document above `tau` — not only to the `m·k`
that reach the generator — the fraction of the corpus above `tau` is the fraction of the
corpus spent per query, and at `M = 1` a spent document never comes back. Fitting a
normal to the measured median and 99th percentile of each corpus gives the operating
point per dataset:

| dataset | p50 | p99 | charged/query (of 20,000) | corpus alive after 40 queries | queries until < 50 live documents remain above `tau` |
|---|---|---|---|---|---|
| natural_questions | 0.066 | 0.240 | ~140 (0.7%) | 76% | ~150 |
| triviaqa | 0.103 | 0.253 | ~230 (1.1%) | 63% | ~130 |
| searchqa | 0.131 | 0.323 | ~1,500 (7.5%) | 4% | ~45 |
| chatdoctor | 0.279 | 0.468 | ~12,800 (64%) | ~0 | ~6 |
| medical | 0.399 | 0.590 | ~19,300 (97%) | 0 | ~3 |
| movielens | 0.498 | 0.586 | 20,000 (100%) | 0 | 1 |

`tau = 0.25` is the value calibrated to charge ~160 documents on the median query for
the three DPR-corpus datasets (NQ, TriviaQA, SearchQA), where it leaves the 40-query
stream comfortably inside the budget. On the three dense corpora it sits below the
median: ChatDoctor burns two thirds of the corpus on the first question and is starved
by roughly the sixth, medical by the third, and movielens charges the entire corpus on
question 1, so questions 2–40 retrieve nothing at all. Even before outright starvation
the *relevant* documents go first — they are the ones above `tau` — so the half-life of
a given document is ~0.7 queries on ChatDoctor and ~0.03 on movielens against ~100 on
NQ. Any reading of MuRAG's utility must therefore be per dataset, next to the
`n_docs_charged` / `n_docs_retrieved` fields and the `prefix_curve` recorded in every
result payload; the aggregate over six datasets mixes a method operating normally with
a method that has run out of corpus.

The asymmetry is a property of a public threshold over corpora with different score
scales, not a bug, and MURAG-ADA is the paper's own answer to it: its adaptive threshold
walks down from the top until it has found about `k` documents, so its per-query charge
is self-limiting on every corpus.

### Silent no-RAG fallback

Algorithm 7 splits the retrieved set into a *fixed* number of `m` voters; a voter whose
chunk is empty gets the no-document baseline prompt, so it produces the baseline token
deterministically. With the shipped `m = 50`, `k = 1` and `theta = m/2 = 25`, a query
that retrieves `n` documents starts every decoding step from an agreement score of at
least `50 - n`, and the SVT fires a private release only when
`s + Lap(4/eps_Lap) ≤ theta + Lap(2/eps_Lap)`. Below `n ≈ 25` the test is stacked
towards the free baseline token:

| documents retrieved | P(private release) at `eps_0 = 2` | at `eps_0 = 0.18` |
|---|---|---|
| 50 | 0.999 | 0.67 |
| 25 | 0.50 | 0.50 |
| 10 | 0.016 | 0.39 |
| 0 | 0.001 | 0.33 |

Those `m` voters are built even when nothing is retrieved, and the redundant forward
passes are the price of the guarantee rather than an oversight. Skipping them — the
obvious optimisation, since every chunk is empty — makes the `n = 0` output a
*deterministic* no-RAG decode, a point mass. But `n = 0` and `n = 1` are neighbouring
corpora, and the `n = 1` side fires the SVT with probability ~0.99 over a 30-token
answer, so outputs reachable on one side have probability zero on the other and no
finite ε bounds the ratio. Measured total-variation distance between the two
neighbours, at the branch layer: 0.995 at ε_workload = 1 and 0.971 at ε = 10 with the
shortcut, against 0.006 and 0.031 without it — the latter comfortably inside the
pure-DP bound `(e^ε − 1)/(e^ε + 1)`. Roughly half the queries on the shipped corpora
retrieve nothing, so this is the common path, not a corner case.

At `n = 0` every voter holds the baseline prompt, `s = m = 50`, and the SVT almost
always takes the free baseline token — so the answer is, with probability ~0.999 per
step, a greedy no-RAG continuation: the model answering from its parameters, in the
paper's terms no longer RAG. It is also indistinguishable from a working run in any
utility metric: no error is raised, no empty prediction is produced, and the payload's
`epsilon` still reports `M·eps_q`. The
only signals are `n_docs_retrieved` (0), `n_private_tokens` (0) and the retriever's
score diagnostics. Combined with the starvation table above, an aggregate MuRAG number
on movielens or medical is largely a zero-shot number: always read
`n_docs_retrieved_total` and `n_private_tokens_total` before quoting utility.

### Bins (MURAG-ADA)

The paper bins DPR inner products in `[70, 100] @ 0.2` (B = 150). Scores here are
cosine similarities, so the range and size are config knobs on the cosine scale
(`[0.0, 0.85] @ 0.005`, B = 170). The faithful transposition covers the *occupied* score
band at comparable relative granularity, not the theoretical score range: the largest
score observed on any of the six corpora is 0.832, and the lower bound is 0 — the honest
floor for a cosine over non-negative-similarity retrieval, and the only one that keeps
resolution where the documents actually are. A `score_min` raised above a corpus median
is not conservative but destructive: 0.15, for instance, sits above the median of NQ
(0.066) and TriviaQA (0.103), so most of those corpora falls below the first bin and the
prefix-sum walk has nothing left to count. Like MuRAG's `tau`, the grid is public and data-independent, hence free.

### Fixed relevance threshold on the cosine scale (MURAG)

`tau` is a cosine similarity here rather than a DPR inner product, for the same reason
as the bins. Its calibration and its per-dataset consequences are the section above.

### Context truncation

The reference implementation (Koga et al., `eval_rag_generator.py`) cuts each context
document at its first `//`. On the corpora it was written for that strips a trailing
source marker; on SearchQA snippets and ChatDoctor entries, which contain URLs, it
truncates the document mid-content. Only this method and its ADA variant inherit the
behaviour, so under the common protocol it would feed two of the eight methods a shorter
context than the other six read from the same corpus. `truncate_at_double_slash` is off
in the shipped configs and on in the paper-faithful ones.

### Shared privacy axis at `delta = 1e-2`

Every method in the benchmark is priced on one axis: the privacy loss over a workload of
Q = 100 questions at `delta = 1e-2` (`metrics/workload_accounting.py`), because the
methods' native units — per query, per corpus, per offline build — are not comparable
with each other. MuRAG's own guarantee is **pure DP**, `delta = 0`, and the result
payload keeps reporting it as such (`epsilon = M·eps_q`, `delta = 0.0`).

Evaluating a pure-DP guarantee at a non-zero `delta` is sound because the relation is
one-directional: an `(eps, 0)`-DP mechanism is `(eps, delta)`-DP for every `delta ≥ 0`,
so reading MuRAG at `delta = 1e-2` is a *relaxation* of a statement it already satisfies
— the axis can never credit it with a guarantee it does not have. The conversion charges
no approximate-DP mass (`workload_delta_additive = 0`) and gives
`epsilon_workload = M·eps_q - ln(1/(1-delta)) ≈ M·eps_q - 0.01`, flat in the number of
queries: the individual filter caps each document at `M` participations, so the whole
stream costs what one document can spend, no matter how long it runs. That flatness is
MuRAG's actual claim, and it is what the shared axis is there to show. What the axis
cannot show is the converse — that MuRAG's `delta = 0` is *stronger* than a competitor's
`delta = 1e-2` at the same `epsilon`. That strictly better property is a separate
statement, made in the per-method payload rather than on the common curve.

### Multi-query cap `M = 1`

The shipped configs use the paper's conservative main setting, `M = 1`: a document may
serve at most one query, and the headline guarantee collapses to `epsilon = eps_q`. It
is also the harshest setting for utility, and it is what makes the corpus-exhaustion
behaviour above visible within a 40-question stream. Raising `M` trades a proportionally
larger headline `epsilon` for a corpus that survives longer.

### Other deviations

- **Seeds**: the paper reports no random seeds or repeated trials. This framework
  uses the shared convention — `random_seed` (default 42) plus a per-query offset
  derived from `hash(query)` for the Laplace / exponential-mechanism draws — so runs
  are reproducible (the ε-DP guarantee is a property of the sampling distribution and
  is unaffected). Same caveat as DP-RAG / DP-KSA / DP-SynRAG.
- **Datasets**: the paper uses NQ, TriviaQA, MQuAKE and ChatDoctor — 100 questions
  each for NQ/TriviaQA/ChatDoctor, over a ~20M-passage Wikipedia corpus (NQ/TriviaQA)
  or the remaining ~200K QA pairs (ChatDoctor). MQuAKE is not in this repo; the
  harmonized 6-dataset set (shared across all methods) is used instead, with a
  tractable index size (`max_index_docs`, default 20,000; full = `null`) standing in
  for the paper's full corpus, and 40 test queries per dataset instead of the
  paper's 100.
- **Retriever/embedder**: the paper retrieves with DPR. This benchmark standardizes
  on one modern embedder (`Snowflake/snowflake-arctic-embed-m-v1.5`, single-tower,
  cosine, vectors L2-normalised) across all methods so the retriever is not a
  confound in the cross-method comparison (fairness prioritized over per-paper
  faithfulness).
- **Generator sweep**: the paper evaluates OPT-1.3B, Pythia-1.4B and Mistral-7B. The
  harmonized (`full`) config sweeps the same two base models plus several ≤4B
  instruct models shared across all methods, and drops Mistral-7B because it does
  not fit the GPU in that sweep; the paper-faithful config keeps Mistral-7B but
  flags that it batches all `m = 50` voters in the Algorithm-7 subroutine without
  `logits_to_keep=1`, so it is VRAM-heavy and is OOM-skipped cleanly if it doesn't
  fit.
- **Privacy unit**: one indexed entry (one row of `embeddings_matrix`) is treated as
  one document `z`, matching the other per-query methods.

## Performance Note

Each private decoding step runs `m + 1` forward passes (m voters + 1 baseline), like
DPSparseVoteRAG. With `M = 1` each document is consumed by at most one query, so a
long query stream progressively exhausts the corpus; the runner logs how many
documents are tracked / exhausted per operating point.
