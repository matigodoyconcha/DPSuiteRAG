# DPVoteRAG / DPSparseVoteRAG

Implementation of the DPVoteRAG and DPSparseVoteRAG methods described in
Koga et al., *"Privacy-Preserving Retrieval-Augmented Generation with Differential
Privacy"* ([arXiv:2412.04697](https://arxiv.org/html/2412.04697v3)).

This implementation follows the official repository
[tacchan7412/DPRAG](https://github.com/tacchan7412/DPRAG) for the LD-Gumbel
mechanism and the RDP accountant. Several budget and configuration choices
deliberately follow the **paper** rather than the reference *code*, or the
**benchmark** rather than the paper, for the reasons detailed below; see
[Deviations from the paper / reference implementation](#deviations-from-the-paper--reference-implementation)
for the complete, consolidated list.

## Components

- [`generator.py`](generator.py) — `VoteRAGGenerator` with the
  Limited Domain Gumbel mechanism (Durfee & Rogers, NeurIPS 2019).
- [`sparse_vote_generator.py`](sparse_vote_generator.py) — `SparseVoteRAGGenerator`
  extending the above with the Sparse Vector Technique (Dwork & Roth).
- [`pipeline.py`](pipeline.py) — RAG orchestration (embedder, indexer, retriever,
  generator, postprocessor).

The `(ε,δ)` accounting is delegated to
[`metrics/privacy_accounting_autodp.py::AutodpLDAccountant`](../../../metrics/privacy_accounting_autodp.py).
By default (`compose_method='rdp'`) it matches the reference runtime accountant:
EM-peeling RDP for the exponential mechanism, composed over tokens and converted
via BBGHS'19, with the reference delta split `delta1 + T·delta0` — a port of
`tacchan7412/DPRAG/utils/dp_accountant.py` (the reference `LimitedDomain` engine
defaults to `compose_method='rdp'`). The advanced+sequential composition of
`utils/privacy.py` is retained as the selectable `compose_method='simple'`. The
accountant is **online**: generation stops when `epsilon_total`/`delta_total` is
exhausted.

## DPVoteRAG (`VoteRAGGenerator`)

For each generated token:

1. Randomly partitions the `m·k` retrieved documents into `m` voters
   (each with `k` documents).
2. Builds `m` prompts (identical to the reference:
   `"Instruction: ... Context: 1. doc1; 2. doc2 ... Question: ... Answer:"`).
3. Each voter produces the next token via greedy decoding.
4. Aggregates a vote histogram and applies **Limited Domain Gumbel**:
   - Sorts counts in descending order and takes the top `k_bar`.
   - Threshold: `h_⊥ = h_{k_bar} + 1 + log(min(sens, k_bar, dim − k_bar) / δ) / ε`.
   - Gumbel(0, 1/ε) noise on counts and threshold.
   - If `max(v_counts) ≥ v_⊥` → winning token is selected (success).
   - If `k_bar` exceeds the number of distinct observed tokens, the
     "unobserved slots" can outcompete the noisy threshold; when that happens
     a uniformly random *unobserved* token is returned instead of halting
     (`output_rand_if_fail` in `_limited_domain_or_fail`, `generator.py`).
   - Otherwise → `NotFoundLDTop1`.  `fail_mode='stop'` (paper default,
     arXiv:2412.04697v3 §2.1: *"we halt the algorithm"*): generation stops
     with `exit_status=1`.
5. The accountant records the ε cost per token; if `epsilon_total`
   or `delta_total` is exceeded, generation stops with `exit_status=2`.

### Relevant parameters

| Parameter | Description | Default |
|---|---|---|
| `m` | Number of voters | 30 |
| `k` | Documents per voter | 1 |
| `epsilon_token` / `delta_token` | LD budget per token | 2.0 / 1e-5 |
| `epsilon_total` / `delta_total` | Total budget per query (online accountant) | 10.0 / 1e-4 |
| `k_bar` | Reduced domain size in LD | `m` |
| `sensitivity` | l₀-sensitivity (add/remove ⇒ 2) | 2.0 |
| `max_length` | Generation-length loop bound (ref `--max_length`); the online budget check still stops generation early once composition is exhausted | 50 |
| `tmax_absolute` | Optional extra safety ceiling on top of `max_length` (`None` disables it — every shipped config sets it to `null`) | 50 |
| `n_peels` | Top-i peels per token (1 = top-1, matching reference) | 1 |
| `fail_mode` | Behaviour on LD failure — only `'stop'` (paper default) | `'stop'` |
| `random_seed` | Seed for partitioning and noise | 42 |

> **Note:** if `len(docs) < m·k` a `ValueError` is raised. The paper assumes
> `n_split·n_docs` retrieved; adjust `k_retrieved` or reduce `m`/`k` in the config.

### Output (`last_run_stats`)

- `exit_status`: 0 = OK, 1 = LD failed (stop), 2 = budget exhausted.
- `tokens_generated`, `n_private_tokens`, `n_nonprivate_tokens`.
- `dp_spent_epsilon`, `dp_spent_delta`, `dp_summary`.

## DPSparseVoteRAG (`SparseVoteRAGGenerator`)

Extends DPVoteRAG with a formal Sparse Vector Technique. At each step:

1. Generates a public token `y_nonRAG` from a prompt without context
   (does not consume LD budget).
2. Measures its agreement `a_t` with the private voters.
3. Compares `a_t + Lap(0, 4/sparse_eps)` against `τ̂ = τ + Lap(0, 2/sparse_eps)`.
4. **If `a_t_noisy ≤ τ̂` → applies LD-Gumbel** (private token, spends LD portion).
   The threshold `τ̂` is resampled only after a private step, following SVT.
5. **If `a_t_noisy > τ̂` → accepts the public token** (no LD cost).

### Per-token budget split (paper §3.3)

The per-token budget `epsilon_token` is split between SVT (Laplace) and LD-Gumbel
via `sparse_fraction` (default 0.5 = equal split, matching the paper):

```
sparse_eps   = epsilon_token × sparse_fraction          # SVT
ld_eps       = epsilon_token × (1 − sparse_fraction)    # LD-Gumbel
```

Paper verbatim: *"(ε_token-RAG, δ_token-RAG) ← (ε_token/2, δ_token)"* and
*"ε_token-Lap ← ε_token/2"*.

> **Design note (paper vs reference code).** The reference repo exposes `dp_eps`
> (LD) and `sparse_eps` (SVT) as *independent* CLI arguments and does not reduce the
> LD budget when SVT is enabled. We instead *derive* both from a single
> `epsilon_token` via `sparse_fraction`, matching the paper's §3.3 equal split. This
> is a benchmarking choice: it keeps the **total** per-step budget comparable across
> DPVoteRAG and DPSparseVoteRAG so the privacy–utility curves are fairly aligned. It
> is a design decision, not a fidelity bug.

### SVT privacy accounting

The Sparse Vector Technique is charged **per Above-Threshold epoch**, following the
standard SVT analysis (Lyu-Su-Li 2017): one noisy threshold `τ̂ = τ + Lap(2/sparse_eps)`,
used until the first above-threshold release, is `sparse_eps`-DP. We therefore charge
`sparse_eps` **once per threshold instantiation** — the initial threshold and each
resample after a private token — via `accountant.account_svt_step(sparse_eps)`. The
below-threshold comparisons and the agreement-count noise `Lap(4/sparse_eps)` are
covered by that same epoch budget and are **not** charged separately. Epochs compose
sequentially into `svt_spent_epsilon`, which reduces the budget available to the LD
side.

> **Deliberate hardening vs reference.** The reference (`utils/ensemble.py`) performs
> the SVT Laplace draws but never composes their cost into its online accountant. We
> compose the SVT-epoch cost online, which is strictly more conservative (the reported
> ε remains a valid upper bound) and never weakens the DP guarantee.

### Optional exclusion of trivial tokens

- `exclude_stopwords`: stopwords loaded from `data/stopword_words.pt`
  (paper convention: the id is taken as `tokenizer(word)['input_ids'][1]`,
  skipping `<bos>`).
- `exclude_question_words`: ids of the question tokens (includes `<bos>`, matching the reference).

When a token falls in the exclusion list, it is accepted directly and
`flag=None` is recorded in `dp_flags_per_token`. The test happens *before* the
sparse comparison, so an excluded token pays neither the SVT epoch nor the
LD-Gumbel share: it is emitted from the public, context-free model for free.

This is the reference behaviour and it is confined to the sparse variant — the
dense LD-Gumbel path (`utils/ensemble.py` in the reference) passes no
`exclude_ids`, and `full/dpvoterag.yaml` keeps `exclude_stopwords: false`
accordingly. It is worth knowing how large the exemption is:
`data/stopword_words.pt` holds 29 words (`a an and are as at be by for from how
in is it of on or that the this to was what when where which who why with`), and
across 1200 recorded DPSparseVoteRAG answers 6069 of 16881 emitted words — 36% —
are on that list. At the shared answer length of 50 tokens that is on the order
of 18 tokens per answer routed around the mechanism. Among the eight benchmarked
methods DPSparseVoteRAG is the only one with a free public path of this kind,
which is why the exemption is stated rather than folded into the utility numbers.

### Additional parameters

| Parameter | Description | Default |
|---|---|---|
| `sparse_fraction` | Fraction of `epsilon_token` allocated to SVT | 0.5 |
| `tau` | Agreement threshold (default `m/2`); YAML key is `threshold` (the runner maps `threshold` → the `tau` constructor argument — `tau` is also accepted) | `m/2` |
| `max_private_tokens` | Ceiling `c` on the number of private LD-Gumbel releases per answer (paper Alg.3 line 25); not a budget divisor. `full/` sets it to 50, equal to `tmax_absolute`, so every token of the answer *can* be private | 32 |
| `exclude_stopwords` | Enable stopword exclusion | False |
| `exclude_question_words` | Enable question-token exclusion | False |
| `stopwords_path` | Path to `stopword_words.pt` | `data/stopword_words.pt` |
| `tmax_absolute` | Maximum tokens per response; every shipped config sets 50, the shared answer length | 200 |

### Additional output

- `dp_flags_per_token`: per token, `True` = private LD, `False` = SVT public,
  `None` = excluded.
- `n_private_tokens`, `n_nonprivate_tokens`, `n_excluded_tokens`, `total_tokens`
  (`total_tokens` stands in for DPVoteRAG's `tokens_generated`).
- Unlike DPVoteRAG, `last_run_stats` has no `dp_summary` key — only
  `dp_spent_epsilon` / `dp_spent_delta`.

## Test-set construction (NQ / TriviaQA)

The paper (§IV-A) does not evaluate on the full NQ/TriviaQA test sets. It first
filters out questions with fewer than 20 or more than 59 ground-truth relevant
documents in the Wikipedia corpus, then stratifies the remainder into four bins by
relevant-document count (20–29, 30–39, 40–49, 50–59) and samples 25 questions per
bin (100 questions total, seed 42). The rationale is stated directly: *"If a
question relates with only a few documents, it is inevitable that DP algorithms
fail since replacing one document would change the output a lot"* (§IV-A). The
reference repo implements this in `subsample_data.py`, deriving the bins from the
`positive_ctxs` counts in the DPR biencoder dev files. Since voters draw the
top-`m·k` retrieved documents from the corpus, this filter guarantees that pool is
dense in relevant documents before any voting happens.

> **Design note (benchmark vs paper).** This benchmark does not apply the §IV-A
> filter: NQ/TriviaQA test questions are sampled without conditioning on how many
> supporting documents exist in the corpus. Pre-selecting only questions backed by
> 20+ relevant documents inflates the method's apparent utility and is not a valid
> comparison setting for a cross-method benchmark — realistic query streams include
> questions with thin or no corpus support, and a DP method's degradation on those
> is part of what the benchmark measures.

**Consequence (expected, not a bug).** On unfiltered NQ/TriviaQA, many voters hold
irrelevant documents, vote histograms flatten, and the Limited Domain Gumbel step
fails more often, halting generation (empty predictions, `exit_status=1`). NQ/
TriviaQA results from this benchmark are therefore not directly comparable to the
paper's utility curves; this test-set construction difference is a primary cause of
the gap. A secondary confound: the paper retrieves with the NQ-supervised DPR
encoder, while this benchmark standardizes retrieval on `arctic-embed-m-v1.5`
across all methods.

## The shared workload axis

`experiments/configs/full/` is not a per-paper configuration. All eight
benchmarked methods run the same protocol — six datasets, 20 000 indexed
documents each, 40 questions, the same seven generators, `arctic-embed-m-v1.5`,
no chunking, 50-token answers — and each method's ε sweep is *derived*, by
[`analysis/derive_epsilon_grids.py`](../../../analysis/derive_epsilon_grids.py),
so that its points land on the common axis {1, 3, 10, 30, 100} computed by
[`metrics/workload_accounting.py`](../../../metrics/workload_accounting.py) at
Q = 100 queries and δ = 1e-2. A point on one method's grid and a point on
another's are then the same privacy level, which is the only way the curves can
be plotted on one figure. `experiments/configs/paper_faithful/` keeps the papers'
own settings and sweeps.

Two consequences shape the vote methods specifically.

**The per-token budget is derived, not pinned.** With
`epsilon_token_from_budget: true` the runners compute
`epsilon_token = epsilon_total / T`, where `T` is `max_length` for DPVoteRAG and
`max_private_tokens` for DPSparseVoteRAG (50 in both). The swept value is
therefore the *whole* budget of a query, and every point of the sweep releases
the same number of tokens. The derived value is what the generator, the release
ceiling, the online accountant, the persisted payload and the sparse skip probe
all read; the `epsilon_token` key in the method block is only the fallback for
`epsilon_token_from_budget: false`, and `epsilon_token_values` collapses to a
single curve.

The alternative — the paper's fixed `epsilon_token = 2` with `epsilon_total`
swept — makes the budget decide the answer *length*: one token at
`epsilon_total = 2`, twenty at 40. The curve then mixes utility-from-length with
utility-from-less-noise, and nothing on it is comparable to another method's
point at the same nominal ε.

**`max_private_tokens` equals the answer length.** Not every token
DPSparseVoteRAG emits is private — the sparse test routes the rest to a
budget-free public path — so a private-release ceiling `c` below `tmax_absolute`
hands the sparse variant free tokens that DPVoteRAG, which pays for all 50, does
not get. Setting `c = tmax_absolute = 50` makes the *private budget* the
comparable quantity across the two and leaves the sparse test, rather than a
config constant, deciding which tokens are private.

### Feasibility of the LD-Gumbel under the shared axis

Deriving the per-token budget from a workload target has a measurable
consequence, and it is the most important fact about these two methods in this
benchmark: **at Q = 100 the shared targets sit far below the budget at which the
Limited-Domain Gumbel can release anything at all.** Every cell of both shipped
`full/` grids returns the null symbol at its first private token and, under
`fail_mode: stop`, an empty answer.

The mechanism bottoms out at

```
h_⊥ = h_{k_bar} + 1 + log(min(sens, k_bar, dim − k_bar) / δ_token) / ε_token
```

with `k_bar = m = 50` larger than the number of distinct tokens 50 voters ever
produce, so `h_{k_bar} = 0` and `min(...) = sens = 2`. The largest count any
token can reach is `m = 50`. Once `h_⊥ ≥ m`, no histogram the voters can produce
clears the threshold.
[`metrics/privacy_accounting_autodp.py::ld_gumbel_feasible`](../../../metrics/privacy_accounting_autodp.py)
evaluates exactly this, and `tests/test_shared_grid.py::test_the_grid_is_reported_when_the_vote_cannot_fire`
pins the outcome for every shipped point:

| ε_workload (Q=100) | DPVoteRAG ε_total | ε_token | h_⊥ | DPSparseVoteRAG ε_total | LD share | h_⊥ |
|---|---|---|---|---|---|---|
| 1 | 0.2917 | 0.00583 | 2487.9 | 0.4104 | 0.00410 | 3536.3 |
| 3 | 0.7044 | 0.01409 | 1030.9 | 0.9910 | 0.00991 | 1465.0 |
| 10 | 1.7642 | 0.03528 | 412.2 | 2.4819 | 0.02482 | 585.6 |
| 30 | 3.7717 | 0.07543 | 193.3 | 5.3051 | 0.05305 | 274.5 |
| 100 | 8.0794 | 0.16159 | 90.8 | 11.3542 | 0.11354 | 128.8 |

Every `h_⊥` exceeds `m = 50`, by a factor of 50 at the tightest point and by
1.8× at the loosest. Release probability is logistic in `(h_⊥ − votes)/(1/ε)`, so
even a token every one of the 50 voters agrees on is released with probability
6.7e-7 (DPVoteRAG, ε_workload = 1) to 1.4e-3 (ε_workload = 100), and 6.1e-7 to
1.3e-4 for DPSparseVoteRAG. One miss ends the answer.

**The feasible frontier.** Release becomes possible when
`ε_token > log(2/δ_token)/(m − 1)`, which at `δ_token = 1e-6` and `m = 50` is
`ε_token > 0.29610`. Multiplying back through the divisor — and, for the sparse
variant, through its half share — the frontier is:

| | ε_total per query | ε_workload at Q=100 | ε_workload at Q=1 |
|---|---|---|---|
| DPVoteRAG | 14.80 | 289.5 | 7.12 |
| DPSparseVoteRAG | 29.61 | 568.9 | 11.93 |

So the loosest shared target, 100, is 2.9× below DPVoteRAG's frontier and 5.7×
below DPSparseVoteRAG's. The sparse variant is the harder of the two for exactly
the reason its budget split exists: the SVT takes half of the per-token budget,
which doubles the ε_total needed to reach the same LD-Gumbel threshold.

This is a property of the method under query-stream composition, not a
configuration accident, and it is the reason the frontier is quoted at Q = 1 as
well. A single query composes 100× more cheaply, and there the frontier lands at
7.12 and 11.93 — inside the range a privacy-utility figure would normally plot.
The vote methods are usable at a single query and are not usable at a hundred:
their cost is linear in the stream length, with no amortisation, while the
mechanism's release threshold is a hard step in `ε_token`.

The paper's own configuration sits comfortably on the feasible side, which is why
this does not appear there: `paper_faithful/dpvoterag.yaml` pins
`epsilon_token = 2` (`h_⊥ = 8.25`) and `paper_faithful/dpsparsevoterag.yaml` an
LD share of 1.0 at `δ_token = 1e-5` (`h_⊥ = 13.2`), both far below `m = 50`, and
both release a unanimous token with probability ≈ 1. What the paper does not do
is state a guarantee over a hundred-query stream.

The `full/` grids are kept and run rather than deleted: the empty answers *are*
the measurement behind the claim above, the accounting stays valid (an answer
that leaks nothing leaks nothing), and the test records the state so it cannot
change in either direction without being noticed.

## Deviations from the paper / reference implementation

Every divergence below is deliberate and documented at its source: either a
paper-vs-reference-code mismatch resolved in the paper's favor, or a
benchmark-motivated choice made in the interest of a fair, tractable
cross-method comparison. Detailed subsections are cross-referenced rather
than repeated here. Everything applies to both generators unless the entry says
otherwise.

- **Derived per-token budget instead of the paper's fixed sweep** — `full/`
  sets `epsilon_token_from_budget: true`, so `epsilon_token` follows from the
  swept query budget and the answer length rather than being pinned at the
  paper's 2 while `epsilon_total` sweeps independently. See "The shared workload
  axis" above.
- **Shared-axis ε grids** — `full/`'s `epsilon_values` are the values whose
  composed workload epsilon hits {1, 3, 10, 30, 100} at Q = 100 and δ = 1e-2,
  not the paper's `epsilon_total ∈ [2, 40]`. They are per-method units and are
  not comparable point-for-point with the paper's x-axis; the workload figure is.
- **The shared targets lie below the LD-Gumbel's feasible frontier** — the
  headline consequence of the two entries above, and the single most important
  caveat when reading these methods' `full/` results: every cell of both grids
  produces an empty answer, and release only begins at ε_workload ≈ 289
  (DPVoteRAG) and ≈ 569 (DPSparseVoteRAG) at Q = 100, or ≈ 7.1 and ≈ 11.9 at
  Q = 1. See "Feasibility of the LD-Gumbel under the shared axis" above.
- **`//` context truncation disabled under the shared protocol** — the reference
  (`eval_rag_generator.py`) cuts each context document at its first `//`, which
  on the corpora it was written for strips a trailing source marker but on
  SearchQA snippets and ChatDoctor entries, both of which contain URLs,
  truncates mid-content. Only these two methods and MuRAG inherit the behaviour,
  so it would feed three of the eight a shorter context than the other five read
  from the same corpus. `full/` sets `truncate_at_double_slash: false`;
  `paper_faithful/` omits the key and gets the reference behaviour from the
  `VoteRAGGenerator.truncate_at_double_slash = True` class default, which
  `SparseVoteRAGGenerator` inherits.
- **`max_private_tokens` matches the answer length** — `full/` sets the private
  release ceiling `c` to 50, equal to `tmax_absolute` and to DPVoteRAG's
  `max_length`, so the private budget is the quantity being compared rather than
  a config constant capping how many of the 50 tokens are allowed to be private.
  `paper_faithful/` keeps `c = 32`. See "The shared workload axis" above.
- **Corpus size** — 20 000 indexed documents per dataset, against the paper's
  full Wikipedia DPR passage collection (~21 M) for NQ/TriviaQA and all
  remaining ChatDoctor QA pairs. This is a tractability bound on indexing time
  and memory, and it thins the pool the top-`m·k` retrieval draws from; combined
  with the missing §IV-A relevant-document filter it is a primary reason voter
  histograms flatten here relative to the paper.
- **Free public path for stopwords, DPSparseVoteRAG only** —
  `exclude_stopwords: true` lets the public model emit any of 29 stopwords
  without touching the SVT or the LD budget. This is reference behaviour, but it
  is unique to this method among the eight, and it covers roughly a third of the
  words in a generated answer. See "Optional exclusion of trivial tokens" above.
- **Sparse per-token budget split** (`sparse_fraction`, §3.3) — derives the
  SVT and LD-Gumbel shares from a single `epsilon_token` instead of the
  reference's independent `--dp_eps`/`--sparse_eps` CLI arguments, so the
  total per-step budget stays comparable between DPVoteRAG and
  DPSparseVoteRAG. See "Per-token budget split (paper §3.3)" above.
- **Online SVT-epoch accounting** — composes the SVT Laplace cost into the
  online accountant once per threshold instantiation; the reference draws
  the noise but never charges it. Strictly more conservative (the reported
  ε stays a valid upper bound). See "SVT privacy accounting" above.
- **§IV-A test-set-construction filter not applied** — NQ/TriviaQA questions
  are sampled without conditioning on relevant-document count in the corpus,
  unlike the paper's stratified 20–59-doc filter. See "Test-set construction
  (NQ / TriviaQA)" above.
- **Standardized retriever** — retrieval uses `arctic-embed-m-v1.5` across
  all benchmarked methods instead of the paper's NQ-supervised DPR encoder,
  removing the retriever as a cross-method confound. See "Test-set
  construction" above.
- **Accountant composition method (`compose_method`)** — the generator/
  accountant code defaults to `compose_method='rdp'` (EM-peeling RDP +
  BBGHS), matching the reference runtime's own default; at low
  `epsilon_total` this collapses `Tmax` to 0 (empty answers). Every shipped
  DPVoteRAG config (`experiments/configs/full/dpvoterag.yaml` and
  `experiments/configs/paper_faithful/dpvoterag.yaml`) therefore sets
  `compose_method: simple` — the method as literally written
  in the paper (footnote 4, Alg. 2: max of advanced/sequential composition)
  — for headline results (`generator.py`, DPVOTE-01). DPSparseVoteRAG's config
  sets no `compose_method` and so takes the reference-matching `rdp` default.
  Two online accountants over the same mechanism, and at the shipped `full/`
  grids the choice is inert: the uncapped `T_max` is 84–100 tokens under
  `simple` and 87–99 under `rdp` for DPVoteRAG, and 99–100 under either for
  DPSparseVoteRAG, all well above the loop cap of 50, so the release ceiling is
  50 at every point either way. What binds there is δ, not ε —
  `delta_total/delta_token = 100` — which is also why the two accountants agree.
  The divergence the switch exists for appears at the paper's own low-`ε_total`,
  high-`ε_token` points, where `rdp` drives `T_max` to 0.
- **`delta_token` lowered to `1e-6`** — the paper's `1e-5` caps the delta
  budget (`delta1 + T·delta_token`) at `T <= 9` given `delta_total = 1e-4`,
  making `epsilon_total = 20` and `= 40` collapse to identical (degenerate)
  results. Both `dpvoterag.yaml` configs and `full/dpsparsevoterag.yaml` use
  `1e-6`, lifting the cap to ~99 tokens so high-epsilon points differentiate
  (`generator.py`, DPVOTE-02); `paper_faithful/dpsparsevoterag.yaml` keeps the
  paper's `1e-5`. `delta_token` is not a reporting knob: it sits inside `h_⊥`,
  so a smaller value raises the halt probability, and over the workload it also
  contributes `T·Q·delta_token` of additive mass (5e-3 at T = 50, Q = 100),
  which `metrics.workload_accounting` composes and reports rather than hides.
- **`epsilon_total=2` dropped from the DPSparseVoteRAG paper-faithful sweep**
  — the paper tabulates `epsilon_total >= 5`; at `epsilon_total=2` this
  implementation always yields empty predictions (`tokens_private=0`), so
  `experiments/configs/paper_faithful/dpsparsevoterag.yaml`'s `epsilon_values`
  sweep starts at 5.
- **Conservative epsilon charging on LD-Gumbel failure** — a failed
  `get_top1` call still inspects the private vote histogram, so both
  generators charge that terminal token's epsilon before halting. The
  reference increments only its delta counter (not epsilon) on the failure
  branch; charging epsilon here is more conservative — bounded to the single
  terminal token, since a failure always ends generation — and keeps the
  reported ε a valid upper bound (`generator.py`, DPVOTE-07;
  `sparse_vote_generator.py`, DPSPARSE-08).
- **Sample size** — evaluates up to `max_queries=40` questions per
  dataset/LLM/epsilon point rather than the paper's 100, for tractability
  across the LLM × dataset × epsilon sweep grid.
- **Extra harmonized datasets** — beyond the paper's three (TriviaQA,
  Natural Questions, ChatDoctor), the full-sweep configs add SearchQA,
  `medical_dirichlet_phi3` and MovieLens so all eight benchmarked DP-RAG
  methods share the same six datasets.

## Recommended hyperparameters

| Regime | m | k | epsilon_token | sparse_fraction | τ | epsilon_total |
|---|---|---|---|---|---|---|
| Quick (smoke) | 10 | 1 | 2 | 0.5 | m/2 = 5 | 10 |
| Paper (NQ) | 50 | 1 | 2 (Vote) / 2 (Sparse) | 0.5 | m/2 = 25 | 10–40 |
| High utility | 50 | 1 | 5 | 0.5 | m/2 = 25 | ≥40 |

YAMLs live under [`experiments/configs/`](../../experiments/configs/):
`full/dpvoterag.yaml`, `full/dpsparsevoterag.yaml` (full sweep) and
`paper_faithful/dpvoterag.yaml`, `paper_faithful/dpsparsevoterag.yaml`
(paper LLMs + datasets).
