# DP-RAG

This module implements DP-RAG (RAG with Differential Privacy) for the BenchDP-RAG framework.

Reference paper:
- Nicolas Grislain et al., Sarus Technologies (2024), arXiv:2412.19291

Reference repository:
- https://github.com/sarus-tech/dp-rag

## What Changes vs Vanilla RAG

Compared to vanilla RAG, this method modifies three components:

1. Retriever (`DPRetriever` or `DPTopPRetriever`):
	 - Uses a noisy private threshold instead of deterministic top-k.
	 - Retrieves candidates with similarity search, then applies a DP threshold.
	 - `DPRetriever`: top-k exponential mechanism (selects k documents by count).
	 - `DPTopPRetriever`: top-p exponential mechanism (selects by weighted fraction p).

2. Prompt builder (`PerDocPromptBuilder`):
	 - Builds k+1 prompts: one per retrieved private document, plus one public prompt.
	 - The public prompt carries no retrieved context: it is a separate
	   `public_template` (default: `PerDocPromptBuilder.PUBLIC_TEMPLATE`, a fixed
	   no-context instruction; `run_dp_rag.py` overrides it per dataset via
	   `dataset.get_task_instruction()`).
	 - Serializes all prompts as a JSON string to keep pipeline compatibility.

3. Generator (`DPAggregatedGenerator`):
	 - Implements DP In-Context Learning (DP-ICL) mechanism.
	 - Token-by-token generation with formal DP guarantees.
	 - Per-token transformations on private logits: l_norm → center → clip.
	 - Public document logits are used as prior (weighted by θ), without transformation.
	 - Stochastic token sampling from exponential mechanism: r ∝ exp(ε·U_ICL/(2C)).

## Public Document

The DP-ICL utility combines a public prior with private document contributions:

```
U_ICL(r) = θ · ln(L_pub(r)) + Σⱼ l_clipped_j(r)
```

`L_pub` is computed from a **non-private** prompt that carries no retrieved
context. In the paper this stands in for optional public background information.
In this implementation, `PerDocPromptBuilder` appends a prompt built from
`public_template` (default: `PerDocPromptBuilder.PUBLIC_TEMPLATE`, a fixed
"answer with no context" instruction; `run_dp_rag.py` overrides this default
with a dataset-specific instruction, `dataset.get_task_instruction()`, on every
run) as the last entry in the list. The generator always treats the last prompt
as public.

- θ=0: public term has no effect (only private documents influence generation).
- θ large: generation is pulled toward the LLM's prior without private context.

## Parameters

- `epsilon_retrieval`:
	DP budget used by retrieval thresholding.

- `epsilon_generation`:
	TOTAL DP budget for generating the whole answer (all `max_new_tokens`
	tokens), not a per-token budget. The per-token budget (`token_epsilon`) is
	calibrated internally via PLD binary search so that composing
	`max_new_tokens` such mechanisms yields exactly this value.

- `K`:
	Number of selected documents used to build per-document prompts. Under
	`retriever_type: top_p` this is not configured: it is however many documents
	cleared the sampled threshold, capped by `max_docs`. `k_retrieved` applies
	only to `retriever_type: top_k`.

- `max_docs`:
	Maximum number of retrieved documents kept after threshold filtering, applied
	after the release is shuffled.

## Privacy Budget

Each individual mechanism is **pure DP** (ε, 0)-DP — the exponential mechanism
does not use delta. `delta` enters only through PLD (Privacy Loss Distribution)
composition, which turns the chain of pure-DP mechanisms into a tighter
(ε_total, δ)-DP guarantee:

- **Headline `epsilon_total`**: a single PLD composition of the retrieval
  mechanism (`eps_r`) with `T` per-token generation mechanisms (`token_epsilon`),
  computed by `pld_dp_rag_total` (`metrics/privacy_accounting.py`). This is the
  number to report.
- **`naive_sum`**: the loose union bound `eps_r + eps_g`, reported alongside the
  headline for comparison only — see "Which ε to read" below.

where `T` is the number of generated tokens (`max_new_tokens`). `delta` is used
by the PLD binary-search calibration inside `DPICLGenerator`
(`_compute_token_epsilon`, which derives `token_epsilon` from
`epsilon_generation`) and by `pld_dp_rag_total`; the raw exponential-mechanism
sampling step itself (`_sample_token`) takes no delta term.

`delta: 1.0e-3` in the config is a **mechanism knob, not a reporting figure**.
It sets how much noise the calibration buys: at `eps_g=5` and `T=70`,
`token_epsilon` is 0.137 at δ=1e-5, 0.175 at 1e-3 and 0.212 at 1e-2 — a 54%
swing under one unchanged `eps_g` label. It is also spent *per query*, so it
cannot be set to the shared axis' δ: at Q=100 a per-query 1e-2 composes to a
workload δ near 1.

### The shared workload axis

`epsilon_total` above is DP-RAG's own number, in DP-RAG's own units (one query,
PLD composition, δ=1e-3). It is not comparable with the numbers the other seven
methods report. Every result payload therefore also carries the shared figure
produced by `workload_accounting_block` (`experiments/common.py`):

    epsilon_workload = privacy loss over a workload of Q questions at δ=1e-2

computed by `metrics/workload_accounting.py::method_epsilon_total`, which
composes one retrieval exponential mechanism plus `max_new_tokens` per-token
exponential mechanisms as an RDP curve and converts once at the end. This is the
axis the sweep is derived against: the `full/` config declares
`epsilon_targets: [1, 5, 10, 50]` and `workload_queries: [1, 5, 25, 100]`, and
for each (target, Q) the runner inverts the axis (`budget_for_target`) for a
per-token budget, from which the retrieval budget follows through the paper's
fixed proportion. A `paper_faithful/` config names `epsilon_generation_values`
in per-query units instead, and those are converted the way the generator would.
`analysis/derive_epsilon_grids.py` performs the same inversion offline for
inspecting a grid.

## Sensitivity Notes

- Retrieval mechanism: exponential mechanism with Delta=1
	`sen_retrieval = 1.0`

- Generation mechanism: DP-ICL sensitivity equals the clipping parameter C.
  Each document contributes at most C to U_ICL (guaranteed by the clipping step),
  so changing one private document shifts the utility by at most C.
  This is why the exponential mechanism uses ε/(2C) as the scaling factor.
	`sen_generation = C`, derived per run as `C = token_epsilon · temperature / 2`
	— not a fixed constant (see `DPICLGenerator.__init__`).

## Important Notes

- The headline `epsilon_total` is a single PLD composition of
  `retrieval(eps_r) ⊕ T × generation(token_epsilon)` — see
  `metrics/privacy_accounting.py::pld_dp_rag_total`.
- Delta set to 1e-3 per paper specification.
- The `naive_sum` bound (`eps_r + eps_g`) reported alongside the headline is
  looser (larger) because `eps_g` is already the PLD-calibrated total for the
  whole `T`-token generation; summing it again with `eps_r` double-counts
  relative to composing the two mechanisms once. Read the **calibrated
  `epsilon_total`** as the headline.

### Which ε to read

`epsilon_generation` (`eps_g`) is the TOTAL budget for the whole answer: the
per-token `token_epsilon` is calibrated internally (PLD binary search) so that
composing `max_new_tokens` such mechanisms yields exactly `eps_g`. The headline
`epsilon_total` is **not** `eps_r + eps_g` — it is a single PLD composition of
`retrieval(eps_r)` with the `T`-token generation mechanism, computed by
`pld_dp_rag_total` and reported as `privacy_accounting.pld.epsilon` in the
output JSON (≈5.3 at the paper's operating point, `eps_r=0.5`, `eps_g=5.0`,
`T=70`, which is what `paper_faithful/dp_rag.yaml` runs).
The output JSON also reports `privacy_accounting.naive_sum.epsilon = eps_r +
eps_g` (e.g. 5.5) as a loose upper bound kept for comparison only — it is never
the number to report as the headline. The reference `sarus-tech/dp-rag` also
uses PLD composition, which is what `epsilon_total` is calibrated against (to
within the search tolerance `tol≈1e-3`).

### Deterministic seeding (reproducibility caveat)

Retrieval and per-token generation draw their exponential-mechanism randomness from
an RNG seeded deterministically from `random_seed` + a per-query/per-step offset
(`retriever.py`, `generator.py`). This makes runs **reproducible**, which is intended
for benchmarking. The ε-DP guarantee is a property of the sampling *distribution* and
is unaffected, but a fixed seed means identical inputs yield identical outputs rather
than fresh draws — not a deployment-grade randomness assumption. Same caveat as
DP-KSA / DP-SynRAG.

## Deviations from the paper

Two configs run this method, and they deviate from the paper for different
reasons. `experiments/configs/paper_faithful/dp_rag.yaml` reproduces the paper's
own setting: the medical corpus alone, `max_new_tokens: 70`, `eps_r=0.5`,
`eps_g` around 5.0. `experiments/configs/full/dp_rag.yaml` runs the benchmark's
**common evaluation protocol**, which is shared by all eight methods and
therefore overrides several of the paper's choices on purpose. Everything below
describes the full config unless it says otherwise.

### Generation temperature 0.3 instead of the paper's 1.0

`temperature: 0.3`, against `T=1.0` in both the paper and `sarus-tech/dp-rag`.
This is a tuned utility knob that no other method in the benchmark has, so it is
the deviation to be most careful about when reading DP-RAG against the others.

The mechanism is worth spelling out, because the knob does not appear in the
accounting at all. The clipping bound is not configured; it is derived after
`token_epsilon` is calibrated, as

    C = token_epsilon · T / 2

and substituting that into the per-token exponential mechanism
`r ∝ exp(token_epsilon · U_ICL / (2C))` cancels `token_epsilon` entirely and
leaves `softmax(U_ICL / T)`. So `T` rescales the utilities and nothing else:
lowering it to 0.3 sharpens the sampling distribution onto the top-utility
token and suppresses the *empirically observed* effect of the DP noise, while
`epsilon_total`, `token_epsilon` and `epsilon_workload` are all bit-for-bit
what they would be at `T=1.0`. Read as: at a fixed reported ε, this method's
answers are drawn from a sharper distribution than the paper's would be, and
part of any utility advantage over the other methods is bought here rather
than by the mechanism. (`experiments/configs/full/dp_rag.yaml`, `temperature`;
`generator.py`, `DPICLGenerator.__init__`)

### `retriever_type: top_p` releases a variable-size shuffled set

The paper's medical evaluation uses the top-p exponential mechanism, so this is
paper-faithful, but it makes DP-RAG the only method whose retrieval output is a
**set** rather than a ranking, and that has consequences for every retrieval
metric:

- The mechanism samples one threshold τ and returns *every* document scoring
  above it. The size of that set is data- and noise-dependent — from a single
  document (the `argmax` fallback when τ lands above every score) to the whole
  corpus.
- The set is then **shuffled** before being truncated to `max_docs: 128`.
  Shuffling is required: ordering the release by score would leak the scores the
  mechanism exists to hide. But it means that whenever more than 128 documents
  clear the threshold, the 128 that survive are a *uniform random* subset of
  them, not the best 128.
- Consequently `retrieval_precision_at_k` (`metrics/utility.py`) scores the
  whole released set against the non-private top-`k_retrieved`, never a prefix
  of it — a prefix of a shuffled set measures the shuffle, not the retrieval.
  The denominator stays `k`, so a method that returns one document does not
  score as perfect.
- Even scored correctly the metric is **not monotone in ε** for this method: a
  larger `eps_r` places τ more accurately at the p=0.02 mass point, which
  releases *more* documents, which makes the 128-document cap bind harder and
  drops more of the true top-k. On a simulated 20 000-document corpus with a
  realistic arctic cosine spread the value sits around 0.6–0.75 and drifts
  slightly downward as `eps_r` grows. It reports how much of the non-private
  ranking survives the release, not retrieval quality.

(`retriever.py`, `DPTopPRetriever.retrieve`; `experiments/configs/full/dp_rag.yaml`,
`retriever_type` / `top_p` / `max_docs`)

### Corpus, datasets and answer length

- **Datasets**: the paper evaluates on one corpus, the synthetic
  `sarus-tech/medical_dirichlet_phi3`. The common protocol runs six
  (medical, chatdoctor, natural_questions, triviaqa, searchqa, movielens) × 40
  questions × seven ≤4B LLMs.
- **Corpus size**: the paper states "around 5000 documents". The protocol caps
  every corpus at `max_index_docs: 20000`. Medical is left uncapped (`null`)
  because its train split holds about 10 000 rows, fewer than the cap, so the
  document set is identical either way; the two spellings do fork the index
  cache key (`experiments/common.py::corpus_fingerprint` keys on the raw value),
  so the medical index is built once for DP-RAG and once for the methods that
  write `20000`.
- **Answer length**: `max_new_tokens: 30` for every method that can express an
  answer length, against 70 in the reference's medical evaluation. A text metric
  should measure the answer, not the budget it was allowed to fill. For DP-RAG
  the length is also a privacy quantity: it is the number `T` of composed
  exponential mechanisms, so 30 tokens both shortens the answer and drops 40
  mechanisms from the workload epsilon.
- **No chunking**: one document is one privacy unit. Splitting a document into
  `c` chunks would multiply the guarantee by group privacy
  (`c·ε`, `c·e^{(c-1)ε}·δ`), which no document-level statement survives.

### The ε grid replaces the paper's operating point

The paper reports a single point (`eps_r=0.5`, `eps_g=5.0`, `eps_total≈5.3` at
δ=1e-3). The full config instead derives a budget for each cell of the shared
grid — targets {1, 5, 10, 50} crossed with workloads {1, 5, 25, 100} at δ=1e-2 —
which is what makes a DP-RAG point and another method's point the same privacy
level. The paper's operating point is far off that axis: charged once per query
and composed over 100 queries, `eps_r=0.5` alone reaches roughly 25, so the
derived retrieval budgets are two orders of magnitude smaller than the paper's.
The paper's point survives in `paper_faithful/`.

The proportion between the two budgets is what carries over from the paper's
point, and it is constant per unit of *per-token* generation budget. The split a
reader interprets, `eps_r / (eps_r + eps_g)`, is not: `eps_g` is a PLD
composition of `max_new_tokens` copies of the per-token budget and is sublinear
in it, so retrieval's share of the total rises as the axis tightens. Over the 16
shipped cells it runs 8.72 % – 24.33 % (lowest at ε_workload = 50, Q = 1; highest
at ε_workload = 1, Q = 100) against the paper's own 9.09 %, and every cell's
total still composes to its target exactly. Each cell records the split it ran at
in `epsilon_retrieval_share`.

Two δ values coexist and must not be confused. `delta: 1.0e-3` is the paper's,
and it is a per-query mechanism knob that decides how much noise the calibration
buys. δ=1e-2 is the reporting delta of the shared axis, chosen because the
per-token approximate-DP mass of the vote-based methods leaves no room below it.
DP-RAG's own δ is pure conversion slack — every one of its mechanisms is pure
ε-DP — so its PLD number can honestly be read at a far tighter δ than the shared
one, and both belong in the write-up.

### Known limitations of the shipped grid

- The pairs in `epsilon_retrieval_values` / `epsilon_generation_values` compose
  to 1.02, 3.05, 10.04, 30.00 and 101.38 on the shared axis at `T=50`, rather
  than exactly 1, 3, 10, 30 and 100 — the derivation is exact at `T=70` and the
  values are carried at 50. The worst point is off by 2.1%, which is inside the
  5% tolerance the grid tests use but outside the 2% band
  `derive_epsilon_grids.py` prints as a match. Re-deriving at `T=50` would give
  `eps_r = [0.006511, 0.015078, 0.037314, 0.078857, 0.16645]` with
  `eps_g = [0.075691, 0.207861, 0.61204, 1.487105, 3.655969]`.
- `tests/test_shared_grid.py` covers dpvoterag, dpsparsevoterag, dpksa,
  private_rag, private_rag_ada and dp_synrag. DP-RAG has no case there, so this
  drift is not caught by the suite.
- `analysis/derive_epsilon_grids.py::_dp_rag_retrieval_ratio` reads the
  retrieval-to-generation proportion out of the config's *own* first grid entry,
  which is itself a derived value rather than the paper's operating point.
  The derivation is therefore not idempotent: the paper's proportion is 2.85,
  and re-reading it from the shipped grid yields 1.02, so each regeneration
  shrinks retrieval's share of the budget.
- `run_dp_rag.py` iterates `product(eps_r_values, eps_g_values)`, so it runs all
  25 combinations while only the 5 paired ones sit on the axis. The 20
  off-diagonal cells compose to arbitrary workload epsilons (1.29, 2.56, 6.03,
  9.36, 92.80, …) and cost 80% of the run — a 5× multiplier on the most
  expensive method in the benchmark, which already needs `O(K·T)` forward passes
  per query. Cross-method plots must select the diagonal by index.

### Retriever and score window

- **Embedder (standardized, and here also paper-faithful)**: retrieval uses the
  benchmark's shared cross-method embedder, `Snowflake/snowflake-arctic-embed-m-v1.5`
  (single-tower, cosine, L2-normalized), so the retriever is not a confound in the
  cross-method comparison. For DP-RAG this shared choice happens to coincide with the
  reference (`pup_vector_store.py` uses the same arctic-embed model with CLS pooling +
  L2-normalization), so — unlike methods whose papers used a different retriever — the
  standardization does not trade away paper-faithfulness here. The one residual fidelity
  caveat is that `HFEmbedder`'s pooling may not exactly reproduce the reference's
  CLS-token pooling. (`experiments/configs/full/dp_rag.yaml` and
  `experiments/configs/paper_faithful/dp_rag.yaml`, `models.embedder`)
- **Top-p score window not recalibrated per dataset**: `DPTopPRetriever`'s threshold window
  (`min_score=-0.5`, `max_score=0.8`) is a fixed constant carried over from the reference,
  calibrated there for arctic-embed's cosine distribution on the medical corpus, and is kept
  as a shared convention rather than re-derived per embedder or dataset. For the harmonized
  sweep over chatdoctor, natural_questions, triviaqa, searchqa and movielens, the window is
  not recalibrated, so retrieval may select too few or too many documents on those datasets.
  (`retriever.py`, `DPTopPRetriever`; `experiments/configs/full/dp_rag.yaml`, `datasets`)

## Performance Note

Generation is token-by-token and requires `O(K * T)` forward computations, where
`K` is the released set size plus the public prompt. Under the common protocol
`T=50` and the top-p release runs up to `max_docs=128`, so a single query can
cost about `6450` forward passes; `use_batch_forward: true` batches the `K`
prompts of one decoding step into one call, which is what keeps this tractable.
The runner calls `free_memory()` after every query for the same reason. This is
by far the most expensive method in the benchmark and is not usable on CPU.
