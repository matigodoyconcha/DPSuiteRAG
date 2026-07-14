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
	Number of selected documents used to build per-document prompts.

- `max_docs`:
	Maximum number of retrieved documents kept after threshold filtering.

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

Per the paper, δ = 1e-3 is used in experiments.

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
output JSON (e.g. ≈5.3 at the paper's operating point eps_r=0.5/eps_g=5.0/T=70).
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

## Deviations from the paper / reference implementation

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
- **Generation temperature**: `temperature` is set to `0.3` rather than the paper/reference
  default `T=1.0`, making token sampling fairly deterministic. `T` is the only knob that
  affects the actual sampling distribution (`softmax(U_ICL/T)`); lowering it sharpens
  generation and reduces the empirically observed noise, while the reported `(epsilon,
  delta)` accounting is unchanged. (`experiments/configs/full/dp_rag.yaml` and
  `experiments/configs/paper_faithful/dp_rag.yaml`, `temperature`)
- **Extended `epsilon_generation` sweep**: the paper reports a single operating point
  (`eps_r=0.5`, `eps_g=5.0`, giving `eps_total≈5.3`). This benchmark keeps that as the
  headline value and adds a sweep (`eps_g` in `{2.0, 7.5, 10.0}`) around it to extend the
  epsilon-utility tradeoff curve, while staying centered on the paper's value.
  (`experiments/configs/full/dp_rag.yaml` and `experiments/configs/paper_faithful/dp_rag.yaml`,
  `epsilon_generation_values`)

## Performance Note

Generation is token-by-token and requires `O(K * T)` forward computations.
For example, `K=5` and `T=64` implies about `320` forward passes per query.
This is expected to be slow on CPU.
