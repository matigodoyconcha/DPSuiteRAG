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
	 - The public prompt uses a non-private context (configurable, default empty).
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

`L_pub` is computed from a **non-private** context `d_pub`. In the paper this is
optional public background information. In this implementation, `PerDocPromptBuilder`
appends an extra prompt with `public_context` (default: empty string) as the last
entry in the list. The generator always treats the last prompt as public.

- θ=0: public term has no effect (only private documents influence generation).
- θ large: generation is pulled toward the LLM's prior without private context.

## Parameters

- `epsilon_retrieval`:
	DP budget used by retrieval thresholding.

- `epsilon_generation`:
	DP budget used per generated token in probability aggregation.

- `K`:
	Number of selected documents used to build per-document prompts.

- `max_docs`:
	Maximum number of retrieved documents kept after threshold filtering.

## Privacy Budget

Each individual mechanism is **pure DP** (ε, 0)-DP — the exponential mechanism
does not use delta. Delta only enters when composing T mechanisms using advanced
or RDP composition, which yields a tighter (ε_total, δ)-DP guarantee:

- Basic composition (upper bound): `ε_total ≤ ε_r + T · ε_g`
- RDP / advanced composition: tighter bound computed by `metrics/privacy_accounting.py`

where `T` is the number of generated tokens. The `delta` parameter in
`DPICLGenerator` is passed to the external accountant; it is not used inside
the mechanism itself.

Per the paper, δ = 1e-3 is used in experiments.

## Sensitivity Notes

- Retrieval mechanism: exponential mechanism with Delta=1
	`sen_retrieval = 1.0`

- Generation mechanism: DP-ICL sensitivity equals the clipping parameter C.
  Each document contributes at most C to U_ICL (guaranteed by the clipping step),
  so changing one private document shifts the utility by at most C.
  This is why the exponential mechanism uses ε/(2C) as the scaling factor.
	`sen_generation = C  (clipping parameter, default 1.0)`

## Important Notes

- Includes basic, RDP, and advanced composition accounting.
- Delta set to 1e-3 per paper specification.
- Of the three recomposition methods, **advanced composition is typically the
  tightest in this regime (small per-token ε, large T) and RDP the loosest** — the
  RDP path uses the loose pure-DP bound `α·ε/(α−1)`. Read the **calibrated
  `epsilon_total`** as the headline (PLD-based, tighter than all three).

### Which ε to read

The per-token `epsilon_generation` is calibrated (PLD binary search) so that
composing `T` tokens yields exactly `eps_g`; the headline budget is
`epsilon_total = eps_r + eps_g`. The `privacy_accounting` block in the output JSON
**recomposes** the per-token ε with basic / RDP / advanced composition, which are
*alternative, looser* upper bounds — basic (`eps_r + T·eps_g`) in particular can
read much larger than the headline budget. All are valid (never under-report); in
this regime **advanced is the tightest of the three and RDP the loosest** (the RDP
pure-DP bound `α·ε/(α−1)` is loose for small per-token ε). The reference
`sarus-tech/dp-rag` uses PLD composition, tighter still — which is what the headline
`epsilon_total` is calibrated against (to within the search tolerance `tol≈1e-3`).
So the numbers differing across the accounting block is expected, not a
contradiction — compare against the calibrated `epsilon_total`.

### Deterministic seeding (reproducibility caveat)

Retrieval and per-token generation draw their exponential-mechanism randomness from
an RNG seeded deterministically from `random_seed` + a per-query/per-step offset
(`retriever.py`, `generator.py`). This makes runs **reproducible**, which is intended
for benchmarking. The ε-DP guarantee is a property of the sampling *distribution* and
is unaffected, but a fixed seed means identical inputs yield identical outputs rather
than fresh draws — not a deployment-grade randomness assumption. Same caveat as
DP-KSA / DP-SynRAG.

## Performance Note

Generation is token-by-token and requires `O(K * T)` forward computations.
For example, `K=5` and `T=64` implies about `320` forward passes per query.
This is expected to be slow on CPU.
