# DPVoteRAG / DPSparseVoteRAG

Implementation of the DPVoteRAG and DPSparseVoteRAG methods described in
Koga et al., *"Privacy-Preserving Retrieval-Augmented Generation with Differential
Privacy"* ([arXiv:2412.04697](https://arxiv.org/html/2412.04697v3)).

This implementation follows the official repository
[tacchan7412/DPRAG](https://github.com/tacchan7412/DPRAG) for the LD-Gumbel
mechanism and the RDP accountant. Two budget choices on the Sparse path follow the
**paper** rather than the reference *code* — they are deliberate, benchmark-motivated
design decisions, flagged inline below: the `sparse_fraction` split (§3.3) and the
online SVT-epoch charging.

## Components

- [`generator.py`](generator.py) — `VoteRAGGenerator` with the
  Limited Domain Gumbel mechanism (Durfee & Rogers, NeurIPS 2019).
- [`sparse_vote_generator.py`](sparse_vote_generator.py) — `SparseVoteRAGGenerator`
  extending the above with the Sparse Vector Technique (Dwork & Roth).
- [`pipeline.py`](pipeline.py) — RAG orchestration (embedder, indexer, retriever,
  generator, postprocessor).

The `(ε,δ)` accounting is delegated to
[`metrics/privacy_accounting_autodp.py::AutodpLDAccountant`](../../metrics/privacy_accounting_autodp.py).
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
| `tmax_absolute` | Maximum tokens per response | 50 |
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
`flag=None` is recorded in `dp_flags_per_token`.

### Additional parameters

| Parameter | Description | Default |
|---|---|---|
| `sparse_fraction` | Fraction of `epsilon_token` allocated to SVT | 0.5 |
| `tau` | Agreement threshold (default `m/2`) | `m/2` |
| `exclude_stopwords` | Enable stopword exclusion | False |
| `exclude_question_words` | Enable question-token exclusion | False |
| `stopwords_path` | Path to `stopword_words.pt` | `data/stopword_words.pt` |
| `tmax_absolute` | Maximum tokens per response | 200 |

### Additional output

- `dp_flags_per_token`: per token, `True` = private LD, `False` = SVT public,
  `None` = excluded.
- `n_private_tokens`, `n_nonprivate_tokens`, `n_excluded_tokens`, `total_tokens`.

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

## Recommended hyperparameters

| Regime | m | k | epsilon_token | sparse_fraction | τ | epsilon_total |
|---|---|---|---|---|---|---|
| Quick (smoke) | 10 | 1 | 2 | 0.5 | m/2 = 5 | 10 |
| Paper (NQ) | 30 | 1 | 2 (Vote) / 2 (Sparse) | 0.5 | m/2 = 15 | 10–40 |
| High utility | 50 | 1 | 5 | 0.5 | m/2 = 25 | ≥40 |

YAMLs live in [`experiments/configs/`](../../experiments/configs/):
`dpvoterag.yaml`, `dpsparsevoterag.yaml` (full sweep) and their
`_paper_faithful.yaml` counterparts (paper LLMs + datasets).
