# DPVoteRAG / DPSparseVoteRAG

Implementation of the DPVoteRAG and DPSparseVoteRAG methods described in
Koga et al., *"Privacy-Preserving Retrieval-Augmented Generation with Differential
Privacy"* ([arXiv:2412.04697](https://arxiv.org/html/2412.04697v1)).

This implementation replicates the behaviour of the official repository
[tacchan7412/DPRAG](https://github.com/tacchan7412/DPRAG).

## Components

- [`generator.py`](generator.py) — `VoteRAGGenerator` with the
  Limited Domain Gumbel mechanism (Durfee & Rogers, NeurIPS 2019).
- [`sparse_vote_generator.py`](sparse_vote_generator.py) — `SparseVoteRAGGenerator`
  extending the above with the Sparse Vector Technique (Dwork & Roth).
- [`pipeline.py`](pipeline.py) — RAG orchestration (embedder, indexer, retriever,
  generator, postprocessor).

The `(ε,δ)` accounting is delegated to
[`metrics/privacy_accounting_autodp.py::AutodpLDAccountant`](../../metrics/privacy_accounting_autodp.py),
a direct port of `utils/dp_accountant.py` from the official repository (EM-peeling
with RDP composition, BBGHS conversion).

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
   - Otherwise → `NotFoundLDTop1`. Behaviour depends on `fail_mode`:
     - `'ld_pate'` (paper default, `utils/ensemble.py:127`): retries once
       with `k_bar = dim − 1` (1-excluded LD); if it fails again, stops
       (`exit_status=1`).
     - `'stop'`: stops generation immediately (`exit_status=1`).
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
| `fail_mode` | `'ld_pate'` (paper default) or `'stop'` | `'ld_pate'` |
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
4. **If `a_t_noisy ≤ τ̂` → applies LD-Gumbel** (private token, spends `epsilon_token`).
   The threshold `τ̂` is resampled only after a private step, following SVT.
5. **If `a_t_noisy > τ̂` → accepts the public token** (no LD cost).

### Optional exclusion of trivial tokens

- `exclude_stopwords`: stopwords loaded from `data/stopword_words.pt`
  (paper convention: the id is taken as `tokenizer(word)['input_ids'][1]`,
  skipping `<bos>`).
- `exclude_question_words`: ids of the question tokens (includes `<bos>`, matching the reference).

When a token falls in the exclusion list, it is accepted directly and
`flag=None` is recorded in `dp_flags_per_token`.

### Budget composition

- `sparse_eps`: SVT budget per step (accounted as `2·sparse_eps`
  per iteration, classic Dwork-Roth composition).
- `epsilon_token`: LD-Gumbel budget per private step.

The accountant sums both: `total_spent_epsilon = ε_LD + ε_SVT`.

### Additional parameters

| Parameter | Description | Default |
|---|---|---|
| `sparse_eps` | SVT budget (independent of `epsilon_token`) | 1.0 |
| `tau` | Agreement threshold (default `m/2`) | `m/2` |
| `exclude_stopwords` | Enable stopword exclusion | False |
| `exclude_question_words` | Enable question-token exclusion | False |
| `stopwords_path` | Path to `stopword_words.pt` | `data/stopword_words.pt` |
| `tmax_absolute` | Maximum tokens per response | 200 |

### Additional output

- `dp_flags_per_token`: per token, `True` = private LD, `False` = SVT public,
  `None` = excluded.
- `n_private_tokens`, `n_nonprivate_tokens`, `n_excluded_tokens`, `total_tokens`.

> Note on `sparse_eps`. The paper (Algorithm 2) splits `ε_token` into
> `(ε_token/2, ε_token/2)` for LD and Laplace. The official repository exposes them as
> two independent parameters (`--dp_eps`, `--sparse_eps`) and this
> implementation replicates that convention. To reproduce the paper strictly,
> set `sparse_eps = epsilon_token / 2`.

## Recommended hyperparameters

| Regime | m | k | epsilon_token | sparse_eps | τ | epsilon_total |
|---|---|---|---|---|---|---|
| Quick (smoke) | 10 | 1 | 2 | 1 | m/2 = 5 | 10 |
| Paper (NQ) | 30 | 1 | 2 (Vote) / 1 (Sparse) | 1 | m/2 = 15 | 10–40 |
| High utility | 50 | 1 | 5 | 1 | m/2 = 25 | ≥40 |

YAMLs live in [`experiments/configs/`](../../experiments/configs/):
`dpvoterag.yaml`, `dpsparsevoterag.yaml` and their `_quick` counterparts.
