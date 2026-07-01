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
- `epsilon_0`: per-token budget (paper `eps_token ∈ {0.5, 1, 2}`).
- `vote_threshold`: SVT threshold `theta` (default `m·k/2`).
- `t_max`: max decoding steps.
- `threshold` (**MURAG**): fixed public relevance threshold `tau` (cosine scale).
- `epsilon_thr` + `score_min`/`score_max`/`bin_size` (**MURAG-ADA**): threshold-
  release budget and the score-bin grid for the prefix sums.

Sweep: `epsilon_q_values` (per-query budget). The headline `epsilon = M·eps_q` is
written to each result JSON (`epsilon` field, read by `score_results.py`).

## Deviations from the paper (deliberate, documented)

- **Bins (MURAG-ADA)**: the paper bins DPR inner products in `[70, 100] @ 0.2`
  (B = 150). This benchmark's FAISS index uses cosine similarity, so the bin range
  and size are exposed as config knobs on the cosine scale (default `[0, 1] @ 0.05`).
- **Fixed threshold (MURAG)**: `tau` is likewise on the cosine scale, a public,
  tunable knob that controls how many documents are charged per query.
- **Seeds**: the paper reports no random seeds or repeated trials. This framework
  uses the shared convention — `random_seed` (default 42) plus a per-query offset
  derived from `hash(query)` for the Laplace / exponential-mechanism draws — so runs
  are reproducible (the ε-DP guarantee is a property of the sampling distribution and
  is unaffected). Same caveat as DP-RAG / DP-KSA / DP-SynRAG.
- **Datasets**: the paper uses NQ, TriviaQA, MQuAKE and ChatDoctor. MQuAKE is not in
  this repo; the harmonized 6-dataset set (shared across all methods) is used.
- **Privacy unit**: one indexed entry (one row of `embeddings_matrix`) is treated as
  one document `z`, matching the other per-query methods.

## Performance Note

Each private decoding step runs `m + 1` forward passes (m voters + 1 baseline), like
DPSparseVoteRAG. With `M = 1` each document is consumed by at most one query, so a
long query stream progressively exhausts the corpus; the runner logs how many
documents are tracked / exhausted per operating point.
