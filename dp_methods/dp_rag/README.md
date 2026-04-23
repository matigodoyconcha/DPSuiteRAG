# DP-RAG

This module implements DP-RAG (RAG with Differential Privacy) for the BenchDP-RAG framework.

Reference paper:
- Nicolas Grislain et al., Sarus Technologies (2024), arXiv:2412.19291

Reference repository:
- https://github.com/sarus-tech/dp-rag

## What Changes vs Vanilla RAG

Compared to vanilla RAG, this method modifies three components:

1. Retriever (`DPRetriever`):
	 - Uses a noisy private threshold instead of deterministic top-k.
	 - Retrieves candidates with similarity search, then applies a DP threshold.

2. Prompt builder (`PerDocPromptBuilder`):
	 - Builds one prompt per retrieved document.
	 - Serializes all prompts as a JSON string to keep pipeline compatibility.

3. Generator (`DPAggregatedGenerator`):
	 - Performs token-by-token DP aggregation of per-document next-token probabilities.
	 - Adds Laplace noise at each generation step and renormalizes before decoding.

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

With basic composition:

`epsilon_total <= epsilon_retrieval + T * epsilon_generation`

where `T` is the number of generated tokens.

## Sensitivity Notes

- Retrieval mechanism sensitivity:
	`sen_retrieval = 1.0`

- Generation mechanism sensitivity:
	`sen_generation = 2 / K`

## Important Notes

- This implementation reports privacy with basic composition only.
- It does not include advanced composition or RDP accounting.
- For tighter privacy accounting in production, use RDP composition.

## Performance Note

Generation is token-by-token and requires `O(K * T)` forward computations.
For example, `K=5` and `T=64` implies about `320` forward passes per query.
This is expected to be slow on CPU.
