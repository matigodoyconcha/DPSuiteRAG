"""Per-query epsilon DP methods.

These methods consume a privacy budget (ε) on every query at inference time.

Methods:
- dp_rag:       Exponential mechanism on retrieval + DP-ICL on generation
- dpvoterag:    Limited-Domain Gumbel voting per token (with online accountant)
- private_rag:  Budget-aware retrieval with Rényi DP composition across queries
"""
