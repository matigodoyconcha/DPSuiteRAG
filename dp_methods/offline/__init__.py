"""Offline (indexing-time) DP methods.

These methods spend the entire privacy budget once during preprocessing/indexing.
Inference at query time is non-private (or uses no additional ε).

Methods:
- dp_synrag: DP budget spent once to build a synthetic corpus
- lprag:     DP noise applied at chunking time (entity perturbation)
- rp_rag:    Random-projection indexing (no formal ε; obfuscation only)
"""
