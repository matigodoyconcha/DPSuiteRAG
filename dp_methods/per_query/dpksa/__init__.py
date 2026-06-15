"""DP-KSA: Differentially Private Keyword Selection and Aggregation.

Tang, Flemings, Wang & Annavaram, "Differentially Private Retrieval-Augmented
Generation", arXiv:2602.14374 (2026).

Per-query method: the top-N retrieved documents each generate a short response
(sample-and-aggregate), a token-frequency histogram is built, FindBestK selects
the number of keywords via the exponential mechanism, TopKWithPTR releases the
exact top-k keywords under a Gaussian Propose-Test-Release test, and the final
answer is generated from the query plus the privately released keywords only.

Key components:
- DPKSAGenerator: runs the full M_KSA(x, D) mechanism for one query.
- DPKSAPipeline: retrieves top-N and delegates to the generator.
"""

from .generator import DPKSAGenerator
from .pipeline import DPKSAPipeline

__all__ = [
    "DPKSAGenerator",
    "DPKSAPipeline",
]
