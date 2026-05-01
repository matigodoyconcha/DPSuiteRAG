"""DPVoteRAG: Voting-based Differentially Private RAG.

Implements differential privacy through voting aggregation on retrieved documents.
Multiple independent model outputs are aggregated via voting to provide privacy
guarantees while maintaining answer quality.

Key components:
- VoteRAGGenerator: Generates responses via voting on multiple document sets
- SparseVoteRAGGenerator: Sparse voting with agreement thresholds for efficiency
"""

from .generator import VoteRAGGenerator
from .sparse_vote_generator import SparseVoteRAGGenerator

__all__ = [
	"VoteRAGGenerator",
	"SparseVoteRAGGenerator",
]
