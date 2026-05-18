"""DP-RAG: Differentially Private Retrieval-Augmented Generation.

Reference: Grislain et al., "RAG with Differential Privacy", arXiv:2412.19291v2 (2024)

This module implements privacy-preserving RAG using exponential mechanism for
retrieval and differential privacy for generation (DP-ICL).

Key components:
- DPRetriever: Exponential mechanism for differentially private document retrieval
- DPICLGenerator: Privacy-preserving generation with in-context learning
- PerDocPromptBuilder: Builds separate prompts for each retrieved document
"""

from .retriever import DPRetriever, DPTopPRetriever
from .prompt_builder import PerDocPromptBuilder
from .generator import DPAggregatedGenerator, DPICLGenerator

__all__ = [
	"DPRetriever",
	"DPTopPRetriever",
	"PerDocPromptBuilder",
	"DPAggregatedGenerator",
	"DPICLGenerator",
]
