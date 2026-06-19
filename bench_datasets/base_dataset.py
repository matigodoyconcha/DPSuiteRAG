from abc import ABC, abstractmethod


class BaseDataset(ABC):
	# One-line task instruction for this dataset. The TASK depends on the DATASET
	# (short-answer QA, medical advice, movie recommendation, ...), NOT on the DP
	# method, so every method reuses the same instruction for a given dataset —
	# fair cross-method comparison — while each task gets an appropriate prompt.
	# Subclasses override TASK_INSTRUCTION only; the templates below are derived
	# from it, so heterogeneous prompt structures (context-augmented, keyword-
	# conditioned, no-context public) all stay consistent with the same task.
	TASK_INSTRUCTION = "Give a simple short answer for the question based on the context"

	def get_task_instruction(self) -> str:
		"""Return the one-line, self-contained task instruction for this dataset.

		Used for prompts that have no retrieved context (e.g. DP-VoteRAG's public
		no-context prompt, DP-KSA's keyword-conditioned prompt).
		"""
		return self.TASK_INSTRUCTION

	def get_prompt_template(self) -> str:
		"""Return the context-augmented prompt template for this dataset.

		Built from ``TASK_INSTRUCTION``. Placeholders ``{context}`` (retrieved
		chunks) and ``{query}`` (the question); ends with ``Answer:`` so
		incremental decoders (DP-VoteRAG) can append generated tokens after
		formatting. Dataset-scoped (not method-scoped) for fair comparison.
		"""
		return (
			f"{self.get_task_instruction()}\n"
			"Context: {context}\nQuestion: {query}\nAnswer:"
		)

	@abstractmethod
	def load(self) -> None:
		"""Load dataset resources into memory."""

	@abstractmethod
	def get_documents(self) -> list[str]:
		"""Return documents that will be indexed by the pipeline."""

	@abstractmethod
	def get_queries(self) -> list[dict]:
		"""Return query-answer pairs, each item should include query and answer keys."""

	@abstractmethod
	def get_private_entities(self) -> list[str]:
		"""Return known private entities for leakage metrics (empty for non-private datasets)."""
