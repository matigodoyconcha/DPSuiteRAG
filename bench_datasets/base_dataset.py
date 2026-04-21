from abc import ABC, abstractmethod


class BaseDataset(ABC):
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
