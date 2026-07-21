import random
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

	# ---- Membership inference attack (MIA) support -------------------------
	# A real MIA needs a member / non-member split: documents present in the
	# retrieval index (members) versus comparable documents held out of it
	# (non-members). This is orthogonal to the utility queries and is only used
	# by the standalone MIA run (experiments/run via BENCHDP_MIA_ONLY), so it
	# never changes what an ordinary utility run indexes.

	# Documents shorter than this (in whitespace tokens) are too small to split
	# into a query prefix and a held-out continuation, so a dataset needs at
	# least a handful of longer documents to support the generic MIA split.
	MIA_MIN_DOC_TOKENS = 4

	def _mia_probe_from_doc(self, doc: str) -> tuple[str, str]:
		"""Split one document into (query, held-out reference) for S2MIA.

		The generic split uses the first half of the document as the query and
		the second half as the held-out continuation the RAG system is expected
		to reproduce only when the document is indexed. Datasets with a native
		question/answer structure (e.g. ChatDoctor) override this to use the
		question as the query and the answer as the held-out reference, matching
		the DPVoteRAG setup.
		"""
		words = str(doc).split()
		if len(words) < self.MIA_MIN_DOC_TOKENS:
			return str(doc), str(doc)
		cut = max(1, len(words) // 2)
		return " ".join(words[:cut]), " ".join(words[cut:])

	def get_mia_split(
		self,
		n_per_class: int = 100,
		index_size: int | None = 2000,
		seed: int = 42,
	) -> dict | None:
		"""Return a balanced member / non-member split for the MIA run.

		Args:
			n_per_class: number of member and of non-member probes to sample.
			index_size: cap on the member corpus size actually indexed (keeps the
				MIA index small and independent of the full utility corpus); pass
				``None`` to index every member document.
			seed: RNG seed for the partition.

		Returns:
			A dict with keys:
			  - ``index_documents``: documents to index (the member corpus).
			  - ``members``: probes whose source doc IS in ``index_documents``.
			  - ``non_members``: probes whose source doc is NOT indexed.
			Each probe is ``{"query": str, "gold": str, "doc": str}``.
			Returns ``None`` when the dataset has too few usable documents to form
			a split, in which case the MIA run skips this dataset.
		"""
		seen: dict[str, None] = {}
		for raw in self.get_documents():
			text = str(raw).strip()
			if text and len(text.split()) >= self.MIA_MIN_DOC_TOKENS:
				seen.setdefault(text, None)
		docs = list(seen.keys())
		if len(docs) < 2 * n_per_class:
			# Fall back to as many probes per class as the corpus allows, but keep
			# at least one member and one non-member or there is no attack to run.
			n_per_class = len(docs) // 2
		if n_per_class < 1:
			return None

		rng = random.Random(seed)
		rng.shuffle(docs)

		non_member_docs = docs[:n_per_class]
		member_pool = docs[n_per_class:]
		if index_size is not None:
			member_pool = member_pool[:index_size]
		if not member_pool:
			return None
		member_docs = member_pool[:n_per_class]

		def probe(doc: str) -> dict:
			query, gold = self._mia_probe_from_doc(doc)
			return {"query": query, "gold": gold, "doc": doc}

		return {
			"index_documents": member_pool,
			"members": [probe(d) for d in member_docs],
			"non_members": [probe(d) for d in non_member_docs],
		}
