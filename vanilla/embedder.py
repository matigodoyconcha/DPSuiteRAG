from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from core.device import resolve_device  # noqa: F401 — ensures load_dotenv() runs
from core.embedder import BaseEmbedder

class HFEmbedder(BaseEmbedder):
    """Embedding model using HuggingFace sentence-transformers.

    Provides efficient embedding generation using pre-trained transformer models
    from the sentence-transformers library.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the HuggingFace embedder.

        Args:
            model_name: Model identifier from HuggingFace model hub.
                       Defaults to all-MiniLM-L6-v2, a lightweight model.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            2D numpy array of embeddings with shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self.model.encode(texts, convert_to_numpy=True).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query.

        Args:
            query: Query string to embed.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        return self.model.encode([query], convert_to_numpy=True)[0].astype(np.float32)


class DPRDualEncoder(BaseEmbedder):
    """DPR dual encoder: separate context (passages) and question (queries) towers.

    DPR (Karpukhin et al., 2020) is natively a **dual** encoder — passages must be
    embedded with the *context* tower and queries with the *question* tower. The
    prior setup embedded both with a single question encoder, which makes the
    passage/query similarities meaningless (audit finding C4). Both towers share the
    same output space, so the existing cosine FAISS index/retriever are reused
    unchanged (vectors are L2-normalised at index time).

    Attributes:
        question_model: SentenceTransformer for queries.
        ctx_model: SentenceTransformer for passages/documents.
    """

    def __init__(self, question_model_name: str, ctx_model_name: str):
        self.question_model_name = question_model_name
        self.ctx_model_name = ctx_model_name
        self.question_model = SentenceTransformer(question_model_name)
        self.ctx_model = SentenceTransformer(ctx_model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed passages/documents with the context tower."""
        if not texts:
            return np.empty((0, self.ctx_model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self.ctx_model.encode(texts, convert_to_numpy=True).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query with the question tower."""
        return self.question_model.encode([query], convert_to_numpy=True)[0].astype(np.float32)


def derive_ctx_encoder(question_model_name: str) -> str | None:
    """Derive the matching DPR context-encoder id from a question-encoder id.

    Returns ``None`` when the id does not contain ``question_encoder`` (so the
    caller can require an explicit ``ctx_encoder`` instead of guessing).
    """
    if "question_encoder" in question_model_name:
        return question_model_name.replace("question_encoder", "ctx_encoder")
    return None


def build_embedder(
    embedder_model: str,
    *,
    dual_encoder: bool = False,
    ctx_encoder: str | None = None,
) -> BaseEmbedder:
    """Construct the embedder for a run.

    Args:
        embedder_model: the (query) embedder id. Also the single-tower id when
            ``dual_encoder`` is False.
        dual_encoder: when True, build a :class:`DPRDualEncoder` (C4 retriever fix).
        ctx_encoder: explicit context-tower id; when omitted it is derived from
            ``embedder_model`` by replacing ``question_encoder`` → ``ctx_encoder``.

    Returns:
        An :class:`HFEmbedder` (single tower) or :class:`DPRDualEncoder`.
    """
    if not dual_encoder:
        return HFEmbedder(model_name=embedder_model)
    ctx = ctx_encoder or derive_ctx_encoder(embedder_model)
    if not ctx:
        raise ValueError(
            "dual_encoder=True needs models.ctx_encoder, or an embedder id "
            f"containing 'question_encoder' to derive it (got '{embedder_model}')."
        )
    return DPRDualEncoder(question_model_name=embedder_model, ctx_model_name=ctx)


def embedder_identity(
    embedder_model: str,
    *,
    dual_encoder: bool = False,
    ctx_encoder: str | None = None,
) -> str:
    """Stable identity string for cache keys.

    Distinguishes a dual-encoder run from a single-tower run so cached *document*
    embeddings (produced by the context tower) are not reused across the two.
    """
    if not dual_encoder:
        return embedder_model
    ctx = ctx_encoder or derive_ctx_encoder(embedder_model) or "ctx"
    return f"dual::q={embedder_model}::ctx={ctx}"
