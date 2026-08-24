import numpy as np
from sentence_transformers import SentenceTransformer

from core.device import cuda_oom_warning  # also ensures load_dotenv() runs
from core.embedder import BaseEmbedder
from core.query_embedder import BaseQueryEmbedder
from vanilla.embedder import HFEmbedder


class HFQueryEmbedder(BaseQueryEmbedder):
    """Query embedding with a HuggingFace sentence-transformers model.

    Takes the model already loaded for the corpus rather than a model id, because a
    single-tower retriever needs both stages to run through the same weights: two
    ``SentenceTransformer`` instances of the same id would double the embedder's GPU
    footprint to hold identical parameters. Passing an id instead loads a second
    tower, which is what an asymmetric two-encoder retriever wants.
    """

    def __init__(self, *, model: SentenceTransformer | None = None, model_name: str = ""):
        """Initialize the query embedder.

        Args:
            model: A loaded SentenceTransformer to encode queries with. When omitted,
                   ``model_name`` is loaded as a tower of its own.
            model_name: Model identifier from HuggingFace model hub. Records which
                        model the vectors come from, and is what gets loaded when no
                        model instance is supplied.
        """
        if model is None:
            if not model_name:
                raise ValueError("HFQueryEmbedder needs either a model or a model_name.")
            with cuda_oom_warning(model_name):
                model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.model = model
        # Query embeddings are a deterministic function of the query text under this
        # instance's model, so repeats of the same query resolve from here. The cache
        # belongs to the instance because the same text embeds differently under a
        # different model, and it is bounded because callers sweep whole datasets of
        # distinct queries through it.
        self._query_cache: dict[str, np.ndarray] = {}

    _QUERY_CACHE_MAX = 4096

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query.

        Args:
            query: Query string to embed.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        cached = self._query_cache.get(query)
        if cached is None:
            # Asymmetric retrievers (e.g. arctic-embed) register a "query" prompt that
            # prefixes the query but not the documents; their default_prompt_name is null,
            # so encode() applies it only when asked. Symmetric models register no such
            # prompt, so guard on its presence and fall back to a bare encode.
            kwargs = {}
            prompts = getattr(self.model, "prompts", None)
            if prompts and "query" in prompts:
                kwargs["prompt_name"] = "query"
            cached = self.model.encode([query], convert_to_numpy=True, **kwargs)[0].astype(np.float32)
            if len(self._query_cache) >= self._QUERY_CACHE_MAX:
                # Insertion-ordered dicts make the oldest entry the first key.
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[query] = cached
        # Callers normalise the vector they receive in place (FAISS retrieval does),
        # so each call gets its own array and the stored one stays as encoded.
        return cached.copy()


class SymmetricQueryEmbedder(BaseQueryEmbedder):
    """Embeds queries exactly as the corpus embedder embeds documents.

    The single-tower case with no query-side prompt: question and passage go through
    the same weights and the same call, so the query vector is just a one-element
    batch through the corpus embedder. Implementations that do treat queries
    differently — an asymmetric prompt, a separate tower — carry that in their own
    ``BaseQueryEmbedder`` instead of here.
    """

    def __init__(self, embedder):
        """Initialize with the corpus embedder whose encoding is being reused.

        Args:
            embedder: BaseEmbedder whose ``embed`` defines the shared vector space.
        """
        self.embedder = embedder

    def embed_query(self, query: str) -> np.ndarray:
        vectors = self.embedder.embed([query])
        return np.asarray(vectors, dtype=np.float32).reshape(-1)


def build_query_embedder(embedder: BaseEmbedder) -> BaseQueryEmbedder:
    """Construct the query-side tower for a run from the corpus-side one.

    The benchmark standardises on a single-tower retriever, so the query stage reuses
    the corpus stage's weights; what it does not reuse is the call, since an
    asymmetric model prefixes queries with a prompt its passages never receive.

    An embedder that carries no sentence-transformers model — a stub, or a tower that
    encodes some other way — has no such prompt to apply either, so its queries go
    through the same call as its documents.

    Raises:
        TypeError: If an ``HFEmbedder`` has no loaded model. Encoding its queries
            symmetrically would drop the query prompt, and a dropped prompt is
            invisible in the answer: it shows up only as retrieval that quietly got
            worse. Falling back is right for a tower that never had a prompt, never
            for one that lost it.
    """
    # Dispatch on what the embedder carries, not on which class it is: an embedder
    # that loads a SentenceTransformer without subclassing HFEmbedder still needs the
    # asymmetric call, and routing it symmetrically would drop the prompt in silence.
    model = getattr(embedder, "model", None)
    if isinstance(model, SentenceTransformer):
        return HFQueryEmbedder(model=model, model_name=getattr(embedder, "model_name", ""))
    if isinstance(embedder, HFEmbedder):
        raise TypeError(
            f"{type(embedder).__name__} carries no loaded SentenceTransformer "
            f"(got {type(model).__name__}); the query tower has no weights to reuse."
        )
    return SymmetricQueryEmbedder(embedder)
