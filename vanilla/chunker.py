from typing import List
import nltk
from core.chunker import BaseChunker


class NoChunker(BaseChunker):
    """Identity chunker: returns each document as a single chunk.

    This is the **correct chunker for DP-RAG** (Grislain et al.,
    arXiv:2412.19291v2).  The paper's exponential mechanism for document
    retrieval assumes one document = one privacy unit (PU).  Splitting a
    document into multiple chunks causes a single individual to contribute
    several entries to the index; the true sensitivity of the mechanism then
    scales by the maximum number of chunks per document, which inflates the
    effective privacy cost beyond the declared ε.

    Use ``NoChunker`` whenever the DP guarantee must be taken at face value.
    Use ``FixedSizeChunker`` only when documents are already guaranteed to be
    short (one document ≤ one PU) or when the experiment intentionally accepts
    the group-privacy degradation.
    """

    def chunk(self, document: str) -> List[str]:
        return [document]

    def get_config(self) -> dict:
        return {"type": "NoChunker"}

# Ensure nltk punkt is downloaded (will not download if already present)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class FixedSizeChunker(BaseChunker):
    """Document chunker that splits text into fixed-size word chunks.

    Splits documents by word count with optional overlap between chunks.

    Attributes:
        chunk_size: Number of words per chunk.
        overlap: Number of words to overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 1024, overlap: int = 50):
        """Initialize the fixed-size chunker.

        Args:
            chunk_size: Number of words per chunk.
            overlap: Number of overlapping words between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: str) -> List[str]:
        """Chunk a document into fixed-size word chunks.

        Args:
            document: Document text to chunk.

        Returns:
            List of text chunks.
        """
        words = document.split()
        chunks = []
        if not words:
            return chunks

        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
            i += (self.chunk_size - self.overlap)
        
        return chunks

    def get_config(self) -> dict:
        """Get the configuration of this chunker.

        Returns:
            Dictionary with chunker type and parameters.
        """
        return {"type": "FixedSizeChunker", "chunk_size": self.chunk_size, "overlap": self.overlap}

class SentenceChunker(BaseChunker):
    """Document chunker that splits text into sentence-based chunks.

    Splits documents by sentences with optional overlap between chunks.
    Uses NLTK's sentence tokenizer for reliable sentence splitting.

    Attributes:
        max_sentences: Maximum number of sentences per chunk.
        overlap: Number of sentences to overlap between consecutive chunks.
    """

    def __init__(self, max_sentences: int = 15, overlap: int = 1):
        """Initialize the sentence-based chunker.

        Args:
            max_sentences: Maximum number of sentences per chunk.
            overlap: Number of overlapping sentences between chunks.
        """
        self.max_sentences = max_sentences
        self.overlap = overlap

    def chunk(self, document: str) -> List[str]:
        """Chunk a document into sentence-based chunks.

        Args:
            document: Document text to chunk.

        Returns:
            List of text chunks separated by sentences.
        """
        sentences = nltk.tokenize.sent_tokenize(document)
        chunks = []
        if not sentences:
            return chunks

        i = 0
        while i < len(sentences):
            chunk_sentences = sentences[i:i + self.max_sentences]
            chunks.append(" ".join(chunk_sentences))
            i += max(1, self.max_sentences - self.overlap)

        return chunks

    def get_config(self) -> dict:
        """Get the configuration of this chunker.

        Returns:
            Dictionary with chunker type and parameters.
        """
        return {"type": "SentenceChunker", "max_sentences": self.max_sentences, "overlap": self.overlap}
