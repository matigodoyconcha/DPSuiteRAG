from typing import List
import nltk
from core.chunker import BaseChunker

# Ensure nltk punkt is downloaded (will not download if already present)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1024, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: str) -> List[str]:
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
        return {"type": "FixedSizeChunker", "chunk_size": self.chunk_size, "overlap": self.overlap}

class SentenceChunker(BaseChunker):
    def __init__(self, max_sentences: int = 15, overlap: int = 1):
        self.max_sentences = max_sentences
        self.overlap = overlap

    def chunk(self, document: str) -> List[str]:
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
        return {"type": "SentenceChunker", "max_sentences": self.max_sentences, "overlap": self.overlap}
