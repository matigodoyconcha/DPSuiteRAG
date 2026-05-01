from abc import ABC, abstractmethod

class BaseChunker(ABC):
    """Base class for document chunking strategies.

    Chunkers are responsible for splitting documents into smaller pieces
    that can be embedded and retrieved. Different chunking strategies
    (fixed size, semantic, etc.) can be implemented by subclasses.
    """

    @abstractmethod
    def chunk(self, document: str) -> list[str]:
        """Split a document into chunks.

        Args:
            document: The full document text to be chunked.

        Returns:
            List of document chunks as strings.
        """
        pass

    @abstractmethod
    def get_config(self) -> dict:
        """Get the configuration parameters of this chunker.

        Returns:
            Dictionary containing chunker configuration details.
        """
        pass
