from core.postprocessor import BasePostprocessor

class IdentityPostprocessor(BasePostprocessor):
    """No-op postprocessor that returns response unchanged.

    This is a baseline postprocessor that does not modify the generated text.
    Useful for vanilla RAG pipelines without additional post-processing.
    """

    def process(self, response: str) -> str:
        """Return response unchanged.

        Args:
            response: Generated response text.

        Returns:
            The same response text unchanged.
        """
        return response
