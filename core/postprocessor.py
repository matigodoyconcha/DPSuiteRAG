from abc import ABC, abstractmethod

class BasePostprocessor(ABC):
    """Base class for post-processing generated responses.

    Postprocessors are responsible for cleaning and formatting the raw output
    from the language model generator. This may include removing artifacts,
    extracting specific information, or reformatting for the user.
    """

    @abstractmethod
    def process(self, response: str) -> str:
        """Post-process a generated response.

        Args:
            response: The raw response from the language model.

        Returns:
            Cleaned and formatted response.
        """
        pass
