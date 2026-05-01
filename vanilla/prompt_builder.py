from typing import List, Dict
from core.prompt_builder import BasePromptBuilder

class SimplePromptBuilder(BasePromptBuilder):
    """Simple prompt builder that concatenates retrieved chunks as context.

    Combines all retrieved document chunks into a single context string
    and formats them with a configurable template for the language model.

    Attributes:
        template: Formatting template with {context} and {query} placeholders.
    """

    def __init__(self, template: str = "Context: {context}\n\nQuestion: {query}\nAnswer:"):
        """Initialize the prompt builder with a template.

        Args:
            template: Template string with {context} and {query} placeholders.
        """
        self.template = template

    def build(self, query: str, chunks: List[Dict]) -> str:
        """Build a prompt by concatenating chunks as context.

        Args:
            query: User's query.
            chunks: List of retrieved document chunks.

        Returns:
            Formatted prompt string ready for the language model.
        """
        context_str = "\n".join([f"- {chunk['text']}" for chunk in chunks])
        return self.template.format(context=context_str, query=query)
