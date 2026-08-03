from typing import List, Dict
from core.prompt_builder import BasePromptBuilder

class SimplePromptBuilder(BasePromptBuilder):
    """Simple prompt builder that concatenates retrieved chunks as context.

    Combines all retrieved document chunks into a single context string
    and formats them with a configurable template for the language model.

    Attributes:
        template: Formatting template with {context} and {query} placeholders.
    """

    def __init__(
        self,
        template: str = "Context: {context}\n\nQuestion: {query}\nAnswer:",
        *,
        token_budget: int | None = None,
        count_tokens=None,
    ):
        """Initialize the prompt builder with a template.

        Args:
            template: Template string with {context} and {query} placeholders.
            token_budget: Largest prompt this model can read. When set, chunks are
                admitted best-first until the budget is reached and the rest are
                dropped, so the context that survives is the context that was ranked
                highest. Leaving this unset lets an over-long prompt reach the
                tokenizer, whose left truncation removes the FRONT of the context —
                and since the retriever returns chunks in descending similarity, the
                front is the best-ranked documents. On a 2048-position model a
                50-document prompt then arrives as the ~8 *worst* of the 50.
            count_tokens: Callable mapping text to a token count, normally the
                model's tokenizer. Required for token_budget to take effect.
        """
        self.template = template
        self.token_budget = token_budget
        self.count_tokens = count_tokens
        #: Chunks admitted by the last build, so a run can report the context the
        #: model actually read rather than the number that was retrieved.
        self.last_kept = 0

    def build(self, query: str, chunks: List[Dict]) -> str:
        """Build a prompt by concatenating chunks as context.

        Args:
            query: User's query.
            chunks: List of retrieved document chunks, best-ranked first.

        Returns:
            Formatted prompt string ready for the language model.
        """
        if self.token_budget is None or self.count_tokens is None:
            self.last_kept = len(chunks)
            context_str = "\n".join([f"- {chunk['text']}" for chunk in chunks])
            return self.template.format(context=context_str, query=query)

        kept: List[str] = []
        prompt = self.template.format(context="", query=query)
        for chunk in chunks:
            candidate = kept + [f"- {chunk['text']}"]
            trial = self.template.format(context="\n".join(candidate), query=query)
            if self.count_tokens(trial) > self.token_budget:
                break
            kept, prompt = candidate, trial
        self.last_kept = len(kept)
        return prompt
