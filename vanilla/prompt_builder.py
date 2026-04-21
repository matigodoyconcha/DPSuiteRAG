from typing import List, Dict
from core.prompt_builder import BasePromptBuilder

class SimplePromptBuilder(BasePromptBuilder):
    def __init__(self, template: str = "Context: {context}\n\nQuestion: {query}\Awnser:"):
        self.template = template

    def build(self, query: str, chunks: List[Dict]) -> str:
        context_str = "\n".join([f"- {chunk['text']}" for chunk in chunks])
        return self.template.format(context=context_str, query=query)
