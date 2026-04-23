from __future__ import annotations

import json
from typing import List, Dict

from core.prompt_builder import BasePromptBuilder


class PerDocPromptBuilder(BasePromptBuilder):
    """
    In DP-RAG, each retrieved document builds an independent prompt.
    The list of prompts is serialized as JSON to keep compatibility with
    BasePromptBuilder's string return type.
    """

    def __init__(self, template: str = None):
        self.template = template or "Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    def build(self, query: str, chunks: List[Dict]) -> str:
        prompts: list[str] = []
        for chunk in chunks:
            context = str(chunk.get("text", ""))
            prompts.append(self.template.format(context=context, query=query))
        return json.dumps(prompts)