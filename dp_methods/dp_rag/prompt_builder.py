from __future__ import annotations

import json
from typing import List, Dict

from core.prompt_builder import BasePromptBuilder


class PerDocPromptBuilder(BasePromptBuilder):
    """
    In DP-RAG, each retrieved document builds an independent prompt.
    A final public prompt (non-private context) is always appended as the
    last entry so the generator can use it as L_pub in the DP-ICL utility.
    The list of k+1 prompts is serialized as JSON.
    """

    def __init__(self, template: str = None, public_context: str = ""):
        self.template = template or "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        self.public_context = public_context

    def build(self, query: str, chunks: List[Dict]) -> str:
        prompts: list[str] = []
        for chunk in chunks:
            context = str(chunk.get("text", ""))
            prompts.append(self.template.format(context=context, query=query))
        # Public document prompt (non-private); generator treats last prompt as L_pub.
        prompts.append(self.template.format(context=self.public_context, query=query))
        return json.dumps(prompts)