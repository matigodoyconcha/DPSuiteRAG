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

    PRIVATE_TEMPLATE = "Document:\n\"{context}\"\n\nQuestion: {query}\nAnswer:"
    PUBLIC_TEMPLATE = "Give a SHORT answer based on your knowledge.\n\nQuestion: {query}\nAnswer:"

    def __init__(self, private_template: str = None, public_template: str = None):
        self.private_template = private_template or self.PRIVATE_TEMPLATE
        self.public_template = public_template or self.PUBLIC_TEMPLATE

    def build(self, query: str, chunks: List[Dict]) -> str:
        prompts: list[str] = []
        for chunk in chunks:
            context = str(chunk.get("text", ""))
            prompts.append(self.private_template.format(context=context, query=query))
        # Public prompt (no private document); generator uses it as L_pub prior.
        prompts.append(self.public_template.format(query=query))
        return json.dumps(prompts)