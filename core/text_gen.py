"""Shared generation-time text helpers.

Two concerns, both used by every generator that talks to a HuggingFace model:

* :func:`to_model_prompt` — **C2**: route a raw prompt through the model's chat
  template when one exists, so instruct models answer instead of echoing the
  ``Instruction:/Context:/Question:/Answer:`` scaffold. Base models without a chat
  template (gpt2, opt, pythia) are returned unchanged, so existing runs are
  unaffected. Auto-detected by default.

* :func:`trim_with_stop_strings` — **C3**: optionally truncate generated text at
  the first stop string. OFF by default (an opt-in extra feature); only active
  when a non-empty ``stop_strings`` list is configured.
"""

from __future__ import annotations

from typing import Any


def chat_template_active(tokenizer: Any, use_chat_template: bool | str = "auto") -> bool:
    """Whether chat templating will be applied for this tokenizer/setting.

    ``"auto"`` (default) enables it iff the tokenizer defines a ``chat_template``;
    an explicit ``True``/``False`` forces the choice (but ``True`` still no-ops when
    the tokenizer has no template, since there is nothing to apply).
    """
    has_template = bool(getattr(tokenizer, "chat_template", None))
    if use_chat_template == "auto":
        return has_template
    return bool(use_chat_template) and has_template


def to_model_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    use_chat_template: bool | str = "auto",
    answer_cue: str = "Answer:",
    system: str | None = None,
) -> str:
    """Return a model-ready prompt string (C2).

    When :func:`chat_template_active` is true, wrap ``prompt`` as a single user turn
    and append the assistant generation prompt; the trailing ``answer_cue`` (e.g.
    ``"Answer:"``) is stripped because ``add_generation_prompt`` already opens the
    assistant turn. Otherwise ``prompt`` is returned verbatim (raw completion).

    Args:
        tokenizer: a HF tokenizer (may or may not define ``chat_template``).
        prompt: the raw prompt built by the method's prompt builder.
        use_chat_template: ``True``/``False`` to force, or ``"auto"`` (default).
        answer_cue: trailing cue stripped before templating (``""`` keeps it).
        system: optional system-message content.

    Returns:
        The chat-templated string (assistant turn open) or ``prompt`` unchanged.
    """
    if not chat_template_active(tokenizer, use_chat_template):
        return prompt

    user = prompt
    if answer_cue:
        stripped = user.rstrip()
        if stripped.endswith(answer_cue):
            user = stripped[: -len(answer_cue)].rstrip()

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def trim_with_stop_strings(text: str, stop_strings: list[str] | None) -> str:
    """Truncate ``text`` at the first occurrence of any stop string (C3, opt-in).

    Returns ``text`` unchanged unless ``stop_strings`` is a non-empty list — the
    feature is OFF by default. When enabled it cuts runaway ``"\\nQuestion:"`` /
    scaffold loops from a generated answer before scoring.
    """
    if not stop_strings:
        return text
    cut = len(text)
    for s in stop_strings:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1 and idx < cut:
            cut = idx
    return text[:cut].strip()
