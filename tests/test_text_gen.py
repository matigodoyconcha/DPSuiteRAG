"""Unit tests for core.text_gen (C2 chat templating + C3 stop-string trimming).

Pure string-logic tests — no model or real tokenizer is loaded.
"""
from __future__ import annotations

from core.text_gen import (
    chat_template_active,
    to_model_prompt,
    trim_with_stop_strings,
)


class _FakeTokenizer:
    """Minimal stand-in exposing the bits the helpers touch."""

    def __init__(self, chat_template=None):
        self.chat_template = chat_template
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.last_messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        body = "\n".join(f"<{m['role']}>{m['content']}" for m in messages)
        return body + "\n<assistant>"


# ---------------------------------------------------------------------------
# chat_template_active
# ---------------------------------------------------------------------------

def test_active_auto_depends_on_template():
    assert chat_template_active(_FakeTokenizer(chat_template="x"), "auto") is True
    assert chat_template_active(_FakeTokenizer(chat_template=None), "auto") is False


def test_active_explicit_true_still_needs_template():
    # Forcing True cannot apply a template that does not exist.
    assert chat_template_active(_FakeTokenizer(chat_template=None), True) is False
    assert chat_template_active(_FakeTokenizer(chat_template="x"), True) is True


def test_active_explicit_false_disables():
    assert chat_template_active(_FakeTokenizer(chat_template="x"), False) is False


# ---------------------------------------------------------------------------
# to_model_prompt
# ---------------------------------------------------------------------------

def test_base_model_prompt_unchanged():
    tok = _FakeTokenizer(chat_template=None)
    raw = "Instruction: do it\nContext: c\nQuestion: q\nAnswer:"
    assert to_model_prompt(tok, raw) == raw  # raw completion preserved
    assert tok.last_messages is None


def test_chat_template_strips_answer_cue_and_wraps():
    tok = _FakeTokenizer(chat_template="present")
    raw = "Instruction: do it\nContext: c\nQuestion: q\nAnswer:"
    out = to_model_prompt(tok, raw)
    # The trailing "Answer:" cue is removed (add_generation_prompt opens the turn).
    assert tok.last_messages == [
        {"role": "user", "content": "Instruction: do it\nContext: c\nQuestion: q"}
    ]
    assert out.endswith("<assistant>")
    assert "Answer:" not in tok.last_messages[0]["content"]


def test_chat_template_with_system_message():
    tok = _FakeTokenizer(chat_template="present")
    out = to_model_prompt(tok, "Question: q\nAnswer:", system="be terse")
    roles = [m["role"] for m in tok.last_messages]
    assert roles == ["system", "user"]
    assert out.endswith("<assistant>")


def test_force_false_keeps_raw_even_with_template():
    tok = _FakeTokenizer(chat_template="present")
    raw = "Question: q\nAnswer:"
    assert to_model_prompt(tok, raw, use_chat_template=False) == raw


# ---------------------------------------------------------------------------
# trim_with_stop_strings  (C3 — OFF by default)
# ---------------------------------------------------------------------------

def test_trim_off_by_default():
    text = "Paris\nQuestion: next?"
    assert trim_with_stop_strings(text, None) == text
    assert trim_with_stop_strings(text, []) == text


def test_trim_cuts_at_first_stop():
    text = "Paris\nQuestion: next?\nContext: x"
    assert trim_with_stop_strings(text, ["\nQuestion:", "\nContext:"]) == "Paris"


def test_trim_no_match_returns_text():
    text = "Paris"
    assert trim_with_stop_strings(text, ["\nQuestion:"]) == "Paris"


def test_trim_picks_earliest_stop():
    text = "a STOP1 b STOP2 c"
    # STOP1 occurs first -> cut there regardless of list order.
    assert trim_with_stop_strings(text, ["STOP2", "STOP1"]) == "a"
