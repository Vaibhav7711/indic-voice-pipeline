"""Chat prompt construction is shared by the pipeline and the agent turn.

CPU only: exercises the fallback ladder with fake tokenizers.
"""

from __future__ import annotations

from llm.prompting import SYSTEM_PROMPTS, build_chat_prompt, system_prompt_for


class _Qwen3LikeTokenizer:
    """Accepts enable_thinking and records what it was given."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking=True):
        self.calls.append(enable_thinking)
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return body + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")


class _StrictTokenizer:
    """Rejects unknown keywords, like older chat templates."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kw):
        if kw:
            raise TypeError("unexpected keyword")
        return "|".join(m["content"] for m in messages)


class _NoTemplateTokenizer:
    pass


def test_thinking_is_disabled_when_template_supports_it():
    tok = _Qwen3LikeTokenizer()
    prompt = build_chat_prompt(tok, "sys", "नमस्ते")
    assert tok.calls == [False]
    assert "</think>" in prompt
    assert "<user>नमस्ते" in prompt


def test_falls_back_when_template_rejects_enable_thinking():
    prompt = build_chat_prompt(_StrictTokenizer(), "sys", "user text")
    assert prompt == "sys|user text"


def test_plain_text_prompt_without_template():
    prompt = build_chat_prompt(_NoTemplateTokenizer(), "sys", "user text")
    assert prompt == "System: sys\n\nUser: user text\n\nAssistant:"
    assert build_chat_prompt(None, "sys", "u").startswith("System: sys")


def test_system_prompt_lookup_defaults():
    assert system_prompt_for("hi") is SYSTEM_PROMPTS["hi"]
    assert system_prompt_for("xx") is SYSTEM_PROMPTS[None]
    assert "Hindi" in SYSTEM_PROMPTS["hi"]
    assert "Telugu" in SYSTEM_PROMPTS["te"]


def test_pipeline_and_turn_use_the_same_builder():
    """Regression: the thinking-mode fix must apply on both paths."""
    from agent import turn
    from pipeline import orchestrator

    assert orchestrator.build_chat_prompt is build_chat_prompt
    assert turn.build_chat_prompt is build_chat_prompt
