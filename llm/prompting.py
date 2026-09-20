"""One chat-prompt builder for every path that talks to the LLM.

The agent turn and the batch pipeline used to build prompts separately, and
only one of them disabled Qwen3's thinking mode. With thinking left on, the
model spends its whole token budget inside ``<think>`` and the spoken answer
is empty. Having a single builder means a template fix lands everywhere.
"""

from __future__ import annotations

__all__ = ["SYSTEM_PROMPTS", "system_prompt_for", "build_chat_prompt"]


SYSTEM_PROMPTS: dict[str | None, str] = {
    "hi": (
        "You are a helpful voice assistant. The user spoke in Hindi "
        "(possibly code-switched with English). Respond entirely in natural "
        "Hindi; keep unavoidable proper nouns and technical terms as-is. "
        "Keep answers brief — this will be spoken aloud."
    ),
    "te": (
        "You are a helpful voice assistant. The user spoke in Telugu "
        "(possibly code-switched with English). Respond entirely in Telugu "
        "and keep the answer brief — this will be spoken aloud."
    ),
    "en": (
        "You are a helpful voice assistant. Keep the answer brief — this "
        "will be spoken aloud."
    ),
    None: (
        "You are a helpful voice assistant. Keep the answer brief — this "
        "will be spoken aloud."
    ),
}


def system_prompt_for(language: str | None) -> str:
    return SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS[None])


def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    """Render a system+user exchange with the tokenizer's chat template.

    ``enable_thinking=False`` is passed first because Qwen3 templates accept
    it and default to thinking otherwise. Templates that reject the keyword
    fall back to a plain call; tokenizers without a template get a text
    prompt. The fallback order is what the tests pin down.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply):
        try:
            return apply(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except (TypeError, ValueError):
            try:
                return apply(messages, tokenize=False, add_generation_prompt=True)
            except (TypeError, ValueError):
                pass
    return f"System: {system}\n\nUser: {user}\n\nAssistant:"
