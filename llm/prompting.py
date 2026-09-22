"""One chat-prompt builder for every path that talks to the LLM.

The agent turn and the batch pipeline used to build prompts separately, and
only one of them disabled Qwen3's thinking mode. With thinking left on, the
model spends its whole token budget inside ``<think>`` and the spoken answer
is empty. Having a single builder means a template fix lands everywhere.
"""

from __future__ import annotations

__all__ = ["SYSTEM_PROMPTS", "system_prompt_for", "build_chat_prompt",
           "render_messages"]


SYSTEM_PROMPTS: dict[str | None, str] = {
    "hi": (
        "You are a helpful voice assistant. The user spoke in Hindi "
        "(possibly code-switched with English). Respond entirely in natural "
        "Hindi; keep unavoidable proper nouns and technical terms as-is. "
        "Answer in one or two short sentences — your reply is spoken aloud, "
        "and a long first sentence keeps the user waiting in silence. Never "
        "repeat the question back."
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
    """Render a single system+user exchange. See :func:`render_messages`."""
    return render_messages(tokenizer, [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])


def render_messages(tokenizer, messages: list[dict]) -> str:
    """Render a message list with the tokenizer's chat template.

    ``enable_thinking=False`` is passed first because Qwen3 templates accept
    it and default to thinking otherwise. Templates that reject the keyword
    fall back to a plain call; tokenizers without a template get a text
    prompt. The fallback order is what the tests pin down.

    Multi-turn history (see :mod:`agent.conversation`) arrives here as
    alternating user/assistant messages, so the template puts them in the
    model's own turn format rather than a hand-rolled transcript.
    """
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
    # No usable template: a labelled transcript, which is what a base model
    # without a chat format expects anyway.
    lines = []
    for message in messages:
        role = {"system": "System", "user": "User", "assistant": "Assistant"}.get(
            message["role"], message["role"].title(),
        )
        lines.append(f"{role}: {message['content']}")
    return "\n\n".join(lines) + "\n\nAssistant:"
