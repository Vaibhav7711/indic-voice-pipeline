"""One chat-prompt builder for every path that talks to the LLM.

The agent turn and the batch pipeline used to build prompts separately, and
only one of them disabled Qwen3's thinking mode. With thinking left on, the
model spends its whole token budget inside ``<think>`` and the spoken answer
is empty. Having a single builder means a template fix lands everywhere.
"""

from __future__ import annotations

__all__ = ["SYSTEM_PROMPTS", "GROUNDED_SUFFIX", "SYSTEM_VARIANTS",
           "system_prompt_for", "build_chat_prompt",
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


#: Added by the ``grounded`` variant. The measured run is the reason: asked the
#: time with no clock the assistant answered "3:45 बजे", and asked the weather
#: with no weather data, "खुशी से बराबर है" -- 0 of 3 unanswerable cases
#: declined. Inventing a plausible answer is worse than refusing, because it
#: sounds like an answer and a listener has no way to tell.
#:
#: Deliberately general rather than a list of the cases it was written for.
#: Naming "the time, the weather, your accounts" as examples rather than as the
#: rule is the difference between an instruction and an overfit.
GROUNDED_SUFFIX = {
    "hi": (
        " अगर आपको उत्तर नहीं पता, या उत्तर के लिए ऐसी जानकारी चाहिए जो आपके "
        "पास नहीं है — जैसे अभी का समय, आज का मौसम, या उपयोगकर्ता की निजी "
        "जानकारी — तो संक्षेप में कहिए कि आपको नहीं पता। कोई तथ्य मत गढ़िए।"
    ),
    "te": (
        " సమాధానం తెలియకపోతే, లేదా సమాధానానికి మీ వద్ద లేని సమాచారం "
        "అవసరమైతే, తెలియదని క్లుప్తంగా చెప్పండి. వాస్తవాలను కల్పించవద్దు."
    ),
    None: (
        " If you do not know the answer, or it needs information you do not "
        "have -- the current time, today's weather, the user's private data -- "
        "say briefly that you do not know. Never invent a fact."
    ),
}

#: Named system-prompt variants, so an A/B records which one it ran rather than
#: leaving it to be inferred from a commit date.
SYSTEM_VARIANTS = ("default", "grounded")


def system_prompt_for(language: str | None, variant: str = "default") -> str:
    """The system prompt for a language, in a named variant.

    ``default`` is what the pipeline has always served. ``grounded`` appends an
    instruction to admit ignorance instead of inventing, which is the one
    untested lever left after sampling was measured and refuted:
    instruction-following was 80% while 0 of 3 unanswerable cases were
    declined, so the model obeys the prompt and the prompt never asked.
    """
    if variant not in SYSTEM_VARIANTS:
        raise ValueError(f"system prompt variant must be one of "
                         f"{SYSTEM_VARIANTS}, got {variant!r}")
    base = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS[None])
    if variant == "default":
        return base
    suffix = GROUNDED_SUFFIX.get(language, GROUNDED_SUFFIX[None])
    return base + suffix


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
