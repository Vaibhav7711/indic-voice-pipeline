"""Semantic endpointing: let the transcript-so-far adjust the silence wait.

The energy endpointer decides "the user stopped" from silence alone. On read
Hindi that split 16% of clips at a comma-length pause
(``results/streaming_eval/``), and for an agent a split means answering
mid-sentence. The transcript knows something the energy does not: a phrase
ending in a postposition (``… के``), a conjunction (``… और``) or a
subordinator (``… कि``) is not finished, however long the pause.

This module is a rule list, not a model, on purpose: every decision is
inspectable, it costs nothing per frame, and it is the baseline a learned
endpointer has to beat. It only ever *lengthens* the wait for an incomplete
phrase; it never shortens it below the configured silence, so a wrong rule
costs latency, not a cut-off user.

LLM analogy: this is a stop criterion that reads the decoded text, the way
a stop-sequence check reads generated tokens, rather than a fixed length.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["EndpointPolicy", "HINDI_INCOMPLETE_ENDINGS", "phrase_is_incomplete"]

#: Tokens that almost never end a Hindi utterance. Postpositions, conjunctions,
#: subordinators, auxiliaries that need a main verb, and the English function
#: words that appear in code-switched speech.
HINDI_INCOMPLETE_ENDINGS: frozenset[str] = frozenset({
    # postpositions
    "का", "के", "की", "को", "से", "में", "पर", "ने", "तक", "लिए", "वाले", "वाला", "वाली",
    "साथ", "बाद", "पहले", "बारे", "अंदर", "बाहर", "ऊपर", "नीचे", "पास", "बिना", "द्वारा",
    # conjunctions / subordinators
    "और", "या", "कि", "लेकिन", "मगर", "परंतु", "तो", "अगर", "यदि", "जब", "क्योंकि", "जो",
    "जिस", "जिन", "ताकि", "फिर", "मतलब", "यानी", "बल्कि", "एवं", "तथा", "व",
    # determiners / quantifiers that expect a noun
    "एक", "यह", "वह", "ये", "वे", "इस", "उस", "इन", "उन", "कुछ", "हर", "सब", "बहुत",
    "थोड़ा", "थोड़ी", "थोड़े", "कई", "अपना", "अपने", "अपनी", "मेरा", "मेरे", "मेरी",
    # English function words in code-switched speech
    "and", "or", "but", "the", "a", "an", "of", "to", "in", "on", "for", "with",
    "because", "so", "if", "that", "which", "my", "your",
})

_TRAILING_PUNCT = re.compile(r"[।॥.!?…,;:\"'()\[\]{}\-–—]+$")


def phrase_is_incomplete(text: str, endings: frozenset[str] = HINDI_INCOMPLETE_ENDINGS) -> bool:
    """True when the last token says the speaker has not finished the phrase.

    A sentence terminator (danda, ``?``, ``!``) wins outright — the ASR model
    emitted one, which is strong evidence of completion. Otherwise the last
    token, stripped of punctuation and lower-cased, is looked up.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[-1] in "।॥?!":
        return False
    last = _TRAILING_PUNCT.sub("", stripped.split()[-1]).lower()
    return last in endings


@dataclass(frozen=True)
class EndpointPolicy:
    """Silence required to close an utterance, given its transcript so far.

    ``incomplete_silence_ms`` applies when :func:`phrase_is_incomplete` says
    the phrase is unfinished; otherwise ``None`` means "use the endpointer's
    configured default". The policy never returns less than the default.
    """

    incomplete_silence_ms: int = 1000
    endings: frozenset[str] = field(default=HINDI_INCOMPLETE_ENDINGS)

    def required_silence_ms(self, text: str) -> int | None:
        if phrase_is_incomplete(text, self.endings):
            return self.incomplete_silence_ms
        return None

    def as_dict(self) -> dict:
        return {"incomplete_silence_ms": self.incomplete_silence_ms,
                "endings": len(self.endings)}
