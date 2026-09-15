"""Speech synthesis.

``TTSSynthesizer`` (buffered, returns a complete clip) is unchanged.
``tts.streaming`` adds an incremental path — see ``docs/STREAMING.md`` for what
is genuinely streaming here and what is not.
"""

from tts.streaming import (
    AudioChunk,
    EdgeStreamingSynthesizer,
    SpeechStream,
    StreamingSynthesizer,
    split_sentences,
    synthesize_stream,
)
from tts.synthesis import TTSResult, TTSSynthesizer

__all__ = [
    "TTSSynthesizer", "TTSResult",
    "EdgeStreamingSynthesizer", "StreamingSynthesizer",
    "synthesize_stream", "split_sentences", "SpeechStream", "AudioChunk",
]
