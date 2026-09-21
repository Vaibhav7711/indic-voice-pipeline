"""Streaming ASR: online endpointing plus a stateful session API.

``asr.vad`` answers "where was the speech in this file". This package answers
"has the user stopped talking yet", which is a different question and needs a
different algorithm — see :mod:`asr.streaming.endpointer`.

Nothing here imports torch. The session talks to an ASR backend through a
Protocol that ``asr.explicit.ASRRunner`` satisfies structurally, so the state
machine is unit-testable on CPU against a fake transcriber.
"""

from asr.streaming.endpointer import (
    EndpointerState,
    EndpointEvent,
    EndpointEventKind,
    StreamEndpointer,
)
from asr.streaming.policy import EndpointPolicy, phrase_is_incomplete
from asr.streaming.session import (
    EndpointReason,
    SessionState,
    StreamingConfig,
    StreamingSession,
    StreamingTranscriber,
    StreamUpdate,
    UpdateKind,
)

__all__ = [
    "EndpointPolicy",
    "phrase_is_incomplete",
    "StreamEndpointer",
    "EndpointerState",
    "EndpointEvent",
    "EndpointEventKind",
    "StreamingSession",
    "StreamingConfig",
    "StreamingTranscriber",
    "StreamUpdate",
    "UpdateKind",
    "SessionState",
    "EndpointReason",
]
