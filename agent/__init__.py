"""Voice-agent output behaviour: playback lifecycle and turn orchestration.

``asr.streaming`` handles the input half of a turn (audio in, transcript out).
This package handles the output half: response generation, speech synthesis,
playback, and interruption — plus the latency accounting that says which of
those to fix.

Nothing here imports torch or contacts a network. Backends are injected through
Protocols, so a full turn is unit-testable against fakes.
"""

from agent.conversation import Conversation, Exchange
from agent.playback import (
    AudioSink,
    BufferSink,
    PlaybackEvent,
    PlaybackEventKind,
    PlaybackResult,
    PlaybackSession,
    PlaybackState,
)
from agent.turn import (
    ResponseGenerator,
    TurnMetrics,
    TurnResult,
    TurnState,
    VoiceTurn,
)

__all__ = [
    "Conversation",
    "Exchange",
    "PlaybackSession", "PlaybackState", "PlaybackEvent", "PlaybackEventKind",
    "PlaybackResult", "AudioSink", "BufferSink",
    "VoiceTurn", "TurnState", "TurnMetrics", "TurnResult", "ResponseGenerator",
]
