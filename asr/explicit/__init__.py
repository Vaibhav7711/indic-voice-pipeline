"""Explicit Whisper ASR runtime.

Owns the full loop: mel → encoder → autoregressive decoder with dual KV cache.
``model.generate()`` and ``pipeline("asr")`` are never in the execution path.
"""

from asr.explicit.loader import LoadedWhisper, load_whisper
from asr.explicit.runner import ASRMetrics, ASRResult, ASRRunner

__all__ = ["LoadedWhisper", "load_whisper", "ASRRunner", "ASRResult", "ASRMetrics"]
