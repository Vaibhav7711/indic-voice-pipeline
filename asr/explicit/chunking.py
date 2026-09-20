"""Deterministic long-form audio chunking and transcript stitching.

Whisper's encoder accepts at most 30 seconds (3000 mel frames).  This module
keeps that model constraint explicit: it produces overlapping, bounded audio
windows and merges only a verified suffix/prefix text overlap at each boundary.
It deliberately does *not* pretend to be VAD; endpointing is a later stage.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    """One bounded window within a normalized 16 kHz waveform."""

    index: int
    start_sample: int
    end_sample: int
    sample_rate: int

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sample_rate


def chunk_audio(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    chunk_seconds: float = 25.0,
    overlap_seconds: float = 5.0,
) -> list[tuple[AudioChunk, np.ndarray]]:
    """Split audio into overlapping chunks shorter than Whisper's 30 s limit."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0 < chunk_seconds <= 30:
        raise ValueError("chunk_seconds must be in (0, 30]")
    if not 0 <= overlap_seconds < chunk_seconds:
        raise ValueError("overlap_seconds must be >= 0 and smaller than chunk_seconds")

    chunk_samples = round(chunk_seconds * sample_rate)
    stride_samples = round((chunk_seconds - overlap_seconds) * sample_rate)
    if len(waveform) == 0:
        return []

    chunks: list[tuple[AudioChunk, np.ndarray]] = []
    for start in range(0, len(waveform), stride_samples):
        end = min(start + chunk_samples, len(waveform))
        spec = AudioChunk(len(chunks), start, end, sample_rate)
        chunks.append((spec, waveform[start:end]))
        if end == len(waveform):
            break
    return chunks


# Do not use ``\W`` here: Unicode combining marks (including Devanagari
# matras) are not word characters and would corrupt the comparison key.
_EDGE_PUNCTUATION = re.compile(r"^[.,!?;:'\"()\[\]{}<>।॥…_-]+|[.,!?;:'\"()\[\]{}<>।॥…_-]+$")


def _match_key(token: str) -> str:
    """Conservative comparison key; emitted text is never normalized here."""
    return _EDGE_PUNCTUATION.sub("", unicodedata.normalize("NFKC", token).casefold())


def merge_overlapping_transcripts(left: str, right: str, *, max_overlap_tokens: int = 40) -> str:
    """Append ``right`` after removing its longest verified token overlap.

    Matching is whitespace-token based and only ignores edge punctuation and
    Unicode presentation differences.  This avoids silently rewriting Hindi
    text while still handling common repeated words at chunk boundaries.
    """
    left_tokens, right_tokens = left.split(), right.split()
    if not left_tokens:
        return " ".join(right_tokens)
    if not right_tokens:
        return " ".join(left_tokens)

    upper = min(len(left_tokens), len(right_tokens), max_overlap_tokens)
    overlap = 0
    for width in range(upper, 0, -1):
        left_keys = [_match_key(token) for token in left_tokens[-width:]]
        right_keys = [_match_key(token) for token in right_tokens[:width]]
        if all(a and a == b for a, b in zip(left_keys, right_keys, strict=True)):
            overlap = width
            break
    return " ".join(left_tokens + right_tokens[overlap:])
