"""Edge-TTS wrapper for Indic voice output. Zero GPU cost."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

DEFAULT_VOICES = {
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
    "ta": "ta-IN-PallaviNeural",
    "en": "en-IN-NeerjaNeural",
    None: "en-IN-NeerjaNeural",
}


@dataclass
class TTSResult:
    audio_bytes: bytes
    output_path: str | None
    synthesis_ms: float
    voice: str


class TTSSynthesizer:
    def __init__(self, language: str | None = None, voice: str | None = None):
        self.language = language
        self.voice = voice or DEFAULT_VOICES.get(language, DEFAULT_VOICES[None])

    def synthesize(self, text: str, output_path: str | None = None) -> TTSResult:
        """Synchronous synthesis — safe from any context."""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run, self._synth(text, output_path),
            ).result()

    async def _synth(self, text: str, output_path: str | None) -> TTSResult:
        import edge_tts

        start = perf_counter_ns()
        comm = edge_tts.Communicate(text, self.voice)
        chunks = []
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])

        audio = b"".join(chunks)
        ms = (perf_counter_ns() - start) / 1_000_000

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(audio)

        return TTSResult(audio, output_path, ms, self.voice)
