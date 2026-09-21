"""Local TTS backends: no network round trip on the critical path.

`EdgeStreamingSynthesizer` is a network service; the first audio chunk waits
on a round trip to Microsoft (200–600 ms measured, and it varies with
connectivity). A local model removes that term entirely. The trade is
quality and hardware: the backends here are chosen for latency on the
project's target (one T4, shared with Whisper and the LLM), not for the best
voice available.

All backends satisfy :class:`tts.streaming.StreamingSynthesizer` — ``stream``
yields audio bytes, ``streaming`` says whether the first chunk arrives before
synthesis finishes, and ``format`` says what the bytes are — so the agent
turn and the device sink do not change when the backend does.

MMS-TTS (Meta, VITS, ~36M params per language)
  ``facebook/mms-tts-hin`` et al. Takes Devanagari directly (no romanizer),
  emits 16 kHz mono. Synthesises a whole sentence at once, so
  ``streaming=False``; the sentence-level pipelining in the turn still applies.
  Runs on CPU at RTF ≈ 0.4 on a laptop and in tens of milliseconds on a GPU.
  Voice quality is serviceable, not premium.
"""

from __future__ import annotations

from collections.abc import Iterator
from time import perf_counter_ns

import numpy as np

from agent.audio import AudioFormat

__all__ = ["MmsTtsSynthesizer", "MMS_LANGUAGES"]

#: Whisper-style language code → MMS ISO-639-3 checkpoint suffix.
MMS_LANGUAGES = {
    "hi": "hin", "te": "tel", "ta": "tam", "bn": "ben", "mr": "mar",
    "gu": "guj", "kn": "kan", "ml": "mal", "pa": "pan", "en": "eng",
}


class MmsTtsSynthesizer:
    """``facebook/mms-tts-<lang>`` through transformers' ``VitsModel``.

    The model is loaded lazily on first use so constructing the synthesizer
    (as the agent does at startup) is free; ``warm_up()`` loads it and runs
    one synthesis so the first real turn does not pay for it.
    """

    streaming = False

    def __init__(
        self,
        language: str = "hi",
        *,
        model_name: str | None = None,
        device: str | None = None,
        chunk_ms: int = 200,
        speaking_rate: float | None = None,
    ):
        if model_name is None:
            if language not in MMS_LANGUAGES:
                raise ValueError(f"no MMS-TTS checkpoint known for {language!r}; "
                                 f"pass model_name= or one of {sorted(MMS_LANGUAGES)}")
            model_name = f"facebook/mms-tts-{MMS_LANGUAGES[language]}"
        self.language = language
        self.model_name = model_name
        self.device_name = device
        self.chunk_ms = chunk_ms
        self.speaking_rate = speaking_rate
        self._model = None
        self._tokenizer = None
        self.format = AudioFormat("pcm_s16le", 16_000, 1)   # corrected on load
        self.last_synthesis_ms: float | None = None

    # -- loading ----------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, VitsModel

        device = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if getattr(self._tokenizer, "is_uroman", False):
            raise RuntimeError(
                f"{self.model_name} expects romanized input (uroman); this backend "
                "feeds native script. Pick a checkpoint whose tokenizer is not uroman."
            )
        model = VitsModel.from_pretrained(self.model_name).to(device).eval()
        if self.speaking_rate is not None:
            model.speaking_rate = self.speaking_rate
        self._model = model
        self._device = device
        self.format = AudioFormat("pcm_s16le", int(model.config.sampling_rate), 1)

    def warm_up(self) -> float:
        """Load and run once; returns the warm-up synthesis time in ms."""
        self._load()
        self.synthesize("नमस्ते" if self.language == "hi" else "hello")
        return self.last_synthesis_ms or 0.0

    # -- synthesis --------------------------------------------------------

    def synthesize(self, text: str) -> np.ndarray:
        """Whole-sentence synthesis → float32 waveform in [-1, 1]."""
        self._load()
        import torch

        start = perf_counter_ns()
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            waveform = self._model(**inputs).waveform[0]
        audio = waveform.float().cpu().numpy()
        self.last_synthesis_ms = (perf_counter_ns() - start) / 1_000_000
        return audio

    def stream(self, text: str) -> Iterator[bytes]:
        """Synthesise ``text`` and yield int16 PCM in ``chunk_ms`` pieces.

        Chunking after the fact does not lower first-audio latency (the
        whole sentence is computed first); it gives playback a cancellation
        point every ``chunk_ms`` so barge-in stops within one chunk.
        """
        if not text or not text.strip():
            return
        audio = self.synthesize(text)
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        step = max(1, int(self.format.sample_rate * self.chunk_ms / 1000))
        for i in range(0, pcm.size, step):
            yield pcm[i:i + step].tobytes()
