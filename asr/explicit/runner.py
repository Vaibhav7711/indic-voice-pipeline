"""End-to-end ASR runner: mel → encoder → autoregressive decoder.

Owns the full inference loop with per-stage CUDA timing. model.generate()
and pipeline("asr") are never called — correctness references only in tests.
"""

from __future__ import annotations

import gzip
from dataclasses import asdict, dataclass, field
from time import perf_counter_ns

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from asr.explicit.chunking import AudioChunk, chunk_audio, merge_overlapping_transcripts
from asr.explicit.decoder import WhisperDecoder
from asr.explicit.encoder import WhisperEncoder
from asr.explicit.mel import extract_mel, load_audio, load_audio_from_array
from asr.explicit.timing import peak_allocated, peak_reserved, reset_peak


def resolve_no_speech_token(processor, model) -> int | None:
    """The model's own ``<|nospeech|>`` id, or None if it has none.

    Checkpoints disagree on both the spelling and the id: large-v3 family
    uses ``<|nospeech|>`` at 50363, while small/medium use ``<|nocaptions|>``
    and put ``<|notimestamps|>`` at 50363. Asking the tokenizer for a token
    it does not have returns the unk/eos id, which would silently read the
    wrong distribution, so a resolved id is only accepted when it round-trips
    back to the name asked for.
    """
    generation_config = getattr(model, "generation_config", None)
    for attr in ("no_speech_token_id", "no_speech_token"):
        value = getattr(generation_config, attr, None)
        if isinstance(value, int):
            return value

    tokenizer = getattr(processor, "tokenizer", None)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    back = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert) or not callable(back):
        return None
    for name in ("<|nospeech|>", "<|nocaptions|>"):
        try:
            token_id = convert(name)
            if isinstance(token_id, int) and back(token_id) == name:
                return token_id
        except Exception:  # noqa: BLE001 - an unusable tokenizer disables the check
            continue
    return None


def compression_ratio(text: str) -> float:
    """``len(text) / len(gzip(text))``. Whisper's own repetition detector:
    natural language compresses ~1.5-2x, a repeated phrase far more."""
    if not text:
        return 0.0
    raw = text.encode("utf-8")
    return len(raw) / max(1, len(gzip.compress(raw)))


@dataclass
class ASRMetrics:
    """Per-stage latency breakdown — the whole point of owning the loop."""

    audio_load_ms: float = 0.0
    mel_extraction_ms: float = 0.0
    encoder_ms: float = 0.0
    decoder_prefill_ms: float = 0.0
    decoder_steps: int = 0
    decode_ms: list[float] = field(default_factory=list)
    total_ms: float = 0.0
    audio_duration_seconds: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    encoder_hidden_size: int = 0
    encoder_seq_length: int = 0
    #: Whisper's own P(<|nospeech|>) for this window, read where generate()
    #: reads it. High values mean the audio has no speech in it.
    no_speech_prob: float | None = None
    #: gzip compression ratio of the transcript. Whisper's reference
    #: implementation treats > 2.4 as a repetition loop; repeated text
    #: compresses far better than language does.
    compression_ratio: float = 0.0
    #: The decode stopped because an n-gram repeated, not at EOS.
    stopped_on_repetition: bool = False
    #: The window was judged to contain no speech and was not transcribed.
    no_speech: bool = False
    #: Tokens the decode loop was allowed to emit for this utterance.
    token_budget: int = 0
    #: True when decoding stopped because the budget ran out rather than at
    #: EOS — the transcript is cut, usually mid-word. Never let this be
    #: silent: a truncated hypothesis inflates WER as deletions.
    hit_token_budget: bool = False
    #: Only set when ``language=None`` triggered detection.
    language_detection_ms: float | None = None
    language_probability: float | None = None

    @property
    def mean_decode_ms(self) -> float:
        return sum(self.decode_ms) / len(self.decode_ms) if self.decode_ms else 0.0

    @property
    def total_decode_ms(self) -> float:
        return sum(self.decode_ms)

    @property
    def real_time_factor(self) -> float:
        """RTF = processing_time / audio_duration. < 1.0 = faster than real-time."""
        if self.audio_duration_seconds <= 0:
            return 0.0
        return (self.total_ms / 1000.0) / self.audio_duration_seconds

    def as_dict(self) -> dict:
        d = asdict(self)
        d["mean_decode_ms"] = self.mean_decode_ms
        d["total_decode_ms"] = self.total_decode_ms
        d["real_time_factor"] = self.real_time_factor
        return d


@dataclass
class ASRResult:
    text: str
    token_ids: list[int]
    language: str | None
    metrics: ASRMetrics


@dataclass
class LongFormMetrics:
    """Aggregate timing and chunk geometry for a long-form transcription."""

    audio_duration_seconds: float
    chunk_count: int
    audio_load_ms: float
    chunk_asr_ms: float
    total_ms: float
    # Decode-guard state aggregated over the windows. Without these a
    # long-form result carries no guard information at all, so the streaming
    # session's degenerate-final filtering was inert on every utterance long
    # enough to be chunked — exactly the ones most likely to loop.
    no_speech: bool = False
    stopped_on_repetition: bool = False
    hit_token_budget: bool = False
    compression_ratio: float = 0.0

    @property
    def real_time_factor(self) -> float:
        if self.audio_duration_seconds <= 0:
            return 0.0
        return (self.total_ms / 1000.0) / self.audio_duration_seconds

    def as_dict(self) -> dict:
        return {
            "audio_duration_seconds": self.audio_duration_seconds,
            "chunk_count": self.chunk_count,
            "audio_load_ms": self.audio_load_ms,
            "chunk_asr_ms": self.chunk_asr_ms,
            "total_ms": self.total_ms,
            "real_time_factor": self.real_time_factor,
            "no_speech": self.no_speech,
            "stopped_on_repetition": self.stopped_on_repetition,
            "hit_token_budget": self.hit_token_budget,
            "compression_ratio": self.compression_ratio,
        }


@dataclass
class LongFormASRResult:
    text: str
    language: str | None
    chunks: list[AudioChunk]
    chunk_results: list[ASRResult]
    metrics: LongFormMetrics


class ASRRunner:
    """Explicit Whisper ASR runtime.

    Owns: mel extraction, encoder forward, autoregressive decoder, timing.
    Does NOT call: model.generate(), pipeline("asr").
    """

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        processor: WhisperProcessor,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        *,
        language_candidates: list[str] | None = None,
        max_new_tokens: int | None = None,
        no_speech_threshold: float | None = 0.6,
        loop_guard_ngram: int = 3,
        loop_guard_repeats: int = 4,
        compression_ratio_threshold: float = 2.4,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype
        #: When ``language=None``, detection chooses among these codes only.
        #: None means every language Whisper knows.
        self.language_candidates = language_candidates
        # Whisper's decoder has a fixed number of positions (448); minus the
        # 3-4 prompt tokens that is the most it can emit. The old default of
        # 225 was half of that and silently truncated long Devanagari
        # utterances: Hindi costs ~6 BPE tokens per word, so a 50-word
        # sentence needs ~300 tokens. Default to the model's real limit.
        self.max_target_positions = int(getattr(model.config, "max_target_positions", 448))
        self.default_max_new_tokens = max_new_tokens
        # Safeguards `model.generate()` applies and a bare greedy loop does
        # not. Without them, marginal audio (a cough, room noise, one
        # syllable) is transcribed as a phrase and then repeated until the
        # token budget runs out — ~2 s of GPU time for garbage, which then
        # goes to the LLM as if it were a question.
        #: Skip transcription when P(<|nospeech|>) exceeds this. None disables.
        self.no_speech_threshold = no_speech_threshold
        #: Stop when the last ``loop_guard_ngram`` tokens have repeated
        #: ``loop_guard_repeats`` times in a row. 0 disables.
        self.loop_guard_ngram = loop_guard_ngram
        self.loop_guard_repeats = loop_guard_repeats
        #: Flag (do not discard) transcripts that compress this well.
        self.compression_ratio_threshold = compression_ratio_threshold
        self.encoder = WhisperEncoder(model, device)
        self.decoder = WhisperDecoder(
            model, device, no_speech_token_id=resolve_no_speech_token(processor, model),
        )
        self.eos_token_id = model.generation_config.eos_token_id

    def transcribe_file(
        self,
        path: str,
        *,
        language: str | None = None,
        max_new_tokens: int | None = None,
    ) -> ASRResult:
        """Transcribe an audio file with full latency breakdown."""
        total_start = perf_counter_ns()
        reset_peak(self.device)

        load_start = perf_counter_ns()
        waveform, duration = load_audio(path)
        load_ms = (perf_counter_ns() - load_start) / 1_000_000

        return self._transcribe(waveform, duration, load_ms, language,
                                max_new_tokens, total_start)

    def transcribe_array(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        language: str | None = None,
        max_new_tokens: int | None = None,
    ) -> ASRResult:
        """Transcribe an in-memory waveform (for Gradio/pipeline use)."""
        total_start = perf_counter_ns()
        reset_peak(self.device)

        load_start = perf_counter_ns()
        waveform, duration = load_audio_from_array(waveform, sample_rate)
        load_ms = (perf_counter_ns() - load_start) / 1_000_000

        return self._transcribe(waveform, duration, load_ms, language,
                                max_new_tokens, total_start)

    def transcribe_long_file(
        self,
        path: str,
        *,
        language: str | None = None,
        chunk_seconds: float = 25.0,
        overlap_seconds: float = 5.0,
        max_new_tokens: int | None = None,
    ) -> LongFormASRResult:
        """Transcribe arbitrary-length file audio through bounded Whisper windows."""
        load_start = perf_counter_ns()
        waveform, _ = load_audio(path)
        load_ms = (perf_counter_ns() - load_start) / 1_000_000
        return self._transcribe_long_normalized(
            waveform, load_ms, language, chunk_seconds, overlap_seconds, max_new_tokens,
        )

    def transcribe_long_array(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        language: str | None = None,
        chunk_seconds: float = 25.0,
        overlap_seconds: float = 5.0,
        max_new_tokens: int | None = None,
    ) -> LongFormASRResult:
        """Transcribe arbitrary-length in-memory audio with overlap-aware merging."""
        load_start = perf_counter_ns()
        waveform, _ = load_audio_from_array(waveform, sample_rate)
        load_ms = (perf_counter_ns() - load_start) / 1_000_000
        return self._transcribe_long_normalized(
            waveform, load_ms, language, chunk_seconds, overlap_seconds, max_new_tokens,
        )

    def _transcribe_long_normalized(
        self,
        waveform: np.ndarray,
        load_ms: float,
        language: str | None,
        chunk_seconds: float,
        overlap_seconds: float,
        max_new_tokens: int | None,
    ) -> LongFormASRResult:
        total_start = perf_counter_ns()
        windows = chunk_audio(
            waveform, 16_000, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds,
        )
        results: list[ASRResult] = []
        text = ""
        for _spec, chunk in windows:
            # The waveform is already normalized to 16 kHz, so this call adds no resampling.
            result = self.transcribe_array(
                chunk, 16_000, language=language, max_new_tokens=max_new_tokens,
            )
            results.append(result)
            text = merge_overlapping_transcripts(text, result.text)

        total_ms = (perf_counter_ns() - total_start) / 1_000_000 + load_ms
        return LongFormASRResult(
            text=text,
            # With language=None each window detects independently; report
            # the first window's decision as the utterance language.
            language=language or (results[0].language if results else None),
            chunks=[spec for spec, _ in windows],
            chunk_results=results,
            metrics=LongFormMetrics(
                audio_duration_seconds=len(waveform) / 16_000,
                chunk_count=len(windows),
                audio_load_ms=load_ms,
                chunk_asr_ms=sum(result.metrics.total_ms for result in results),
                total_ms=total_ms,
                # A guard that fired on any window describes the whole result:
                # one looped or truncated window poisons the stitched text.
                no_speech=all(r.metrics.no_speech for r in results) if results else False,
                stopped_on_repetition=any(r.metrics.stopped_on_repetition for r in results),
                hit_token_budget=any(r.metrics.hit_token_budget for r in results),
                compression_ratio=compression_ratio(text),
            ),
        )

    def token_budget(self, language: str | None, requested: int | None) -> int:
        """Tokens the decode loop may emit: the model's limit, or less if the
        caller asked for less. Asking for more than the model supports would
        run off the end of its position embeddings."""
        prompt_len = len(self.decoder._build_prompt_ids(language or "en"))
        limit = self.max_target_positions - prompt_len
        wanted = requested if requested is not None else self.default_max_new_tokens
        return limit if wanted is None else max(1, min(int(wanted), limit))

    def _transcribe(
        self,
        waveform: np.ndarray,
        duration: float,
        load_ms: float,
        language: str | None,
        max_new_tokens: int | None,
        total_start: int,
    ) -> ASRResult:
        metrics = ASRMetrics(audio_load_ms=load_ms, audio_duration_seconds=duration)

        # 1. Mel spectrogram.
        mel = extract_mel(waveform, duration, self.processor, self.device, self.dtype)
        metrics.mel_extraction_ms = mel.mel_ms

        # 2. Encoder forward.
        enc = self.encoder.forward(mel.input_features)
        metrics.encoder_ms = enc.encoder_ms
        metrics.encoder_hidden_size = enc.hidden_size
        metrics.encoder_seq_length = enc.sequence_length

        # 3. Language: detect from one decoder step when not given, so the
        #    prompt is always the well-formed <|sot|><|lang|><|task|> sequence.
        if language is None:
            language, prob, detect_ms = self.decoder.detect_language(
                enc.encoder_outputs, candidates=self.language_candidates,
            )
            metrics.language_detection_ms = detect_ms
            metrics.language_probability = prob

        # 4. Decoder prefill.
        state, prefill_ms = self.decoder.prefill(
            enc.encoder_outputs, language=language,
        )
        metrics.decoder_prefill_ms = prefill_ms

        # 5. Autoregressive decode loop.
        max_new_tokens = self.token_budget(language, max_new_tokens)
        metrics.token_budget = max_new_tokens
        eos_ids = (
            {self.eos_token_id}
            if isinstance(self.eos_token_id, int)
            else set(self.eos_token_id or [])
        )

        metrics.no_speech_prob = state.no_speech_prob
        if (
            self.no_speech_threshold is not None
            and state.no_speech_prob is not None
            and state.no_speech_prob >= self.no_speech_threshold
        ):
            # Whisper says there is nothing to transcribe. Returning empty is
            # the honest answer; decoding anyway produces a hallucination.
            metrics.no_speech = True
            metrics.decoder_steps = 0
            metrics.total_ms = (perf_counter_ns() - total_start) / 1_000_000
            metrics.peak_allocated_bytes = peak_allocated(self.device)
            metrics.peak_reserved_bytes = peak_reserved(self.device)
            return ASRResult("", [], language, metrics)

        n, repeats = self.loop_guard_ngram, self.loop_guard_repeats
        for step in range(max_new_tokens):
            token_id = int(state.next_token.item())
            if token_id in eos_ids:
                state.decoded_tokens.append(token_id)
                break

            state.decoded_tokens.append(token_id)
            if n and repeats and len(state.decoded_tokens) >= n * repeats:
                tail = state.decoded_tokens[-n * repeats:]
                if all(tail[i * n:(i + 1) * n] == tail[:n] for i in range(1, repeats)):
                    # A repetition loop: drop the repeats and stop. Keeping
                    # one copy loses nothing real and saves the rest of the
                    # token budget.
                    del state.decoded_tokens[-n * (repeats - 1):]
                    metrics.stopped_on_repetition = True
                    break
            if step == max_new_tokens - 1:
                break

            old_tokens = state.decoded_tokens
            state, step_ms = self.decoder.decode_one(state)
            state.decoded_tokens = old_tokens
            metrics.decode_ms.append(step_ms)

        metrics.decoder_steps = len(state.decoded_tokens)
        metrics.hit_token_budget = (
            len(state.decoded_tokens) >= max_new_tokens
            and (not state.decoded_tokens or state.decoded_tokens[-1] not in eos_ids)
        )
        metrics.total_ms = (perf_counter_ns() - total_start) / 1_000_000
        metrics.peak_allocated_bytes = peak_allocated(self.device)
        metrics.peak_reserved_bytes = peak_reserved(self.device)

        text = self.processor.tokenizer.decode(
            state.decoded_tokens, skip_special_tokens=True,
        ).strip()
        metrics.compression_ratio = compression_ratio(text)

        return ASRResult(text, state.decoded_tokens, language, metrics)
