"""Run the voice agent inside a notebook — no server, no tunnel, no ports.

The Gradio demo needs a listening server *and* a browser route into it. In a
sandboxed hosted notebook either can be blocked, and then a URL is printed
that never resolves. This module removes both: audio comes from a file or
the browser's own recorder, and the answer is played with
``IPython.display.Audio``, which is just data in the output cell.

    from demo.notebook import NotebookAgent
    agent = NotebookAgent(adapter="/content/v2-final/best", tts="mms").load()

    agent.turn("question.wav")          # a file you uploaded
    agent.turn(record(5))               # or record from the microphone
    agent.history()                     # what the agent remembers
    agent.reset()

``turn`` prints the transcript, the answer, the measured latencies, and
displays the spoken reply as a playable widget. It returns the numbers too,
so a session can be logged rather than just watched.

**Which VAD is in play.** ``turn`` hands the whole recording to Whisper: a
fixed-length clip is already bounded, so nothing has to decide where speech
ends, and its ``response_latency_ms`` carries no endpoint term.
``stream_turn`` pushes the same clip through ``StreamingSession`` in
microphone-sized blocks, so the *online* endpointer decides where the
utterance ends and the number includes the silence wait the user really
sits through. Use ``turn`` to measure the model, ``stream_turn`` to measure
the agent.
"""

from __future__ import annotations

import base64
import io
import wave
from pathlib import Path

import numpy as np

__all__ = ["NotebookAgent", "record", "upload", "pcm_to_wav_bytes", "audio_report"]


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """int16 PCM → a WAV container, so any player can take it."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())
    return buffer.getvalue()


#: Peak the recorder normalizes to. Laptop microphones through a browser
#: often land near -24 dBFS, and Whisper transcribes such takes noticeably
#: worse than the same words at a healthy level. -3 dBFS leaves headroom
#: against clipping, which would be worse than quiet.
TARGET_PEAK_DBFS = -3.0


def normalize_peak(pcm: np.ndarray, target_dbfs: float = TARGET_PEAK_DBFS) -> tuple[np.ndarray, float]:
    """Scale int16 PCM so its peak sits at ``target_dbfs``. Returns the gain.

    Peak rather than RMS: it cannot clip, and it needs no assumption about
    how much of the clip is speech. Digital silence is left alone.
    """
    peak = float(np.abs(pcm).max()) if pcm.size else 0.0
    if peak <= 0:
        return pcm, 0.0
    target = 10 ** (target_dbfs / 20) * 32767.0
    gain = target / peak
    if 0.98 < gain < 1.02:
        return pcm, 0.0
    scaled = np.clip(pcm.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
    return scaled, 20 * float(np.log10(gain))


def _webm_to_wav(data: bytes, out_path: str, *, normalize: bool = True) -> str:
    """Decode whatever the browser recorded into 16 kHz mono WAV."""
    import av
    import soundfile as sf

    with av.open(io.BytesIO(data)) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16_000)
        chunks = []
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
    pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    pcm = pcm.astype(np.int16)
    if normalize:
        pcm, gain_db = normalize_peak(pcm)
        if gain_db:
            print(f"input gain {gain_db:+.1f} dB (peak normalized to "
                  f"{TARGET_PEAK_DBFS:.0f} dBFS)")
    sf.write(out_path, pcm, 16_000)
    return out_path


def record(seconds: float = 5.0, out_path: str = "recording.wav",
           *, normalize: bool = True) -> str:
    """Record from the browser microphone in Colab; returns a WAV path.

    Uses the page's own ``MediaRecorder`` and passes the bytes back through
    the kernel bridge, so nothing has to listen on a port. The take is peak
    normalized by default — browser microphone levels vary by more than
    10 dB between takes and the quieter ones transcribe worse. This is
    capture-stage conditioning and touches nothing in the evaluation path.
    """
    from google.colab import output
    from IPython.display import Javascript, display

    display(Javascript("""
      async function recordBlob(ms) {
        const stream = await navigator.mediaDevices.getUserMedia({audio: true});
        const rec = new MediaRecorder(stream);
        const parts = [];
        rec.ondataavailable = e => parts.push(e.data);
        rec.start();
        await new Promise(r => setTimeout(r, ms));
        rec.stop();
        await new Promise(r => rec.onstop = r);
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(parts);
        const buf = await blob.arrayBuffer();
        const bytes = new Uint8Array(buf);
        let bin = '';
        for (const b of bytes) bin += String.fromCharCode(b);
        return btoa(bin);
      }
      window.recordBlob = recordBlob;
    """))
    print(f"recording {seconds:.0f}s — speak now…")
    encoded = output.eval_js(f"recordBlob({int(seconds * 1000)})", timeout_sec=seconds + 30)
    print("recorded, decoding…")
    return _webm_to_wav(base64.b64decode(encoded), out_path, normalize=normalize)


def upload() -> str:
    """Pick a local audio file through the notebook's file dialog."""
    from google.colab import files

    uploaded = files.upload()
    name = next(iter(uploaded))
    if not name.lower().endswith(".wav"):
        return _webm_to_wav(uploaded[name], "uploaded.wav")
    Path(name).write_bytes(uploaded[name])
    return name


def audio_report(path: str, show: bool = True) -> dict:
    """Duration, level and where the offline VAD finds speech.

    Worth running before reading a surprising transcript: it separates "the
    model misheard" from "there was nothing there to hear".
    """
    from asr.explicit.mel import load_audio
    from asr.vad import VADConfig, detect_speech, frame_dbfs

    wave_, seconds = load_audio(str(path))
    levels = [frame_dbfs(wave_[i:i + 480]) for i in range(0, max(1, len(wave_) - 480), 160)]
    levels = np.asarray(levels) if levels else np.zeros(1)
    segments = detect_speech(wave_, 16_000, VADConfig())
    speech = sum(s.duration_seconds for s in segments)
    report = {
        "path": str(path), "seconds": round(seconds, 2),
        "peak_dbfs": round(float(levels.max()), 1),
        "median_dbfs": round(float(np.median(levels)), 1),
        "noise_floor_dbfs": round(float(np.percentile(levels, 5)), 1),
        "speech_segments": len(segments),
        "speech_seconds": round(speech, 2),
        "speech_fraction": round(speech / seconds, 2) if seconds else 0.0,
        "spans": [(round(s.start_seconds, 2), round(s.end_seconds, 2)) for s in segments],
    }
    if show:
        print(f"{report['seconds']}s  peak {report['peak_dbfs']} dBFS  "
              f"median {report['median_dbfs']}  floor {report['noise_floor_dbfs']}")
        print(f"offline VAD: {report['speech_seconds']}s speech in "
              f"{report['speech_segments']} segment(s) "
              f"({report['speech_fraction'] * 100:.0f}% of the clip)  {report['spans']}")
    return report


class NotebookAgent:
    """The voice agent with a notebook front end. Same components as the
    live agent and the Gradio demo, so the numbers are comparable."""

    def __init__(
        self,
        *,
        whisper: str = "openai/whisper-large-v3-turbo",
        adapter: str | None = None,
        llm: str = "Qwen/Qwen3-4B",
        tts: str = "mms",
        language: str = "hi",
        device: str | None = None,
        history_turns: int = 6,
        asr_engine: str = "explicit",
        ct2_model: str | None = None,
        ct2_compute_type: str = "int8_float16",
        llm_engine: str = "explicit",
        llm_base_url: str = "http://127.0.0.1:8000/v1",
        llm_api_key: str | None = None,
        max_history_tokens: int = 800,
    ):
        self.language = language
        self.config = {"whisper": whisper, "adapter": adapter, "llm": llm, "tts": tts,
                       "device": device, "history_turns": history_turns,
                       "asr_engine": asr_engine, "ct2_model": ct2_model,
                       "ct2_compute_type": ct2_compute_type,
                       "llm_engine": llm_engine, "llm_base_url": llm_base_url,
                       "max_history_tokens": max_history_tokens}
        self.llm_api_key = llm_api_key
        self.llm_info: dict = {}
        self.asr = None
        self.llm_runner = None
        self.synth = None
        self.conversation = None
        self.turns: list[dict] = []

    def load(self) -> NotebookAgent:
        if self.asr is not None:
            return self
        from agent import Conversation
        from llm.engines import build_asr, build_llm

        cfg = self.config
        print(f"loading asr ({cfg['asr_engine']}): {cfg['whisper']}"
              + (f" + {cfg['adapter']}" if cfg["adapter"] else ""), flush=True)
        self.asr = build_asr(
            cfg["asr_engine"], model=cfg["whisper"], adapter=cfg["adapter"],
            device=cfg["device"], ct2_model=cfg["ct2_model"],
            ct2_compute_type=cfg["ct2_compute_type"],
            language_candidates=["hi", "en", "te"],
        )
        print(f"loading llm ({cfg['llm_engine']}): {cfg['llm']}", flush=True)
        self.llm_runner, tokenizer, self.llm_info = build_llm(
            cfg["llm_engine"], model=cfg["llm"], device=cfg["device"],
            base_url=cfg["llm_base_url"], api_key=self.llm_api_key,
        )
        self._tokenizer = tokenizer
        if cfg["llm_engine"] == "http":
            print("probing the llm endpoint:", self.llm_runner.probe(), flush=True)
        if cfg["tts"] == "mms":
            from tts.local import MmsTtsSynthesizer

            self.synth = MmsTtsSynthesizer(self.language)
            print(f"tts warm-up {self.synth.warm_up():.0f} ms", flush=True)
        else:
            from tts import EdgeStreamingSynthesizer

            self.synth = EdgeStreamingSynthesizer(language=self.language)
        if cfg["history_turns"]:
            self.conversation = Conversation(
                max_turns=cfg["history_turns"],
                max_history_tokens=cfg["max_history_tokens"],
                tokenizer=tokenizer,
            )
        print(f"ready: {self.llm_info}", flush=True)
        return self

    # -- one turn ---------------------------------------------------------

    def turn(self, audio_path: str, *, language: str | None = None,
             max_tokens: int = 96, speak: bool = True, quiet: bool = False) -> dict:
        """Transcribe a WAV, answer it, play the reply. Returns the metrics."""
        self.load()
        from agent.audio import DecodingBufferSink
        from agent.turn import VoiceTurn

        asr = self.asr.transcribe_file(str(audio_path),
                                       language=language or self.language)
        sink = DecodingBufferSink(self.synth.format)
        turn = VoiceTurn(self.llm_runner, self.synth, response_language="Hindi",
                         llm_max_tokens=max_tokens, conversation=self.conversation)
        result = turn.run(asr.text, speech_end_to_transcript_ms=asr.metrics.total_ms,
                          sink=sink if speak else None)

        record_ = {
            "audio": str(audio_path),
            "audio_seconds": round(asr.metrics.audio_duration_seconds, 2),
            "transcript": asr.text,
            "response": result.response,
            "state": result.state.value,
            "asr_ms": round(asr.metrics.total_ms, 1),
            "asr_rtf": (None if asr.metrics.real_time_factor is None
                        else round(asr.metrics.real_time_factor, 3)),
            "asr_hit_token_budget": asr.metrics.hit_token_budget,
            "first_token_ms": result.metrics.final_transcript_to_first_llm_token_ms,
            "to_audio_ms": result.metrics.first_llm_token_to_playback_start_ms,
            "response_latency_ms": result.metrics.response_latency_ms,
            "llm_total_ms": result.metrics.llm_total_ms,
            "tts_first_chunk_ms": result.metrics.tts_first_chunk_ms,
            "turn_total_ms": result.metrics.total_turn_ms,
            "spoken_seconds": round(sink.seconds, 2) if speak else 0.0,
            "history_turns": self.conversation.snapshot()["turns"] if self.conversation else 0,
        }
        self.turns.append(record_)
        if not quiet:
            self._show(record_, sink if speak else None)
        return record_

    def _show(self, record_: dict, sink) -> None:
        from IPython.display import Audio, Markdown, display

        def ms(value):
            return "—" if value is None else f"{value:.0f} ms"

        lines = [
            f"**You said:** {record_['transcript']}",
            f"**Agent:** {record_['response']}",
            "",
            "| segment | time |", "| --- | ---: |",
            f"| ASR ({record_['audio_seconds']}s audio, RTF {record_['asr_rtf']}) "
            f"| {ms(record_['asr_ms'])} |",
            f"| transcript → first LLM token | {ms(record_['first_token_ms'])} |",
            f"| first token → audio out | {ms(record_['to_audio_ms'])} |",
            f"| **response latency** | **{ms(record_['response_latency_ms'])}** |",
            f"| turn total | {ms(record_['turn_total_ms'])} |",
        ]
        if record_["asr_hit_token_budget"]:
            lines.append("\n⚠️ the transcript was cut off at the token budget")
        if record_["state"] != "completed":
            lines.append(f"\n⚠️ turn ended **{record_['state']}**")
        if record_["history_turns"]:
            lines.append(f"\n_{record_['history_turns']} turn(s) in memory_")
        display(Markdown("\n".join(lines)))
        if sink is not None and sink.audio.size:
            display(Audio(pcm_to_wav_bytes(sink.audio, self.synth.format.sample_rate),
                          autoplay=True))

    def stream_turn(self, audio_path: str, *, max_tokens: int = 96, speak: bool = True,
                    block_ms: int = 100, trailing_silence: float = 1.5,
                    quiet: bool = False, **stream_options) -> dict:
        """Replay a clip through ``StreamingSession`` and answer every final.

        The online endpointer decides where each utterance ends, partials
        appear as audio arrives, and a candidate final decoded during the
        silence wait becomes the final at no extra cost. The clock advances
        with the audio, not with wall time, so ASR compute does not distort
        the endpoint decision — which is also why ``endpoint_to_final_ms``
        below is reconstructed (silence waited + ASR after the endpoint)
        rather than measured with a stopwatch.

        ``trailing_silence`` is appended because a recording that stops the
        instant you stop talking never contains ``min_silence_ms`` of quiet;
        without it the utterance would only close at ``flush()``.
        ``stream_options`` go to :class:`asr.streaming.StreamingConfig`, so
        ``incremental_finals=True`` or ``semantic_endpointing=True`` can be
        tried from the notebook.
        """
        self.load()
        from agent.audio import DecodingBufferSink
        from agent.turn import VoiceTurn
        from asr.explicit.mel import load_audio
        from asr.streaming import StreamingConfig, StreamingSession, UpdateKind

        class _AudioClock:
            now = 0.0

            def __call__(self):
                return self.now

        wave_, seconds = load_audio(str(audio_path))
        wave_ = np.concatenate([
            wave_, np.zeros(int(trailing_silence * 16_000), dtype=np.float32),
        ])
        config = StreamingConfig(language=self.language, emit_partials=True, **stream_options)
        clock = _AudioClock()
        session = StreamingSession(self.asr, config, clock=clock)

        block = 16_000 * block_ms // 1000
        updates = []
        for i in range(0, len(wave_), block):
            chunk = wave_[i:i + block]
            clock.now += len(chunk) / 16_000
            updates.extend(session.push(chunk))
        updates.extend(session.flush())

        finals = [u for u in updates if u.kind == UpdateKind.FINAL and u.text.strip()]
        trace = [{"kind": u.kind.value, "at": round(u.stream_seconds, 2),
                  "asr_ms": round(u.asr_ms, 1), "text": u.text,
                  "flags": ", ".join(f for f, on in
                                     (("no_speech", u.no_speech), ("looped", u.degenerate))
                                     if on)}
                 for u in updates if u.text.strip() or u.no_speech or u.degenerate]

        answers = []
        for final in finals:
            # Silence the endpointer waited, plus ASR that ran after the
            # endpoint (zero when the candidate was reused).
            wait_ms = max(0.0, (final.stream_seconds
                                - (final.utterance_start_seconds + final.audio_seconds)) * 1000
                          + config.vad.padding_ms)
            endpoint_to_final = wait_ms + final.asr_ms_after_endpoint
            sink = DecodingBufferSink(self.synth.format)
            turn = VoiceTurn(self.llm_runner, self.synth, response_language="Hindi",
                             llm_max_tokens=max_tokens, conversation=self.conversation)
            result = turn.run(final.text, speech_end_to_transcript_ms=endpoint_to_final,
                              sink=sink if speak else None)
            answers.append({
                "transcript": final.text, "response": result.response,
                "state": result.state.value,
                "endpoint_reason": final.endpoint_reason.value if final.endpoint_reason else None,
                "from_candidate": final.from_candidate,
                "utterance_seconds": round(final.audio_seconds, 2),
                "asr_ms": round(final.asr_ms, 1),
                "asr_after_endpoint_ms": round(final.asr_ms_after_endpoint, 1),
                "silence_wait_ms": round(wait_ms, 1),
                "endpoint_to_final_ms": round(endpoint_to_final, 1),
                "first_token_ms": result.metrics.final_transcript_to_first_llm_token_ms,
                "to_audio_ms": result.metrics.first_llm_token_to_playback_start_ms,
                "response_latency_ms": result.metrics.response_latency_ms,
                # Decomposition of to_audio_ms: how much was the LLM still
                # generating, and how much was synthesis. Without these the
                # segment is one opaque number and the wrong stage gets blamed.
                "llm_total_ms": result.metrics.llm_total_ms,
                "tts_first_chunk_ms": result.metrics.tts_first_chunk_ms,
                "first_token_to_first_unit_ms": result.metrics.first_token_to_first_unit_ms,
                "tts_synthesis_ms": result.metrics.tts_synthesis_ms,
                "llm_tokens": result.metrics.llm_generated_tokens,
                "spoken_units": len(result.speech.sentences) if result.speech else 0,
                "first_unit_chars": (len(result.speech.sentences[0])
                                     if result.speech and result.speech.sentences else 0),
                # Prefill grows with the prompt, and the prompt grows with
                # history: across 12 live turns first_token_ms rose 209 -> 1474
                # ms (r=0.99 against turn index). Recording the history size
                # makes that visible per turn instead of hidden in a mean.
                "history_turns": (self.conversation.snapshot()["turns"]
                                  if self.conversation else 0),
                "history_tokens": (self.conversation.history_tokens()
                                   if self.conversation else 0),
                "sink": sink if speak else None,
            })

        dropped = [u for u in updates
                   if u.kind == UpdateKind.FINAL and (u.no_speech or u.degenerate)]
        record_ = {
            "audio": str(audio_path), "audio_seconds": round(seconds, 2),
            # Without these a latency distribution cannot be attributed to a
            # configuration: 12 records were persisted with no LLM name, TTS
            # backend or device, so they supported a latency claim but not a
            # comparison between candidates.
            "config": {
                "whisper": self.config["whisper"], "adapter": self.config["adapter"],
                "asr_engine": self.config["asr_engine"],
                "ct2_compute_type": (self.config["ct2_compute_type"]
                                     if self.config["asr_engine"] == "ct2" else None),
                "llm": self.config["llm"], "tts": self.config["tts"],
                "language": self.language, "max_tokens": max_tokens,
                "max_history_tokens": self.config["max_history_tokens"],
                "history_turns_config": self.config["history_turns"],
                **self.llm_info,
            },
            "vad": config.vad.as_dict(),
            "utterances": len(finals),
            "dropped_finals": len(dropped),
            "partials": sum(1 for u in updates if u.kind == UpdateKind.PARTIAL),
            "candidates": sum(1 for u in updates if u.kind == UpdateKind.CANDIDATE),
            "trace": trace,
            "answers": [{k: v for k, v in a.items() if k != "sink"} for a in answers],
        }
        # Persist the complete per-answer record, not a latency-only projection:
        # turns.jsonl is the live-run evidence and must retain the transcript,
        # endpoint timing, and LLM/TTS decomposition used in the ledger.
        self.turns.extend(
            {
                "audio": str(audio_path),
                "streaming": True,
                "asr_rtf": None,
                "turn_total_ms": None,
                **{key: value for key, value in answer.items() if key != "sink"},
            }
            for answer in answers
        )
        if not quiet:
            self._show_stream(record_, answers)
        return record_

    def _show_stream(self, record_: dict, answers: list[dict]) -> None:
        from IPython.display import Audio, Markdown, display

        def ms(value):
            return "—" if value is None else f"{value:.0f} ms"

        vad = record_["vad"]
        lines = [
            f"**online endpointer** on {record_['audio_seconds']}s of audio: "
            f"{record_['utterances']} utterance(s), {record_['partials']} partial(s), "
            f"{record_['candidates']} candidate(s)"
            + (f", {record_['dropped_finals']} dropped (no speech / repetition loop)"
               if record_.get("dropped_finals") else ""),
            f"_min_silence {vad['min_silence_ms']} ms · padding {vad['padding_ms']} ms · "
            f"threshold {'adaptive' if vad['adaptive_threshold'] else vad['threshold_dbfs']}_",
            "",
        ]
        for event in record_["trace"]:
            flags = f" **[{event['flags']}]**" if event.get("flags") else ""
            text = event["text"][:120] + ("…" if len(event["text"]) > 120 else "")
            lines.append(f"- `{event['kind']:9s}` @{event['at']:5.1f}s "
                         f"({event['asr_ms']:.0f} ms){flags} {text}")
        for a in answers:
            lines += [
                "", f"**You:** {a['transcript']}", f"**Agent:** {a['response']}", "",
                "| segment | time |", "| --- | ---: |",
                f"| silence wait (`min_silence_ms`) | {ms(a['silence_wait_ms'])} |",
                f"| ASR after the endpoint | {ms(a['asr_after_endpoint_ms'])}"
                + (" _(candidate reused)_" if a["from_candidate"] else "") + " |",
                f"| → final transcript | {ms(a['endpoint_to_final_ms'])} |",
                f"| transcript → first LLM token | {ms(a['first_token_ms'])} |",
                f"| first token → audio out | {ms(a['to_audio_ms'])} |",
                f"| ⤷ LLM generating until a unit was speakable "
                f"({a['llm_tokens']} tokens in the reply) "
                f"| {ms(a['first_token_to_first_unit_ms'])} |",
                f"| ⤷ synthesis of that unit ({a['first_unit_chars']} chars, "
                f"{a['spoken_units']} unit(s) total) | {ms(a['tts_synthesis_ms'])} |",
                f"| **response latency** (as the user feels it) "
                f"| **{ms(a['response_latency_ms'])}** |",
                f"\n_utterance {a['utterance_seconds']}s, whole-utterance ASR "
                f"{ms(a['asr_ms'])}, endpoint: {a['endpoint_reason']}_",
            ]
        display(Markdown("\n".join(lines)))
        for a in answers:
            if a["sink"] is not None and a["sink"].audio.size:
                display(Audio(pcm_to_wav_bytes(a["sink"].audio,
                                               self.synth.format.sample_rate), autoplay=True))

    # -- conversation -----------------------------------------------------

    def history(self) -> None:
        from IPython.display import Markdown, display

        from demo.app import history_markdown

        display(Markdown(history_markdown(self.conversation)))

    def reset(self) -> None:
        if self.conversation is not None:
            self.conversation.reset()
        print("conversation cleared")

    def summary(self) -> dict:
        """Aggregate streaming and offline turns separately."""
        if not self.turns:
            return {}

        def measures(rows: list[dict]) -> dict:
            def mean(key):
                values = [row[key] for row in rows if row.get(key) is not None]
                return round(float(np.mean(values)), 1) if values else None

            return {
                "turns": len(rows),
                "asr_ms_mean": mean("asr_ms"),
                "asr_rtf_mean": mean("asr_rtf"),
                "first_token_ms_mean": mean("first_token_ms"),
                "to_audio_ms_mean": mean("to_audio_ms"),
                "response_latency_ms_mean": mean("response_latency_ms"),
                "turn_total_ms_mean": mean("turn_total_ms"),
            }

        streaming = [row for row in self.turns if row.get("streaming")]
        offline = [row for row in self.turns if not row.get("streaming")]
        return {"streaming": measures(streaming), "offline": measures(offline)}
