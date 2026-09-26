"""Gradio demo: speak Hindi, get a spoken Hindi answer, see where the time went.

Two tabs, because they show different things:

**Turn** — record, submit, get a transcript, an answer and audio, with the
latency breakdown and the dialogue history. This is the reliable path and
uses the same `VoiceTurn` the live agent does, so the numbers on screen are
the numbers the agent measures.

**Stream** — microphone chunks feed the real `StreamingSession`, so partial
transcripts appear while you are still talking and the endpointer decides
when you stopped. What it demonstrates is the endpointing and the partials;
it depends on the browser delivering audio steadily, so the Turn tab stays
the one to trust for measurement.

Configuration is by environment variable so the demo can be pointed at a new
adapter without editing code:

    WHISPER_MODEL   default openai/whisper-large-v3-turbo
    WHISPER_ADAPTER_PATH   a local directory or a Hub id (the shipped v2 adapter)
    LLM_MODEL       default Qwen/Qwen3-1.7B
    TTS_BACKEND     edge (network, default) | mms (local)
    DEVICE          cuda | cpu (default: cuda when available)

    python demo/app.py --preload                 # load models before serving
    python demo/app.py --no-share --port 7860    # local only

``--preload`` matters on a fresh machine: the first turn otherwise spends a
minute downloading Whisper, the LLM and the TTS voice with no feedback in
the browser, which looks like a broken page. ``--share`` uses Gradio's
tunnel, which some hosts block; the launcher reports whether the tunnel
actually came up instead of printing a URL that may not resolve.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

LANGUAGES = ["hi", "en", "te"]
DEFAULT_WHISPER = "openai/whisper-large-v3-turbo"
DEFAULT_LLM = "Qwen/Qwen3-1.7B"


# --------------------------------------------------------------------------
# Pure formatting helpers (no models; unit-tested)
# --------------------------------------------------------------------------


def to_float_mono(audio) -> tuple[np.ndarray, int]:
    """Gradio numpy audio ``(sample_rate, array)`` → float32 mono in [-1, 1]."""
    sample_rate, wave = audio
    wave = np.asarray(wave)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if wave.dtype.kind in "iu":
        wave = wave.astype(np.float32) / float(np.iinfo(wave.dtype).max)
    return wave.astype(np.float32), int(sample_rate)


def _ms(value) -> str:
    return "—" if value is None else f"{value:.0f}"


def latency_table(asr_metrics, turn_metrics) -> str:
    """Markdown table of the segments the user actually waits through.

    ASR here is the whole-utterance transcribe the Turn tab performs, so it
    stands in for "speech end → final transcript"; in the streaming path that
    segment also includes the endpointer's silence wait, and a candidate
    final removes most of the ASR from it (see docs/STREAMING.md).
    """
    rows = [
        ("Mel extraction", _ms(getattr(asr_metrics, "mel_extraction_ms", None))),
        ("Whisper encoder", _ms(getattr(asr_metrics, "encoder_ms", None))),
        ("Whisper decode", _ms(getattr(asr_metrics, "total_decode_ms", None))),
        ("**ASR total**", f"**{_ms(getattr(asr_metrics, 'total_ms', None))}**"),
        ("Transcript → first LLM token", _ms(
            getattr(turn_metrics, "final_transcript_to_first_llm_token_ms", None))),
        ("First token → audio out", _ms(
            getattr(turn_metrics, "first_llm_token_to_playback_start_ms", None))),
        ("**Response latency** (excl. endpoint silence)",
         f"**{_ms(getattr(turn_metrics, 'response_latency_ms', None))}**"),
        ("LLM total", _ms(getattr(turn_metrics, "llm_total_ms", None))),
        ("TTS first chunk", _ms(getattr(turn_metrics, "tts_first_chunk_ms", None))),
        ("Turn total", _ms(getattr(turn_metrics, "total_turn_ms", None))),
    ]
    lines = ["| Stage | ms |", "| --- | ---: |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    notes = []
    if getattr(turn_metrics, "first_token_is_prefill_proxy", False):
        notes.append("first-token time is a prefill proxy (backend does not stream)")
    if getattr(asr_metrics, "hit_token_budget", False):
        notes.append("**transcript was cut off at the token budget**")
    if getattr(asr_metrics, "language_probability", None) is not None:
        notes.append(f"language detected at p={asr_metrics.language_probability:.2f}")
    if notes:
        lines.append("")
        lines += [f"- {n}" for n in notes]
    return "\n".join(lines)


def history_markdown(conversation) -> str:
    if conversation is None:
        return "_memory disabled_"
    snap = conversation.snapshot()
    if not snap["exchanges"]:
        return "_no turns yet_"
    lines = [f"**{snap['turns']} turn(s), {snap['history_tokens']} tokens in prompt**"
             + (f" · {snap['dropped']} dropped" if snap["dropped"] else ""), ""]
    for i, ex in enumerate(snap["exchanges"], 1):
        mark = " _(interrupted)_" if ex["interrupted"] else ""
        lines.append(f"{i}. **You:** {ex['user']}")
        lines.append(f"   **Agent:**{mark} {ex['assistant']}")
    return "\n".join(lines)


def stream_status(updates, session_state: str) -> str:
    """One-line status for the streaming tab."""
    kinds = {}
    for u in updates:
        kinds[u.kind.value] = kinds.get(u.kind.value, 0) + 1
    parts = [f"state: {session_state}"]
    parts += [f"{k}: {v}" for k, v in sorted(kinds.items())]
    return " · ".join(parts)


# --------------------------------------------------------------------------
# The agent behind the UI
# --------------------------------------------------------------------------


@dataclass
class Config:
    """Read from the environment when the demo starts, not when this module is
    imported — a notebook that sets the variables after ``import`` (the usual
    order) must still be honoured, so these are default factories."""

    whisper_model: str = field(
        default_factory=lambda: os.getenv("WHISPER_MODEL", DEFAULT_WHISPER))
    adapter: str | None = field(
        default_factory=lambda: os.getenv("WHISPER_ADAPTER_PATH") or None)
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", DEFAULT_LLM))
    tts_backend: str = field(default_factory=lambda: os.getenv("TTS_BACKEND", "edge"))
    device: str | None = field(default_factory=lambda: os.getenv("DEVICE") or None)
    #: Engine selection is explicit (env-driven here) rather than "use the fast
    #: one if it is installed" — an engine that changes underneath you makes
    #: every recorded latency incomparable to the last.
    asr_engine: str = field(default_factory=lambda: os.getenv("ASR_ENGINE", "explicit"))
    ct2_model: str | None = field(default_factory=lambda: os.getenv("CT2_MODEL") or None)
    ct2_compute_type: str = field(
        default_factory=lambda: os.getenv("CT2_COMPUTE_TYPE", "int8_float16"))
    llm_engine: str = field(default_factory=lambda: os.getenv("LLM_ENGINE", "explicit"))
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    llm_api_key: str | None = field(default_factory=lambda: os.getenv("LLM_API_KEY") or None)


class DemoAgent:
    """Loads models once, then serves turns. Everything is lazy so importing
    this module (and running the pure-function tests) costs nothing."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.asr = None
        self.llm_runner = None
        self.synth = None
        self.conversation = None
        self.info = {}

    # -- loading ----------------------------------------------------------

    def load(self):
        if self.asr is not None:
            return self
        from agent import Conversation
        from llm.engines import build_asr, build_llm

        self.asr = build_asr(
            self.config.asr_engine, model=self.config.whisper_model,
            adapter=self.config.adapter, device=self.config.device,
            ct2_model=self.config.ct2_model, ct2_compute_type=self.config.ct2_compute_type,
            language_candidates=LANGUAGES,
        )
        self.llm_runner, tokenizer, llm_info = build_llm(
            self.config.llm_engine, model=self.config.llm_model, device=self.config.device,
            base_url=self.config.llm_base_url, api_key=self.config.llm_api_key,
        )
        if self.config.tts_backend == "mms":
            from tts.local import MmsTtsSynthesizer

            self.synth = MmsTtsSynthesizer("hi")
            self.synth.warm_up()
        else:
            from tts import EdgeStreamingSynthesizer

            self.synth = EdgeStreamingSynthesizer(language="hi")
        self.conversation = Conversation(max_turns=6, tokenizer=tokenizer)
        self.info = {
            "whisper": self.config.whisper_model,
            "adapter": self.config.adapter or "none (base model)",
            "asr_engine": self.config.asr_engine,
            "llm": self.config.llm_model,
            "tts": self.config.tts_backend,
            **llm_info,
        }
        return self

    def status(self) -> str:
        if not self.info:
            return "_models not loaded yet — submit a turn to load them_"
        return " · ".join(f"**{k}**: {v}" for k, v in self.info.items())

    # -- turns ------------------------------------------------------------

    def run_turn(self, audio, language: str, max_tokens: int, speak: bool):
        """Gradio handler: audio in → transcript, answer, table, audio, history."""
        if audio is None:
            return "", "", "_record something first_", None, history_markdown(self.conversation)
        try:
            self.load()
        except Exception as exc:  # noqa: BLE001 - must reach the browser
            return ("", "", f"**model load failed**\n\n`{type(exc).__name__}: {exc}`",
                    None, history_markdown(self.conversation))
        from agent.audio import DecodingBufferSink
        from agent.turn import VoiceTurn

        wave, sample_rate = to_float_mono(audio)
        asr = self.asr.transcribe_array(
            wave, sample_rate, language=None if language == "auto" else language,
        )
        if not asr.text.strip():
            return ("", "", "_no speech recognised_", None,
                    history_markdown(self.conversation))

        sink = DecodingBufferSink(self.synth.format)
        turn = VoiceTurn(self.llm_runner, self.synth, response_language="Hindi",
                         llm_max_tokens=max_tokens, conversation=self.conversation,
                         split_into_sentences=True)
        result = turn.run(asr.text, speech_end_to_transcript_ms=asr.metrics.total_ms,
                          sink=sink if speak else None)

        audio_out = None
        if speak and sink.audio.size:
            audio_out = (self.synth.format.sample_rate, sink.audio)
        return (asr.text, result.response or f"_{result.state.value}: {result.error or ''}_",
                latency_table(asr.metrics, result.metrics), audio_out,
                history_markdown(self.conversation))

    def reset(self):
        if self.conversation is not None:
            self.conversation.reset()
        return "", "", "", None, history_markdown(self.conversation)

    # -- streaming tab ----------------------------------------------------

    def new_session(self):
        self.load()
        from asr.streaming import StreamingConfig, StreamingSession

        return StreamingSession(self.asr, StreamingConfig(language="hi", emit_partials=True))

    def push_chunk(self, audio, session):
        """Feed one microphone chunk; return (session, live text, status)."""
        from asr.streaming import UpdateKind

        if audio is None:
            return session, "", "waiting for audio"
        if session is None:
            try:
                session = self.new_session()
            except Exception as exc:  # noqa: BLE001 - must reach the browser
                return None, "", f"**model load failed**: `{type(exc).__name__}: {exc}`"
        wave, sample_rate = to_float_mono(audio)
        if sample_rate != 16_000:
            from asr.explicit.mel import load_audio_from_array

            wave, _ = load_audio_from_array(wave, sample_rate)
        updates = session.push(wave)
        text = ""
        for u in updates:
            if u.kind in (UpdateKind.PARTIAL, UpdateKind.CANDIDATE, UpdateKind.FINAL):
                text = u.text
        return session, text, stream_status(updates, session.state.value)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------


def launch_kwargs(**overrides) -> dict:
    """Launch settings that work in a sandboxed notebook.

    ``ssr_mode=False`` is the important one: Gradio 5+ server-side rendering
    starts a Node subprocess, and where that is blocked (hosted notebooks,
    restricted containers) the page becomes unreachable even though the
    Python server is listening. Everything else here is just explicit.
    """
    kwargs = {"ssr_mode": False, "server_name": "0.0.0.0", "server_port": 7860,
              "share": False, "quiet": False}
    kwargs.update(overrides)
    return kwargs


def diagnose(port: int = 7860, share: bool = True) -> dict:
    """Launch, then report what is actually reachable. Returns the findings.

    Prints the three routes in order of reliability so a dead URL can be
    told apart from a dead app: the page fetched from inside the process,
    the share tunnel, and (in Colab) the kernel proxy.
    """
    import urllib.request

    ui = build_ui(DemoAgent())
    _, local_url, share_url = ui.launch(**launch_kwargs(
        server_port=port, share=share, prevent_thread_lock=True, quiet=True,
    ))
    findings = {"local_url": local_url, "share_url": share_url, "local_status": None,
                "local_bytes": 0, "error": None}
    try:
        response = urllib.request.urlopen(local_url, timeout=10)
        body = response.read(4000)
        findings["local_status"] = response.status
        findings["local_bytes"] = len(body)
    except Exception as exc:  # noqa: BLE001
        findings["error"] = f"{type(exc).__name__}: {exc}"

    print(f"server listening: {findings['local_status'] == 200} "
          f"({findings['local_status']}, {findings['local_bytes']} bytes)")
    print(f"share tunnel: {share_url or 'UNAVAILABLE'}")
    try:
        from google.colab import output  # noqa: F401

        print(f"colab: run  from google.colab import output; "
              f"output.serve_kernel_port_as_iframe({port}, height=900)")
    except ImportError:
        print("colab: not running in Colab")
    if findings["local_status"] != 200:
        print("The app itself is not serving — this is not a URL problem. "
              f"error: {findings['error']}")
    findings["ui"] = ui
    return findings


def build_ui(agent: DemoAgent):
    import gradio as gr

    with gr.Blocks(title="Indic Voice Pipeline") as ui:
        gr.Markdown(
            "# Indic Voice Pipeline\n"
            "Hindi speech → Whisper (explicit loop, LoRA v2) → LLM → TTS, with "
            "dialogue memory. Every latency below is measured, not estimated; "
            "see `docs/STREAMING.md` for what each segment includes."
        )
        status = gr.Markdown(agent.status())

        with gr.Tab("Turn"):
            with gr.Row():
                with gr.Column():
                    audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy",
                                        label="Speak Hindi")
                    language = gr.Dropdown(["hi", "auto", "en", "te"], value="hi",
                                           label="Language ('auto' detects)")
                    tokens = gr.Slider(16, 256, 96, step=16, label="Max LLM tokens")
                    speak = gr.Checkbox(True, label="Speak the answer")
                    with gr.Row():
                        run = gr.Button("Run turn", variant="primary")
                        clear = gr.Button("Reset conversation")
                with gr.Column():
                    transcript = gr.Textbox(label="Transcript (what it heard)")
                    answer = gr.Textbox(label="Answer", lines=3)
                    audio_out = gr.Audio(label="Spoken answer", type="numpy", autoplay=True)
            with gr.Row():
                table = gr.Markdown(label="Latency")
                history = gr.Markdown(history_markdown(None))

            outputs = [transcript, answer, table, audio_out, history]
            run.click(agent.run_turn, [audio_in, language, tokens, speak], outputs).then(
                lambda: agent.status(), None, status)
            clear.click(agent.reset, None, outputs)

        with gr.Tab("Stream"):
            gr.Markdown(
                "Partial transcripts while you talk. The endpointer decides when "
                "you stopped; a `candidate` update is a full decode taken during "
                "the silence wait, which becomes the final for free."
            )
            session = gr.State(None)
            stream_in = gr.Audio(sources=["microphone"], streaming=True, type="numpy",
                                 label="Microphone (live)")
            live_text = gr.Textbox(label="Live transcript", lines=2)
            stream_info = gr.Markdown()
            stream_in.stream(agent.push_chunk, [stream_in, session],
                             [session, live_text, stream_info])

    return ui


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preload", action="store_true",
                        help="Load the models now rather than on the first turn")
    parser.add_argument("--port", type=int, default=int(os.getenv("GRADIO_PORT", "7860")))
    share_default = os.getenv("GRADIO_SHARE", "1") == "1"
    parser.add_argument("--share", dest="share", action="store_true", default=share_default)
    parser.add_argument("--no-share", dest="share", action="store_false")
    args = parser.parse_args(argv)

    agent = DemoAgent()
    print(f"config: {agent.config}")
    if args.preload:
        print("preloading models (first run downloads several GB)...", flush=True)
        agent.load()
        print(f"ready: {agent.status()}", flush=True)

    ui = build_ui(agent)
    _, local_url, share_url = ui.launch(**launch_kwargs(
        share=args.share, server_port=args.port, prevent_thread_lock=True,
    ))
    print(f"local:  {local_url}")
    if args.share:
        # launch() returns None for share_url when the tunnel binary could
        # not be fetched or the host blocks it — say so, rather than leaving
        # a printed URL that never resolves.
        print(f"public: {share_url}" if share_url else
              "public: UNAVAILABLE — the gradio.live tunnel did not come up "
              "(blocked host or frpc download failed). Use the local URL, or on "
              "Colab: from google.colab import output; "
              f"output.serve_kernel_port_as_window({args.port})")
    try:
        ui.block_thread()
    except KeyboardInterrupt:
        print("stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
