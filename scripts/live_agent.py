"""Live voice agent: microphone → streaming ASR → LLM → TTS → speakers.

Everything upstream of this file is unit-tested against fakes and validated
on a GPU with recorded audio; this is where it meets a real microphone and a
real speaker. Barge-in is real: the microphone keeps feeding the streaming
session while the agent speaks, and a new speech onset cancels playback,
synthesis and the LLM decode.

    python scripts/live_agent.py --adapter Hugme6969/whisper-medium-hindi-lora
    python scripts/live_agent.py --tts mms --llm-compile      # no network TTS, CUDA-graph decode

Needs a CUDA GPU for Whisper and the LLM, and ``pip install -e ".[audio]"``.
Every turn's metrics are appended to ``--log`` as JSONL — the four latencies,
ASR after endpoint, whether the final came from a candidate — so a session
is evidence, not an anecdote.

Threads
-------
* **capture** — ``sounddevice.InputStream`` callback → queue (never blocks).
* **main** — drains the queue into ``StreamingSession.push``; on a final,
  runs the turn on the **turn** thread; if a new ``speech_start`` arrives
  while a turn is speaking, calls ``turn.interrupt()``.

The session's ``push`` blocks while ASR runs (a candidate or a final). Audio
keeps accumulating in the queue meanwhile and is drained afterwards, so
nothing is lost; the endpoint decision is made on audio time, not wall time.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper-model", default="openai/whisper-medium")
    parser.add_argument("--adapter", default="Hugme6969/whisper-medium-hindi-lora")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--llm-compile", action="store_true")
    parser.add_argument("--tts", default="edge", choices=["edge", "mms"])
    parser.add_argument("--language", default="hi")
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output-device", default=None)
    parser.add_argument("--incremental-finals", action="store_true")
    parser.add_argument("--semantic-endpointing", action="store_true")
    parser.add_argument("--log", default="results/live/turns.jsonl")
    parser.add_argument("--max-turns", type=int, default=0, help="0 = until Ctrl-C")
    args = parser.parse_args()

    import sounddevice as sd

    from agent import VoiceTurn
    from agent.audio import SoundDeviceSink
    from asr.explicit import ASRRunner, load_whisper
    from asr.streaming import StreamingConfig, StreamingSession, UpdateKind
    from llm import LLMRunner, load_llm

    print("loading models…", flush=True)
    whisper = load_whisper(args.whisper_model, adapter_path=args.adapter)
    asr = ASRRunner(whisper.model, whisper.processor, whisper.device, whisper.dtype)
    llm = load_llm(args.llm_model)
    llm_runner = LLMRunner(llm.model, llm.tokenizer, llm.device,
                           static_cache=args.llm_compile, compile_decode=args.llm_compile,
                           max_cache_len=1024)
    if args.tts == "edge":
        from tts import EdgeStreamingSynthesizer

        synth = EdgeStreamingSynthesizer(language=args.language)
    else:
        from tts.local import MmsTtsSynthesizer

        synth = MmsTtsSynthesizer(args.language)
        print(f"tts warm-up {synth.warm_up():.0f} ms")
    if args.llm_compile:
        llm_runner.generate("नमस्ते", max_new_tokens=4)          # graph capture off the first turn

    config = StreamingConfig(
        language=args.language, emit_partials=True,
        incremental_finals=args.incremental_finals,
        semantic_endpointing=args.semantic_endpointing,
    )
    session = StreamingSession(asr, config)
    turn = VoiceTurn(llm_runner, synth, response_language="Hindi",
                     sink_factory=lambda: SoundDeviceSink(synth.format, device=args.output_device))

    audio_q: queue.Queue = queue.Queue()

    def on_audio(indata, frames, time_info, status):
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    turn_thread: threading.Thread | None = None
    turn_state = {"speaking": False, "result": None}
    turns_done = 0

    def run_turn(text: str, speech_end_to_final_ms: float):
        turn_state["speaking"] = True
        try:
            result = turn.run(text, speech_end_to_transcript_ms=speech_end_to_final_ms)
            turn_state["result"] = result
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")
            m = result.metrics
            print(f"  ↳ [{result.state.value}] {result.response}")
            print(f"    response latency {m.response_latency_ms and round(m.response_latency_ms)} ms "
                  f"(asr {round(speech_end_to_final_ms)} | first token "
                  f"{m.final_transcript_to_first_llm_token_ms and round(m.final_transcript_to_first_llm_token_ms)} | "
                  f"to audio {m.first_llm_token_to_playback_start_ms and round(m.first_llm_token_to_playback_start_ms)})"
                  + ("  [barge-in]" if m.barge_in else ""))
        finally:
            turn_state["speaking"] = False

    print(f"listening ({args.language}); speak, pause, and the agent answers. Ctrl-C to stop.")
    stream = sd.InputStream(samplerate=16_000, channels=1, dtype="float32",
                            blocksize=1600, device=args.input_device, callback=on_audio)
    with stream:
        try:
            while True:
                block = audio_q.get()
                updates = session.push(block)
                for u in updates:
                    if u.kind == UpdateKind.STATE and u.state.value == "speaking":
                        if turn_state["speaking"] and turn_thread and turn_thread.is_alive():
                            turn.interrupt("barge_in")
                            print("  [barge-in: user started speaking]")
                    elif u.kind in (UpdateKind.PARTIAL, UpdateKind.CANDIDATE):
                        print(f"  … {u.text}", end="\r", flush=True)
                    elif u.kind == UpdateKind.FINAL and u.text.strip():
                        # Speech end → final: the silence the endpointer waited
                        # plus the ASR that ran after it (0 for a candidate).
                        wait_ms = (u.stream_seconds - (u.utterance_start_seconds + u.audio_seconds)) * 1000 \
                            + config.vad.padding_ms
                        speech_end_to_final = max(0.0, wait_ms) + u.asr_ms_after_endpoint
                        print(f"\n▶ {u.text}   (final{' from candidate' if u.from_candidate else ''}, "
                              f"asr after endpoint {u.asr_ms_after_endpoint:.0f} ms)")
                        if turn_thread and turn_thread.is_alive():
                            turn.interrupt("new_utterance")
                            turn_thread.join(timeout=5)
                        turn_thread = threading.Thread(
                            target=run_turn, args=(u.text, speech_end_to_final), daemon=True,
                        )
                        turn_thread.start()
                        turns_done += 1
                        if args.max_turns and turns_done >= args.max_turns:
                            turn_thread.join()
                            return 0
        except KeyboardInterrupt:
            print("\nstopping")
            turn.interrupt("shutdown")
            for u in session.flush():
                if u.kind == UpdateKind.FINAL and u.text.strip():
                    print(f"▶ {u.text} (flushed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
