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
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper-model", default="openai/whisper-medium")
    parser.add_argument("--adapter", default="Hugme6969/whisper-medium-hindi-lora")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--llm-compile", action="store_true")
    parser.add_argument("--asr-engine", default="explicit", choices=["explicit", "ct2"],
                        help="ct2 serves from a converted CTranslate2 model "
                             "(scripts/convert_ct2.py); measured 1.31x, but ASR is "
                             "off the critical path so expect ~90 ms of a ~4 s turn")
    parser.add_argument("--ct2-model", default=None)
    parser.add_argument("--ct2-compute-type", default="int8_float16")
    parser.add_argument("--llm-engine", default="explicit", choices=["explicit", "http"],
                        help="http serves from an OpenAI-compatible endpoint "
                             "(vLLM, SGLang, llama.cpp server, a custom engine). "
                             "This is where the seconds are: the LLM was 3.3 s of a "
                             "4.4 s measured turn")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--tts", default="edge", choices=["edge", "mms"])
    parser.add_argument("--language", default="hi")
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--input-wav", default=None,
                        help="Replay this file as the microphone (real-time pacing), "
                             "then trailing silence. Deterministic end-to-end check.")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--whisper-dtype", default=None, choices=[None, "float16", "float32"])
    parser.add_argument("--output-device", default=None)
    parser.add_argument("--sink", default="device", choices=["device", "buffer"],
                        help="buffer: capture audio in memory instead of a sound device (headless GPU boxes)")
    parser.add_argument("--incremental-finals", action="store_true")
    parser.add_argument("--semantic-endpointing", action="store_true")
    parser.add_argument("--log", default="results/live/turns.jsonl")
    parser.add_argument("--max-turns", type=int, default=0, help="0 = until Ctrl-C")
    parser.add_argument("--history-turns", type=int, default=6,
                        help="Dialogue turns kept in the prompt; 0 disables memory")
    parser.add_argument("--history-tokens", type=int, default=800)
    args = parser.parse_args()

    sd = None
    if args.sink == "device" or not args.input_wav:
        import sounddevice as sd
    import torch

    from agent import Conversation, VoiceTurn
    from agent.audio import SoundDeviceSink
    from asr.streaming import StreamingConfig, StreamingSession, UpdateKind
    from llm.engines import build_asr, build_llm

    print("loading models…", flush=True)
    dtype = getattr(torch, args.whisper_dtype) if args.whisper_dtype else None
    asr = build_asr(args.asr_engine, model=args.whisper_model,
                    adapter=args.adapter or None, device=args.device, dtype=dtype,
                    ct2_model=args.ct2_model, ct2_compute_type=args.ct2_compute_type,
                    language_candidates=[args.language])
    llm_runner, llm_tokenizer, llm_info = build_llm(
        args.llm_engine, model=args.llm_model, device=args.device,
        static_cache=args.llm_compile, compile_decode=args.llm_compile,
        base_url=args.llm_base_url, api_key=args.llm_api_key,
    )
    print(f"asr engine: {args.asr_engine} | llm: {llm_info}")
    if args.tts == "edge":
        from tts import EdgeStreamingSynthesizer

        synth = EdgeStreamingSynthesizer(language=args.language)
    else:
        from tts.local import MmsTtsSynthesizer

        synth = MmsTtsSynthesizer(args.language)
        print(f"tts warm-up {synth.warm_up():.0f} ms")
    if args.llm_engine == "http":
        # Fail at startup, not mid-turn, if the endpoint is not serving.
        print("probing the llm endpoint…", flush=True)
        print(llm_runner.probe())
    elif args.llm_compile:
        llm_runner.generate("नमस्ते", max_new_tokens=4)          # graph capture off the first turn

    config = StreamingConfig(
        language=args.language, emit_partials=True,
        incremental_finals=args.incremental_finals,
        semantic_endpointing=args.semantic_endpointing,
    )
    session = StreamingSession(asr, config)
    if args.sink == "buffer":
        from agent.audio import DecodingBufferSink

        def sink_factory():
            return DecodingBufferSink(synth.format)
    else:
        def sink_factory():
            return SoundDeviceSink(synth.format, device=args.output_device)

    conversation = None
    if args.history_turns > 0:
        conversation = Conversation(max_turns=args.history_turns,
                                    max_history_tokens=args.history_tokens,
                                    tokenizer=llm_tokenizer)
    turn = VoiceTurn(llm_runner, synth, response_language="Hindi",
                     sink_factory=sink_factory, conversation=conversation)

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
            if conversation is not None:
                snap = conversation.snapshot()
                print(f"    history: {snap['turns']} turn(s), {snap['history_tokens']} tokens"
                      + (f", {snap['dropped']} dropped" if snap["dropped"] else ""))
            print(f"    response latency {m.response_latency_ms and round(m.response_latency_ms)} ms "
                  f"(asr {round(speech_end_to_final_ms)} | first token "
                  f"{m.final_transcript_to_first_llm_token_ms and round(m.final_transcript_to_first_llm_token_ms)} | "
                  f"to audio {m.first_llm_token_to_playback_start_ms and round(m.first_llm_token_to_playback_start_ms)})"
                  + ("  [barge-in]" if m.barge_in else ""))
        finally:
            turn_state["speaking"] = False

    import contextlib

    if args.input_wav:
        from asr.explicit.mel import load_audio

        def replay():
            wav, _ = load_audio(args.input_wav)
            wav = np.concatenate([np.zeros(8_000, np.float32), wav, np.zeros(16_000 * 3, np.float32)])
            for i in range(0, len(wav), 1600):
                audio_q.put(wav[i:i + 1600])
                time.sleep(0.1)                      # real-time pacing like a mic
            while True:                              # keep "listening" in silence
                audio_q.put(np.zeros(1600, np.float32))
                time.sleep(0.1)

        threading.Thread(target=replay, daemon=True).start()
        stream = contextlib.nullcontext()
        print(f"replaying {args.input_wav} as the microphone")
    else:
        stream = sd.InputStream(samplerate=16_000, channels=1, dtype="float32",
                                blocksize=1600, device=args.input_device, callback=on_audio)
    print(f"listening ({args.language}); speak, pause, and the agent answers. Ctrl-C to stop.")
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
