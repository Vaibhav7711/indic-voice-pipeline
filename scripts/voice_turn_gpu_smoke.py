"""Run one real Hindi turn: Whisper LoRA -> LLM -> streaming TTS.

Playback is captured by BufferSink, so this validates orchestration without
opening an operating-system audio device.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--whisper-model", default="openai/whisper-medium")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output", default="results/streaming/voice_turn_gpu_smoke.json")
    args = parser.parse_args()

    from agent import VoiceTurn
    from asr.explicit import ASRRunner, load_whisper
    from llm import LLMRunner, load_llm
    from tts import EdgeStreamingSynthesizer

    whisper = load_whisper(args.whisper_model, adapter_path=args.adapter)
    asr = ASRRunner(whisper.model, whisper.processor, whisper.device, whisper.dtype)
    transcript = asr.transcribe_file(args.audio, language="hi")

    llm = load_llm(args.llm_model)
    turn = VoiceTurn(
        LLMRunner(llm.model, llm.tokenizer, llm.device),
        EdgeStreamingSynthesizer(language="hi"),
        response_language="Hindi",
    )
    result = turn.run(transcript.text, speech_end_to_transcript_ms=transcript.metrics.total_ms)

    report = {
        "audio": str(Path(args.audio).resolve()),
        "transcript": transcript.text,
        "transcript_metrics": transcript.metrics.as_dict(),
        "turn": result.as_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"TRANSCRIPT: {transcript.text}")
    print(f"RESPONSE: {result.response}")
    print(f"TTS chunks: {result.speech.chunks if result.speech else 0}")
    print(f"saved: {output}")
    return 0 if result.state.value == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
