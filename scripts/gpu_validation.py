"""One-shot GPU validation of everything built after the evaluation harness.

Training and ``benchmarks.asr_eval`` have real GPU evidence under
``results/eval/``. Everything downstream — the served explicit runner with the
merged medium adapter, language detection, the streaming session, the LLM
prompt/decoding fixes, the pipeline waterfall, the agent turn with real TTS and
barge-in — had only run against fakes. This script runs each of those against
real models on a real GPU and writes a single JSON report with a pass/fail per
check, so the gap between "logic-complete" and "validated" is closed with
evidence rather than a claim.

Usage (Kaggle/Colab, T4 or better, internet enabled)::

    python scripts/gpu_validation.py \\
        --adapter Hugme6969/whisper-medium-hindi-lora \\
        --out-dir results/gpu_validation

Every check is independent: a failure is recorded and the next check runs.
The exit code is non-zero if any non-skipped check failed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DEVANAGARI = range(0x0900, 0x0980)


@dataclass
class Check:
    name: str
    status: str = "pending"        # pass | warn | fail | skip
    detail: dict = field(default_factory=dict)
    error: str | None = None
    seconds: float = 0.0


class Warn(Exception):
    """Raise from a check to record a finding that is not a failure."""


class Report:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.checks: list[Check] = []
        self.context: dict = {}

    def run(self, name: str, fn, *, skip_if: str | None = None):
        check = Check(name)
        start = time.perf_counter()
        if skip_if:
            check.status, check.error = "skip", skip_if
        else:
            try:
                check.detail = fn() or {}
                check.status = "pass"
            except Warn as exc:
                check.status = "warn"
                check.error = str(exc)
                check.detail = dict(getattr(exc, "detail", {}) or {})
            except Exception as exc:  # noqa: BLE001 - recorded, run continues
                check.status = "fail"
                check.error = f"{type(exc).__name__}: {exc}"
                check.detail["traceback"] = traceback.format_exc()
        check.seconds = time.perf_counter() - start
        self.checks.append(check)
        marker = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}[check.status]
        print(f"[{marker}] {name} ({check.seconds:.1f}s)" + (f" — {check.error}" if check.error else ""))
        self.save()
        return check

    def save(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"context": self.context, "checks": [asdict(c) for c in self.checks]}
        (self.out_dir / "report.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    def markdown(self) -> str:
        lines = ["| Check | Status | Key numbers |", "| --- | --- | --- |"]
        for c in self.checks:
            keys = {k: v for k, v in c.detail.items()
                    if isinstance(v, (int, float, str, bool)) and k != "traceback"}
            summary = ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in list(keys.items())[:5])
            lines.append(f"| {c.name} | {c.status} | {summary or c.error or ''} |")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def has_devanagari(text: str) -> bool:
    return any(ord(ch) in DEVANAGARI for ch in text)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER in percent at the ``standard`` level — the ledger's reporting policy."""
    from benchmarks.metrics import score_text

    rate = score_text(reference, hypothesis, level="standard").error_rate
    return 0.0 if rate is None else float(rate) * 100


def fleurs_clips(config: str, split: str, indices: list[int], out_dir: Path):
    """Pull specific FLEURS rows to 16 kHz WAV files; returns (path, ref, seconds)."""
    import soundfile as sf

    from asr.explicit.mel import load_audio_from_array
    from benchmarks.fleurs import extract_audio, load_fleurs

    ds = load_fleurs(config, split)
    clips = []
    for idx in indices:
        row = ds[idx]
        waveform, sr = extract_audio(row)
        waveform, seconds = load_audio_from_array(np.asarray(waveform, dtype=np.float32), sr)
        path = out_dir / f"{config}_{split}_{idx}.wav"
        sf.write(path, waveform, 16_000)
        clips.append((path, row["transcription"], seconds))
    return clips


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_environment(report: Report):
    import torch
    import transformers

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    info = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "git_commit": sha,
        "python": sys.version.split()[0],
    }
    report.context.update(info)
    return info


def check_gpu_unit_tests():
    """The repo's own GPU tests: explicit loop vs generate(), LLM, pipeline."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_asr.py", "tests/test_llm.py",
         "tests/test_pipeline.py", "-q", "-rfE", "--tb=short", "-p", "no:cacheprovider",
         "-W", "ignore"],
        capture_output=True, text=True,
    )
    lines = proc.stdout.strip().splitlines()
    summary = lines[-1] if lines else proc.stderr[-500:]
    failures = [ln for ln in lines if ln.startswith(("FAILED", "ERROR"))]
    if proc.returncode != 0:
        # The short-summary lines carry the assertion; the traceback block is
        # kept for the report but not for the one-line error.
        tb_start = next((i for i, ln in enumerate(lines) if ln.startswith("=") and "FAILURES" in ln), 0)
        raise RuntimeError(
            f"pytest exit {proc.returncode}: {summary} | " + " ; ".join(failures)
            + "\n" + "\n".join(lines[tb_start:tb_start + 60])
        )
    return {"summary": summary}


def check_served_model_matches_generate(runner, loaded, clips):
    """Explicit loop vs HF generate() on the *merged medium adapter*.

    tests/test_asr.py proves this for whisper-small on a sine wave; this is
    the model and audio actually being served.
    """
    import torch

    from asr.explicit.mel import load_audio

    per_clip = []
    for path, _ref, _sec in clips:
        waveform, _ = load_audio(str(path))
        explicit = runner.transcribe_array(waveform, 16_000, language="hi")
        features = loaded.processor.feature_extractor(
            waveform, sampling_rate=16_000, return_tensors="pt",
        ).input_features.to(device=loaded.device, dtype=loaded.dtype)
        with torch.inference_mode():
            ref_ids = loaded.model.generate(
                features, max_new_tokens=runner.token_budget("hi", None),
                language="hi", task="transcribe",
                do_sample=False, num_beams=1,
            )[0].tolist()
        # Same stripping on both sides: transformers 5.x generate() omits the
        # decoder prompt, older versions include it; the explicit loop keeps EOS.
        ref_out = runner.decoder.strip_generate_output(ref_ids)
        ours = runner.decoder.strip_generate_output(explicit.token_ids)
        n = min(len(ours), len(ref_out))
        matches = sum(a == b for a, b in zip(ours[:n], ref_out[:n], strict=True))
        per_clip.append({
            "clip": path.name, "explicit_tokens": len(ours),
            "generate_tokens": len(ref_out), "prefix_match": matches / n if n else 1.0,
            "exact": ours == ref_out,
            "explicit_text": explicit.text,
            "generate_text": loaded.processor.tokenizer.decode(ref_out, skip_special_tokens=True),
        })
    worst = min(c["prefix_match"] for c in per_clip)
    if worst < 0.9:
        raise AssertionError(f"explicit loop diverges from generate(): {per_clip}")
    return {"clips": len(per_clip), "worst_prefix_match": worst,
            "all_exact": all(c["exact"] for c in per_clip), "per_clip": per_clip}


def check_asr_quality(runner, clips):
    rows = []
    for path, ref, sec in clips:
        result = runner.transcribe_file(str(path), language="hi")
        rows.append({
            "clip": path.name, "seconds": round(sec, 2),
            "wer": word_error_rate(ref, result.text),
            "rtf": result.metrics.real_time_factor,
            "total_ms": result.metrics.total_ms,
            "hypothesis": result.text, "reference": ref,
        })
    mean_wer = float(np.mean([r["wer"] for r in rows]))
    if mean_wer > 60:
        raise AssertionError(f"mean WER {mean_wer:.1f}% on FLEURS clips — adapter not effective?")
    return {"mean_wer": mean_wer, "mean_rtf": float(np.mean([r["rtf"] for r in rows])), "rows": rows}


def _detect_rows(runner, clips):
    rows = []
    for path, _ref, _sec in clips:
        expected = "hi" if path.name.startswith("hi_in") else "en"
        result = runner.transcribe_file(str(path), language=None)
        rows.append({
            "clip": path.name, "expected": expected, "detected": result.language,
            "probability": result.metrics.language_probability,
            "detection_ms": result.metrics.language_detection_ms,
            "text": result.text,
        })
    return rows


def _detect_summary(rows):
    return {"clips": len(rows),
            "mean_detection_ms": float(np.mean([r["detection_ms"] for r in rows])),
            "min_probability": min(r["probability"] for r in rows),
            "wrong": [f"{r['clip']}→{r['detected']}" for r in rows if r["detected"] != r["expected"]],
            "rows": rows}


def check_language_detection_base(whisper_model: str, clips):
    """Detection code validated on the *base* model, which is what the
    procedure (one step on <|sot|>, argmax over language tokens) assumes."""
    import torch

    from asr.explicit import ASRRunner, load_whisper

    loaded = load_whisper(whisper_model)
    runner = ASRRunner(loaded.model, loaded.processor, loaded.device, loaded.dtype)
    try:
        rows = _detect_rows(runner, clips)
    finally:
        del runner, loaded
        torch.cuda.empty_cache()
    summary = _detect_summary(rows)
    # Hindi and Urdu share phonology; base Whisper flips between them on
    # short clips. The check validates the *mechanism* (a valid code at high
    # confidence), so a hi<->ur flip on the base model is a warning, not a
    # failure. Anything else is a real error.
    confusable = {"hi", "ur"}
    hard = [w for w in summary["wrong"] if w.split("\u2192")[1] not in confusable]
    if hard:
        raise AssertionError(f"base model misdetected: {hard}")
    if summary["wrong"]:
        warn = Warn(f"base model flips hi<->ur on {summary['wrong']} "
                    "(expected; the adapter is checked separately)")
        warn.detail = summary
        raise warn
    return summary


def check_language_detection_adapter(runner, clips):
    """Same check on the served adapter, unrestricted and restricted.

    An adapter trained with labels that lack the language token distorts the
    post-<|sot|> distribution, so unrestricted detection can fail while
    transcription is fine. That is recorded as a warning with the numbers,
    not hidden; the restricted form is what the demo uses.
    """
    unrestricted = _detect_summary(_detect_rows(runner, clips))
    saved = runner.language_candidates
    runner.language_candidates = ["hi", "en", "te"]
    try:
        restricted = _detect_summary(_detect_rows(runner, clips))
    finally:
        runner.language_candidates = saved
    detail = {
        "unrestricted_wrong": len(unrestricted["wrong"]),
        "restricted_wrong": len(restricted["wrong"]),
        "unrestricted": unrestricted, "restricted_hi_en_te": restricted,
    }
    if restricted["wrong"]:
        raise AssertionError(f"adapter misdetects even among hi/en/te: {restricted['wrong']}")
    if unrestricted["wrong"]:
        warn = Warn(f"adapter skews unrestricted detection: {unrestricted['wrong']} "
                    "(training labels lacked <|lang|>; see EXPERIMENTS.md)")
        warn.detail = detail
        raise warn
    return detail


def check_long_form(runner, clips):
    """Concatenate clips past 30 s and confirm chunk-and-stitch keeps the words."""
    from asr.explicit.mel import load_audio

    parts, refs = [], []
    for path, ref, _sec in clips:
        wav, _ = load_audio(str(path))
        parts.append(wav)
        parts.append(np.zeros(8_000, dtype=np.float32))  # 0.5 s gap
        refs.append(ref)
    audio = np.concatenate(parts)
    if len(audio) / 16_000 < 31:
        raise AssertionError("need > 30 s of audio for a long-form check; add clips")
    result = runner.transcribe_long_array(audio, 16_000, language="hi")
    wer = word_error_rate(" ".join(refs), result.text)
    return {"audio_seconds": round(len(audio) / 16_000, 1), "chunks": result.metrics.chunk_count,
            "wer_vs_concatenated_refs": wer, "rtf": result.metrics.real_time_factor,
            "text": result.text}


def check_streaming_session(runner, clips):
    """Real Whisper behind StreamingSession, microphone-sized blocks, audio clock.

    Reports three numbers so a bad result is attributable: WER against the
    references (model + VAD), WER against the *offline* transcript of the
    same audio (VAD/segmentation penalty only), and how much of each clip the
    endpointer actually covered (audio deleted before Whisper saw it).
    """
    from asr.explicit.mel import load_audio
    from asr.streaming import StreamingConfig, StreamingSession, UpdateKind
    from asr.vad import detect_speech

    class AudioClock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = AudioClock()
    config = StreamingConfig(language="hi", partial_interval_ms=1000, min_partial_audio_ms=1200)
    session = StreamingSession(runner, config, clock=clock)

    gap = 1.0
    parts, refs, offline, spans = [], [], [], []
    cursor = 0.0
    for path, ref, _sec in clips[:2]:
        wav, _ = load_audio(str(path))
        seconds = len(wav) / 16_000
        spans.append((path.name, cursor, cursor + seconds))
        cursor += seconds + gap
        parts += [wav, np.zeros(int(gap * 16_000), dtype=np.float32)]
        refs.append(ref)
        offline.append(runner.transcribe_array(wav, 16_000, language="hi").text)
    audio = np.concatenate(parts)

    block = 1_600  # 100 ms
    updates = []
    for start in range(0, len(audio), block):
        chunk = audio[start:start + block]
        clock.now += len(chunk) / 16_000
        updates.extend(session.push(chunk))
    updates.extend(session.flush())

    finals = [u for u in updates if u.kind == UpdateKind.FINAL]
    partials = [u for u in updates if u.kind == UpdateKind.PARTIAL]
    if not finals:
        raise AssertionError("streaming produced no final transcript")
    if not all(f.text.strip() for f in finals):
        raise AssertionError(f"empty final: {[f.as_dict() for f in finals]}")

    final_spans = [(f.utterance_start_seconds, f.utterance_start_seconds + f.audio_seconds)
                   for f in finals]

    def overlap(a0, a1):
        return sum(max(0.0, min(a1, f1) - max(a0, f0)) for f0, f1 in final_spans)

    # Informational: fraction of each *clip* inside a final. Clips carry their
    # own leading/trailing silence, so this cannot reach 1.0.
    coverage = [{"clip": name, "covered_fraction": round(overlap(c0, c1) / (c1 - c0), 3)}
                for name, c0, c1 in spans]

    # The real test: the online endpointer against the offline detector on the
    # same audio with the same rules. They should agree almost exactly.
    offline_segments = detect_speech(audio, 16_000, config.vad)
    offline_speech = sum(s.duration_seconds for s in offline_segments)
    agreed = sum(overlap(s.start_seconds, s.end_seconds) for s in offline_segments)
    online_vs_offline_vad = round(agreed / offline_speech, 3) if offline_speech else None

    joined = " ".join(f.text for f in finals)
    detail = {
        "utterances_expected": len(refs), "finals": len(finals), "partials": len(partials),
        "wer_vs_refs": word_error_rate(" ".join(refs), joined),
        "wer_vs_offline": word_error_rate(" ".join(offline), joined),
        "online_vs_offline_vad": online_vs_offline_vad,
        "offline_vad_segments": len(offline_segments),
        "min_clip_coverage": min(c["covered_fraction"] for c in coverage),
        "final_asr_ms_mean": float(np.mean([f.asr_ms for f in finals])),
        "asr_after_endpoint_ms_mean": float(np.mean([f.asr_ms_after_endpoint for f in finals])),
        "finals_from_candidate": sum(1 for f in finals if f.from_candidate),
        "candidates": sum(1 for u in updates if u.kind == UpdateKind.CANDIDATE),
        "endpoint_reasons": [f.endpoint_reason.value for f in finals],
        "finals_detail": [
            {"start": round(f.utterance_start_seconds, 2), "seconds": round(f.audio_seconds, 2),
             "reason": f.endpoint_reason.value, "from_candidate": f.from_candidate,
             "asr_ms_after_endpoint": round(f.asr_ms_after_endpoint, 1), "text": f.text}
            for f in finals
        ],
        "offline_vad_detail": [
            {"start": round(s.start_seconds, 2), "end": round(s.end_seconds, 2)}
            for s in offline_segments
        ],
        "coverage": coverage,
        "offline_hypotheses": offline,
        "updates": [u.as_dict() for u in updates],
    }
    if online_vs_offline_vad is not None and online_vs_offline_vad < 0.9:
        raise AssertionError(
            f"online endpointer covers only {online_vs_offline_vad:.0%} of offline-VAD speech; "
            f"finals={detail['finals_detail']} offline={detail['offline_vad_detail']} "
            f"wer_vs_offline={detail['wer_vs_offline']:.1f}%"
        )
    return detail


def check_llm_prompt_and_decoding(pipe, transcript: str):
    """Thinking mode off, Hindi answer, no repetition loop — on the real model."""
    prompt = pipe._build_prompt(transcript, "hi")
    result = pipe.llm_runner.generate(prompt, max_new_tokens=96)
    problems = []
    if "<think>" in result.text:
        problems.append("<think> block in output — thinking mode not disabled")
    if not has_devanagari(result.text):
        problems.append("no Devanagari in response — model not answering in Hindi")
    if result.metrics.stopped_on_repetition:
        problems.append("loop guard fired despite repetition penalty")
    if problems:
        raise AssertionError("; ".join(problems) + f" | response={result.text!r}")
    return {
        "prompt_has_empty_think": "</think>" in prompt,
        "generated_tokens": result.metrics.generated_tokens,
        "prefill_ms": result.metrics.prefill_ms, "mean_decode_ms": result.metrics.mean_decode_ms,
        "response": result.text,
    }


def check_llm_compiled_matches_eager(pipe, transcript: str):
    """Static cache + CUDA-graph decode must produce the eager tokens, and
    should be faster per token. Reports both so the gain is measured."""
    from llm import LLMRunner

    prompt = pipe._build_prompt(transcript, "hi")
    eager = pipe.llm_runner
    fast = LLMRunner(pipe.llm.model, pipe.llm.tokenizer, pipe.llm.device,
                     repetition_penalty=eager.repetition_penalty,
                     static_cache=True, compile_decode=True, max_cache_len=1024)
    a = eager.generate(prompt, max_new_tokens=64)
    fast.generate(prompt, max_new_tokens=8)                  # compile / graph capture
    b = fast.generate(prompt, max_new_tokens=64)
    c = fast.generate(prompt, max_new_tokens=64)             # steady state
    detail = {
        "tokens_match": a.token_ids == b.token_ids == c.token_ids,
        "eager_ms_per_token": a.metrics.mean_decode_ms,
        "compiled_ms_per_token": c.metrics.mean_decode_ms,
        "speedup": (a.metrics.mean_decode_ms / c.metrics.mean_decode_ms
                    if c.metrics.mean_decode_ms else None),
        "eager_prefill_ms": a.metrics.prefill_ms, "compiled_prefill_ms": c.metrics.prefill_ms,
        "generated_tokens": a.metrics.generated_tokens,
        "eager_text": a.text, "compiled_text": c.text,
    }
    if not detail["tokens_match"]:
        raise AssertionError(f"compiled decode diverges from eager: {detail}")
    return detail


def check_ct2_matches_explicit(runner, clips, ct2_dir: str, compute_type: str):
    """The engine tier must reproduce the explicit runner's tokens on the
    same audio (fp16/fp32) or stay within a small WER of it (int8), and be
    faster. Both are reported."""
    from asr.engines.ct2 import CT2Transcriber
    from asr.explicit.mel import load_audio

    engine = CT2Transcriber(ct2_dir, device="cuda", compute_type=compute_type)
    rows = []
    for path, _ref, _sec in clips:
        wav, _ = load_audio(str(path))
        ex = runner.transcribe_array(wav, 16_000, language="hi")
        ct = engine.transcribe_array(wav, 16_000, language="hi")
        ex_tokens = runner.decoder.strip_generate_output(ex.token_ids)
        rows.append({
            "clip": path.name, "tokens_equal": ex_tokens == list(ct.token_ids),
            "wer_ct2_vs_explicit": word_error_rate(ex.text, ct.text),
            "explicit_ms": ex.metrics.total_ms, "ct2_ms": ct.metrics.total_ms,
            "explicit_text": ex.text, "ct2_text": ct.text,
        })
    exact = sum(1 for r in rows if r["tokens_equal"])
    mean_wer = float(np.mean([r["wer_ct2_vs_explicit"] for r in rows]))
    speedup = float(np.mean([r["explicit_ms"] / r["ct2_ms"] for r in rows if r["ct2_ms"]]))
    detail = {"clips": len(rows), "tokens_equal": exact, "mean_wer_vs_explicit": mean_wer,
              "speedup": speedup, "compute_type": compute_type, "rows": rows}
    quantized = compute_type.startswith("int8")
    if (not quantized and exact != len(rows)) or (quantized and mean_wer > 5.0):
        raise AssertionError(f"engine diverges from the explicit runner: {detail}")
    return detail


def check_pipeline_waterfall(pipe, clips):
    import torch

    path = str(clips[0][0])
    pipe.run(path, language="hi", llm_max_tokens=48)  # warm-up
    runs = [pipe.run(path, language="hi", llm_max_tokens=48) for _ in range(3)]

    def avg(fn):
        return float(np.mean([fn(r.metrics) for r in runs]))

    detail = {
        "strategy": pipe.strategy.value,
        "mel_ms": avg(lambda m: m.asr.mel_extraction_ms),
        "encoder_ms": avg(lambda m: m.asr.encoder_ms),
        "asr_decode_ms": avg(lambda m: m.asr.total_decode_ms),
        "llm_prefill_ms": avg(lambda m: m.llm_prefill_ms),
        "llm_decode_ms": avg(lambda m: sum(m.llm_decode_ms)),
        "total_ms": avg(lambda m: m.total_pipeline_ms),
        "audio_to_first_llm_token_ms": avg(lambda m: m.audio_to_first_llm_token_ms),
        "peak_vram_gib": max(r.metrics.peak_allocated_bytes for r in runs) / 2**30,
        "transcript": runs[0].transcript, "answer": runs[0].answer,
    }
    if "<think>" in runs[0].answer:
        raise AssertionError("pipeline answer contains <think>")
    torch.cuda.empty_cache()
    return detail


def check_voice_turn(llm_runner, transcript: str):
    """LLM → sentence-split edge-tts streaming → BufferSink, all real."""
    from agent import VoiceTurn
    from tts import EdgeStreamingSynthesizer

    turn = VoiceTurn(llm_runner, EdgeStreamingSynthesizer(language="hi"), response_language="Hindi")
    # No endpointer runs in this check, so speech-end → transcript was never
    # measured. Passing 0.0 made response_latency_ms a plausible small number
    # instead of None — which is how a cited "719 ms" turn latency came to
    # omit ~3.8 s of LLM decode. The turn's own rule: None, never zero.
    result = turn.run(transcript, speech_end_to_transcript_ms=None)
    if result.state.value != "completed":
        raise AssertionError(f"turn ended {result.state.value}: {result.error}")
    if result.speech and result.speech.error:
        raise AssertionError(f"TTS error: {result.speech.error}")
    if not result.speech or not result.speech.chunks:
        raise AssertionError("no audio chunks produced")
    m = result.metrics
    if not m.llm_streaming or m.first_token_is_prefill_proxy:
        raise AssertionError("LLMRunner.stream() was not used; first-token time is a proxy")
    return {
        "sentences": len(result.speech.sentences), "chunks": len(result.speech.chunks),
        "audio_bytes": result.speech.total_bytes,
        "tts_first_chunk_ms": m.tts_first_chunk_ms,
        "final_transcript_to_first_llm_token_ms": m.final_transcript_to_first_llm_token_ms,
        "first_llm_token_to_playback_start_ms": m.first_llm_token_to_playback_start_ms,
        "response_latency_ms": m.response_latency_ms,
        "llm_total_ms": m.llm_total_ms, "llm_generated_tokens": m.llm_generated_tokens,
        "total_turn_ms": m.total_turn_ms,
        "llm_streaming": m.llm_streaming,
        "response": result.response, "turn": result.as_dict(),
    }


def check_barge_in(llm_runner, transcript: str):
    """Cancel as soon as the first real audio chunk reaches the sink.

    With an in-memory sink, playback is only as slow as edge-tts delivers, so
    a short reply can finish within tens of milliseconds of its first chunk;
    a timer-based interrupt races that and loses. Cancelling from inside
    ``write`` is deterministic: the playback loop must observe the cancel
    before the next chunk. Thread-safety of ``cancel()`` is covered by the
    unit tests; what this proves is the lifecycle against real synthesis.
    """
    from agent import BufferSink, VoiceTurn
    from tts import EdgeStreamingSynthesizer

    turn = VoiceTurn(llm_runner, EdgeStreamingSynthesizer(language="hi"),
                     response_language="Hindi", llm_max_tokens=160)
    sink = BufferSink()
    original_write = sink.write
    fired = {"at_chunk": None}

    def write(chunk: bytes) -> None:
        original_write(chunk)
        if fired["at_chunk"] is None:
            fired["at_chunk"] = len(sink.chunks)
            turn.interrupt("validation_barge_in")

    sink.write = write  # type: ignore[method-assign]
    result = turn.run(transcript, sink=sink)

    if result.state.value != "interrupted":
        raise AssertionError(f"expected interrupted, got {result.state.value}: {result.error}")
    speech = result.speech
    return {
        "cancelled_after_chunk": fired["at_chunk"],
        "llm_stopped_by_barge_in": result.metrics.llm_stopped_by_barge_in,
        "llm_generated_tokens": result.metrics.llm_generated_tokens,
        "sentences_planned": len(speech.sentences) if speech else None,
        "chunks_synthesised": len(speech.chunks) if speech else None,
        "playback_state": result.playback.state.value if result.playback else None,
        "barge_in_flag": result.metrics.barge_in,
        "turn": result.as_dict(),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="Hugme6969/whisper-medium-hindi-lora")
    parser.add_argument("--whisper-model", default="openai/whisper-medium")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--out-dir", default="results/gpu_validation")
    parser.add_argument("--hi-indices", default="7,42,123,256,311",
                        help="FLEURS hi_in test rows to use as clips")
    parser.add_argument("--en-indices", default="5,77", help="FLEURS en_us test rows")
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument("--skip-network", action="store_true",
                        help="Skip edge-tts checks (voice turn, barge-in)")
    parser.add_argument("--ct2-model", default=None,
                        help="CTranslate2 model dir (scripts/convert_ct2.py) to validate against the runner")
    parser.add_argument("--ct2-compute-type", default="int8_float16")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    report = Report(out_dir)
    report.context["argv"] = sys.argv

    report.run("environment", lambda: check_environment(report))
    report.run("gpu_unit_tests", check_gpu_unit_tests,
               skip_if="--skip-unit-tests" if args.skip_unit_tests else None)

    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    hi_idx = [int(x) for x in args.hi_indices.split(",") if x]
    en_idx = [int(x) for x in args.en_indices.split(",") if x]
    state: dict = {}

    def fetch():
        state["hi"] = fleurs_clips("hi_in", "test", hi_idx, clips_dir)
        state["en"] = fleurs_clips("en_us", "test", en_idx, clips_dir)
        return {"hi_clips": len(state["hi"]), "en_clips": len(state["en"]),
                "hi_seconds": round(sum(c[2] for c in state["hi"]), 1)}

    if report.run("fetch_fleurs_clips", fetch).status != "pass":
        print("Cannot continue without audio.")
        return 1
    hi, en = state["hi"], state["en"]

    def load():
        from asr.explicit import ASRRunner, load_whisper

        loaded = load_whisper(args.whisper_model, adapter_path=args.adapter)
        state["whisper"] = loaded
        state["runner"] = ASRRunner(loaded.model, loaded.processor, loaded.device, loaded.dtype)
        return {"model": loaded.model_name, "adapter_dir": loaded.adapter_path}

    if report.run("load_whisper_with_hub_adapter", load).status != "pass":
        print("Cannot continue without the ASR model.")
        return 1
    runner, loaded = state["runner"], state["whisper"]

    report.run("served_model_matches_generate",
               lambda: check_served_model_matches_generate(runner, loaded, hi[:3]))
    report.run("asr_quality_on_clips", lambda: check_asr_quality(runner, hi))
    report.run("language_detection_base_model",
               lambda: check_language_detection_base(args.whisper_model, hi[:2] + en))
    report.run("language_detection_adapter",
               lambda: check_language_detection_adapter(runner, hi[:2] + en))
    report.run("long_form_chunking", lambda: check_long_form(runner, hi))
    report.run("streaming_session", lambda: check_streaming_session(runner, hi))
    report.run("ct2_matches_explicit",
               lambda: check_ct2_matches_explicit(runner, hi[:3], args.ct2_model, args.ct2_compute_type),
               skip_if=None if args.ct2_model else "--ct2-model not given")

    def load_llm_and_pipe():
        from llm import load_llm
        from pipeline import VoicePipeline

        llm = load_llm(args.llm_model)
        state["pipe"] = VoicePipeline(loaded, llm)
        return {"llm": llm.model_name, "strategy": state["pipe"].strategy.value}

    if report.run("load_llm", load_llm_and_pipe).status != "pass":
        report.save()
        print(report.markdown())
        return 1
    pipe = state["pipe"]
    transcript = hi[0][1]

    report.run("llm_prompt_and_decoding", lambda: check_llm_prompt_and_decoding(pipe, transcript))
    report.run("llm_compiled_matches_eager",
               lambda: check_llm_compiled_matches_eager(pipe, transcript))
    report.run("pipeline_waterfall", lambda: check_pipeline_waterfall(pipe, hi))
    net = "--skip-network" if args.skip_network else None
    report.run("voice_turn_real_tts", lambda: check_voice_turn(pipe.llm_runner, transcript),
               skip_if=net)
    report.run("barge_in_real_tts", lambda: check_barge_in(pipe.llm_runner, transcript),
               skip_if=net)

    report.save()
    (out_dir / "summary.md").write_text(report.markdown() + "\n", encoding="utf-8")
    print("\n" + report.markdown())
    print(f"\nreport: {out_dir / 'report.json'}")
    failed = report.failed
    print(f"\n{len(report.checks) - len(failed)}/{len(report.checks)} checks passed"
          + (f"; FAILED: {[c.name for c in failed]}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
