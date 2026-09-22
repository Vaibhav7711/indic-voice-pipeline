"""CTranslate2 engine vs the explicit runner, on CPU with whisper-small.

Runs only when the converted model exists locally (scripts/convert_ct2.py)
and the HF checkpoint plus MMS-TTS are cached — no downloads in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("faster_whisper")

CT2_DIR = Path("models/ct2/whisper-small")
CT2_FP32_DIR = Path("models/ct2/whisper-small-fp32")


def _cached(model_name: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(model_name, "config.json"), str)
    except Exception:  # noqa: BLE001
        return False


needs_models = pytest.mark.skipif(
    not (CT2_DIR.is_dir() and _cached("openai/whisper-small") and _cached("facebook/mms-tts-hin")),
    reason="converted whisper-small, HF whisper-small and MMS-hin must be local",
)


@pytest.fixture(scope="module")
def clips_and_runner():
    from asr.explicit import ASRRunner, load_whisper
    from asr.explicit.mel import load_audio_from_array
    from tts.local import MmsTtsSynthesizer

    tts = MmsTtsSynthesizer("hi", seed=0)
    clips = [load_audio_from_array(tts.synthesize(s), tts.format.sample_rate)[0]
             for s in ("भारत की राजधानी क्या है", "आज मौसम कैसा रहेगा")]
    loaded = load_whisper("openai/whisper-small", device="cpu")
    runner = ASRRunner(loaded.model, loaded.processor, loaded.device, loaded.dtype)
    return clips, runner, [runner.transcribe_array(w, 16_000, language="hi") for w in clips]


@needs_models
@pytest.mark.skipif(not CT2_FP32_DIR.is_dir(), reason="fp32 conversion not present")
def test_ct2_fp32_reproduces_explicit_runner_tokens_exactly(clips_and_runner):
    from asr.engines.ct2 import CT2Transcriber

    clips, runner, explicit = clips_and_runner
    engine = CT2Transcriber(CT2_FP32_DIR, device="cpu", compute_type="float32")
    for wav, ex in zip(clips, explicit, strict=True):
        ct = engine.transcribe_array(wav, 16_000, language="hi")
        assert runner.decoder.strip_generate_output(ex.token_ids) == list(ct.token_ids), \
            (ex.text, ct.text)


@needs_models
def test_ct2_int8_stays_close_to_explicit_runner_and_is_faster(clips_and_runner):
    from asr.engines.ct2 import CT2Transcriber
    from benchmarks.metrics import score_text

    clips, runner, explicit = clips_and_runner
    engine = CT2Transcriber(CT2_DIR, device="cpu", compute_type="int8")
    wers, speedups = [], []
    for wav, ex in zip(clips, explicit, strict=True):
        ct = engine.transcribe_array(wav, 16_000, language="hi")
        wers.append(score_text(ex.text, ct.text, level="standard").error_rate * 100)
        speedups.append(ex.metrics.total_ms / ct.metrics.total_ms)
    # int8 may flip a low-confidence token; it must not rewrite the sentence.
    assert max(wers) <= 25.0, wers
    assert min(speedups) > 1.0, speedups

    det = engine.transcribe_array(clips[0], 16_000, language=None)
    assert det.language == "hi" and det.metrics.language_probability > 0.9


def test_engine_import_is_lazy():
    import asr.engines  # noqa: F401  (must not import faster_whisper at package import)
