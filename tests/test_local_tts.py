"""Local TTS backend contract. Model-backed tests run only if the MMS
checkpoint is already in the HF cache (no downloads in CI)."""

from __future__ import annotations

import numpy as np
import pytest

from tts.local import MMS_LANGUAGES, MmsTtsSynthesizer


def _cached(model_name: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(model_name, "config.json"), str)
    except Exception:  # noqa: BLE001
        return False


def test_construction_is_lazy_and_language_map_is_sane():
    s = MmsTtsSynthesizer("hi")
    assert s._model is None
    assert s.model_name == "facebook/mms-tts-hin"
    assert s.streaming is False
    assert s.format.codec == "pcm_s16le"
    assert {"hi", "te", "ta", "en"} <= set(MMS_LANGUAGES)
    with pytest.raises(ValueError, match="no MMS-TTS checkpoint"):
        MmsTtsSynthesizer("xx")


def test_empty_text_yields_nothing_without_loading():
    s = MmsTtsSynthesizer("hi")
    assert list(s.stream("   ")) == []
    assert s._model is None


@pytest.mark.skipif(not _cached("facebook/mms-tts-hin"), reason="MMS-hin not in HF cache")
def test_mms_hindi_synthesises_pcm_chunks_that_decode_to_speech_length_audio():
    from agent.audio import DecodingBufferSink

    s = MmsTtsSynthesizer("hi", chunk_ms=100)
    chunks = list(s.stream("नमस्ते, आप कैसे हैं?"))
    assert len(chunks) > 5
    sink = DecodingBufferSink(s.format)
    for c in chunks:
        sink.write(c)
    sink.close()
    assert 0.8 < sink.seconds < 4.0
    pcm = sink.audio.astype(np.float32)
    assert np.abs(pcm).max() > 1000, "not silence"
    assert s.last_synthesis_ms is not None and s.last_synthesis_ms > 0
    # Chunk size honoured (100 ms at 16 kHz = 1600 samples = 3200 bytes).
    assert all(len(c) == 3200 for c in chunks[:-1])
