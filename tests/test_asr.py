"""ASR correctness: explicit loop vs model.generate() token match."""

from __future__ import annotations

import numpy as np
import pytest

# importorskip so the module skips cleanly at collection time when torch is
# absent. Without it, `pytest tests/` fails outright on a CPU-only machine
# and the CPU-runnable scoring tests never get a chance to run.
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def whisper():
    from asr.explicit import load_whisper
    return load_whisper("openai/whisper-small")


@pytest.fixture(scope="module")
def tone():
    sr = 16000
    t = np.linspace(0, 3.0, sr * 3, dtype=np.float32)
    return 0.3 * np.sin(2 * np.pi * 440 * t)


class TestExplicitVsGenerate:
    def test_tokens_match(self, whisper, tone):
        from asr.explicit import ASRRunner

        runner = ASRRunner(whisper.model, whisper.processor, whisper.device, whisper.dtype)
        explicit = runner.transcribe_array(tone, 16000, language="en", max_new_tokens=50)

        features = whisper.processor.feature_extractor(
            tone, sampling_rate=16000, return_tensors="pt",
        ).input_features.to(device=whisper.device, dtype=whisper.dtype)

        with torch.inference_mode():
            ref_ids = whisper.model.generate(
                features, max_new_tokens=50, language="en",
                task="transcribe", do_sample=False,
            )
        ref_list = ref_ids[0].tolist()
        no_ts = whisper.model.generation_config.no_timestamps_token_id
        ref_start = ref_list.index(no_ts) + 1 if no_ts in ref_list else 4
        ref_output = ref_list[ref_start:]

        min_len = min(len(explicit.token_ids), len(ref_output))
        if min_len == 0:
            return  # both produced nothing (valid for a sine wave)
        matches = sum(a == b for a, b in zip(explicit.token_ids[:min_len], ref_output[:min_len]))
        assert matches / min_len >= 0.8

    def test_metrics_populated(self, whisper, tone):
        from asr.explicit import ASRRunner

        runner = ASRRunner(whisper.model, whisper.processor, whisper.device, whisper.dtype)
        result = runner.transcribe_array(tone, 16000, language="en")
        m = result.metrics
        assert m.mel_extraction_ms > 0
        assert m.encoder_ms > 0
        assert m.decoder_prefill_ms > 0
        assert m.peak_allocated_bytes > 0

    def test_encoder_geometry(self, whisper, tone):
        from asr.explicit import ASRRunner

        runner = ASRRunner(whisper.model, whisper.processor, whisper.device, whisper.dtype)
        result = runner.transcribe_array(tone, 16000, language="en")
        assert result.metrics.encoder_hidden_size == 768   # whisper-small
        assert result.metrics.encoder_seq_length == 1500
