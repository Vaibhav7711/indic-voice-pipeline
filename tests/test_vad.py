import numpy as np

from asr.vad import VADConfig, detect_speech


def _audio(*durations_and_amplitudes: tuple[float, float], sample_rate: int = 1_000) -> np.ndarray:
    return np.concatenate([
        np.full(round(seconds * sample_rate), amplitude, dtype=np.float32)
        for seconds, amplitude in durations_and_amplitudes
    ])


def test_vad_discards_short_noise_and_pads_speech_boundaries():
    audio = _audio((0.5, 0.0), (0.1, 0.5), (0.5, 0.0), (0.4, 0.5), (0.5, 0.0))
    config = VADConfig(frame_ms=20, hop_ms=10, threshold_dbfs=-20, min_speech_ms=200, padding_ms=50)

    segments = detect_speech(audio, 1_000, config)

    assert len(segments) == 1
    # Analysis frames may straddle the amplitude transition; padding is then
    # applied around that frame-level endpoint.
    assert (segments[0].start_sample, segments[0].end_sample) == (1_040, 1_560)


def test_vad_merges_short_silence_but_splits_a_real_endpoint():
    audio = _audio((0.3, 0.5), (0.2, 0.0), (0.3, 0.5), (0.6, 0.0), (0.3, 0.5))
    config = VADConfig(frame_ms=20, hop_ms=10, threshold_dbfs=-20, min_speech_ms=100, min_silence_ms=300, padding_ms=0)

    segments = detect_speech(audio, 1_000, config)

    assert [(s.start_sample, s.end_sample) for s in segments] == [(0, 810), (1_390, 1_700)]


def test_vad_handles_empty_and_silent_audio():
    assert detect_speech(np.array([], dtype=np.float32), 16_000) == []
    assert detect_speech(np.zeros(16_000, dtype=np.float32), 16_000) == []
