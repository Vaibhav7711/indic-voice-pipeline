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


# ---------------------------------------------------------------------------
# Adaptive threshold: quiet speech must not be deleted
# ---------------------------------------------------------------------------

def _dbfs_tone(seconds: float, dbfs: float, sr: int = 16_000) -> np.ndarray:
    amp = 10 ** (dbfs / 20)
    return np.full(round(seconds * sr), amp, dtype=np.float32)


def test_quiet_speech_below_fixed_threshold_is_kept_with_adaptive_rule():
    """Clip 42 in the GPU sweep: speech around -45 dBFS, floor near -62. A
    fixed -40 threshold deleted most of it; the adaptive rule must keep it."""
    sr = 16_000
    audio = np.concatenate([
        _dbfs_tone(1.0, -62, sr), _dbfs_tone(2.0, -45, sr), _dbfs_tone(1.0, -62, sr),
    ])
    fixed = detect_speech(audio, sr, VADConfig(adaptive_threshold=False))
    adaptive = detect_speech(audio, sr, VADConfig())
    assert fixed == []
    assert len(adaptive) == 1
    assert abs(adaptive[0].start_seconds - 0.7) < 0.05      # 1.0 s minus padding
    assert abs(adaptive[0].end_seconds - 3.3) < 0.05


def test_adaptive_threshold_never_exceeds_fixed_ceiling():
    """Loud audio with no gaps (or starting mid-speech) still counts as speech."""
    sr = 16_000
    audio = _dbfs_tone(4.0, -10, sr)                        # no dips at all
    assert len(detect_speech(audio, sr, VADConfig())) == 1
    assert detect_speech(audio, sr, VADConfig())[0].start_sample == 0


def test_adaptive_threshold_has_a_floor_for_digital_silence():
    from asr.vad import AdaptiveThreshold

    thr = AdaptiveThreshold(VADConfig())
    for _ in range(50):
        level = thr.update(-200.0)
    assert level == VADConfig().threshold_floor_dbfs == -70.0
    assert thr.noise_floor_dbfs == -200.0


def test_adaptive_threshold_tracks_a_rising_noise_floor():
    from asr.vad import AdaptiveThreshold

    cfg = VADConfig(noise_window_ms=100, hop_ms=10, noise_margin_db=10)
    thr = AdaptiveThreshold(cfg)
    for _ in range(20):
        thr.update(-70.0)
    assert thr.update(-70.0) == -60.0                       # -70 + 10
    for _ in range(11):                                    # window is 10 frames
        level = thr.update(-50.0)
    assert level == -40.0                                   # old quiet frames aged out


def test_offline_and_online_agree_on_quiet_speech():
    from asr.streaming import EndpointEventKind, StreamEndpointer

    sr = 16_000
    audio = np.concatenate([
        _dbfs_tone(0.5, -62, sr), _dbfs_tone(1.5, -45, sr), _dbfs_tone(0.3, -62, sr),
        _dbfs_tone(1.0, -47, sr), _dbfs_tone(1.0, -62, sr),
    ])
    cfg = VADConfig()
    offline = detect_speech(audio, sr, cfg)
    ep = StreamEndpointer(sr, cfg)
    events = ep.push(audio) + ep.flush()
    starts = [e.sample for e in events if e.kind == EndpointEventKind.SPEECH_START]
    ends = [e.sample for e in events if e.kind == EndpointEventKind.SPEECH_END]
    assert len(offline) == len(starts) == 1                 # 0.3 s gap merged
    assert abs(offline[0].start_sample - starts[0]) <= ep.hop_samples
    assert abs(offline[0].end_sample - ends[0]) <= ep.hop_samples
