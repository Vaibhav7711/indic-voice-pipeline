import numpy as np
import pytest

from asr.explicit.chunking import chunk_audio, merge_overlapping_transcripts


def test_chunk_audio_uses_bounded_overlapping_windows():
    waveform = np.arange(51, dtype=np.float32)
    chunks = chunk_audio(waveform, 1, chunk_seconds=25, overlap_seconds=5)

    specs = [spec for spec, _ in chunks]
    assert [(s.start_sample, s.end_sample) for s in specs] == [(0, 25), (20, 45), (40, 51)]
    assert [s.index for s in specs] == [0, 1, 2]


def test_chunk_audio_validates_whisper_window_and_stride():
    waveform = np.ones(10, dtype=np.float32)
    with pytest.raises(ValueError, match="30"):
        chunk_audio(waveform, 1, chunk_seconds=31)
    with pytest.raises(ValueError, match="smaller"):
        chunk_audio(waveform, 1, chunk_seconds=10, overlap_seconds=10)


def test_merge_removes_only_verified_boundary_overlap():
    assert merge_overlapping_transcripts("यह एक अच्छी किताब है", "अच्छी किताब है और उपयोगी है") == "यह एक अच्छी किताब है और उपयोगी है"
    assert merge_overlapping_transcripts("वह सही है", "है और तैयार है") == "वह सही है और तैयार है"
    assert merge_overlapping_transcripts("Hello, world", "world! again") == "Hello, world again"
    assert merge_overlapping_transcripts("पहला वाक्य", "दूसरा वाक्य") == "पहला वाक्य दूसरा वाक्य"
