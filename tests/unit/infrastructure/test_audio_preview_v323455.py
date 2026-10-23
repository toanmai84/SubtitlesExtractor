"""[v3.23.455] Test đệm im lặng đầu audio cho nghe thử (chống nuốt đầu câu do cold-start)."""
from __future__ import annotations
import numpy as np
import pytest
from subtitles_extractor.infrastructure.tts.audio_preview import prepend_silence

SR = 48000


def test_prepend_adds_silence_before_audio():
    audio = np.ones(4800, dtype=np.float32)
    out = prepend_silence(audio, SR, 0.8)
    pad = int(0.8 * SR)
    assert len(out) == pad + 4800
    assert np.all(out[:pad] == 0.0)      # đầu im lặng
    assert np.all(out[pad:] == 1.0)      # sau là tiếng


def test_zero_silence_returns_audio():
    audio = np.ones(100, dtype=np.float32)
    out = prepend_silence(audio, SR, 0.0)
    assert len(out) == 100


def test_stereo_reduced_to_mono():
    st = np.column_stack([np.ones(100), np.ones(100) * 0.5]).astype(np.float32)
    out = prepend_silence(st, SR, 0.0)
    assert out.ndim == 1 and len(out) == 100


def test_invalid_sample_rate_raises():
    with pytest.raises(ValueError):
        prepend_silence(np.ones(10, dtype=np.float32), 0, 0.8)


def test_output_is_float32():
    out = prepend_silence(np.ones(10, dtype=np.float32), SR, 0.5)
    assert out.dtype == np.float32
