"""[v3.23.181] Test soft-clip giữa các tầng DSP (thay hard clip, giảm méo hài).

Bug: giữa các tầng voice_clarity, biên độ vọt qua ±1.0 do cộng dồn pha (EQ nâng clarity
+70%% đẩy 0.9 -> 1.05). Trước đây ``np.clip`` (hard clip) cắt PHẲNG đỉnh -> sinh hài bậc
cao chói tai. Fix: hàm thuần ``_inter_node_soft_clip`` dùng tanh knee — giữ nguyên phần
dưới ngưỡng, chỉ nén mềm phần vượt -> đỉnh < 1.0 mà không méo cứng.
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    _inter_node_soft_clip,
)

_SR = 24000


def test_peak_below_one_after_clip() -> None:
    signal = np.array([0.5, 1.048, -1.1, 0.9, 1.5], dtype=np.float32)
    out = _inter_node_soft_clip(signal)
    assert np.max(np.abs(out)) < 1.0


def test_samples_below_threshold_unchanged() -> None:
    signal = np.array([0.1, 0.5, -0.7, 0.85], dtype=np.float32)
    out = _inter_node_soft_clip(signal, threshold=0.90)
    # Toàn bộ dưới 0.90 -> giữ nguyên.
    assert np.array_equal(out, signal)


def test_no_op_when_all_within_range() -> None:
    signal = np.array([0.3, -0.6, 0.89], dtype=np.float32)
    out = _inter_node_soft_clip(signal, threshold=0.90)
    assert np.array_equal(out, signal)


def test_sign_preserved() -> None:
    signal = np.array([1.5, -1.5], dtype=np.float32)
    out = _inter_node_soft_clip(signal)
    assert out[0] > 0 and out[1] < 0


def test_soft_clip_less_harsh_than_hard_clip() -> None:
    # Soft clip sinh ÍT hài bậc cao hơn hard clip trên tín hiệu vượt mạnh.
    t = np.arange(_SR)
    signal = (1.4 * np.sin(2 * np.pi * 300 * t / _SR)).astype(np.float32)
    hard = np.clip(signal, -1.0, 1.0)
    soft = _inter_node_soft_clip(signal)

    def _high_harmonic_ratio(waveform: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(waveform * np.hanning(len(waveform))))
        fundamental = int(np.argmax(spectrum))
        return float(np.sum(spectrum[3 * fundamental:]) / np.sum(spectrum))

    assert _high_harmonic_ratio(soft) < _high_harmonic_ratio(hard)


def test_output_is_float32() -> None:
    signal = np.array([0.5, 1.2], dtype=np.float64)
    out = _inter_node_soft_clip(signal)
    assert out.dtype == np.float32


def test_empty_array() -> None:
    out = _inter_node_soft_clip(np.array([], dtype=np.float32))
    assert out.size == 0
