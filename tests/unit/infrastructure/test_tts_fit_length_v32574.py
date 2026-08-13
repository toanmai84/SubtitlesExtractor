"""[v3.23.174] Test khớp độ dài audio sau time-stretch KHÔNG chèn im lặng cuối.

Bug: OLA/WSOLA sinh độ dài lệch nhẹ so với đích (do chia nguyên khi tính số khung).
Phần THIẾU trước đây được ``np.pad`` bằng zeros -> chèn im lặng vào ĐUÔI giọng (đo
thực: tới 17ms ở nén nhẹ ratio 1.03) -> câu nghe "hụt hơi" cuối. Fix: hàm thuần
``_fit_length_no_silence`` nội suy tuyến tính giãn phần giọng về đúng đích thay vì pad.
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    EdgeTTSAdapter,
    _fit_length_no_silence,
)

_SR = 24000


def _tone(duration_s: float, freq: float = 200.0) -> np.ndarray:
    samples = np.arange(int(_SR * duration_s))
    return (0.5 * np.sin(2 * np.pi * freq * samples / _SR)).astype(np.float32)


# ── Hàm thuần _fit_length_no_silence ─────────────────────────────────────


def test_exact_length_unchanged() -> None:
    audio = _tone(0.5)
    result = _fit_length_no_silence(audio, len(audio))
    assert len(result) == len(audio)
    assert np.array_equal(result, audio)


def test_longer_is_truncated() -> None:
    audio = _tone(0.5)
    target = len(audio) - 500
    result = _fit_length_no_silence(audio, target)
    assert len(result) == target


def test_shorter_is_interpolated_not_padded() -> None:
    audio = _tone(0.5)
    target = len(audio) + 400
    result = _fit_length_no_silence(audio, target)
    assert len(result) == target
    # Không có im lặng cuối: 400 mẫu cuối vẫn là tín hiệu (nội suy), không phải zeros.
    tail_energy = float(np.sqrt(np.mean(result[-400:] ** 2)))
    assert tail_energy > 0.1


def test_single_sample_repeats_value() -> None:
    result = _fit_length_no_silence(np.array([0.7], dtype=np.float32), 10)
    assert len(result) == 10
    assert np.allclose(result, 0.7)


def test_empty_target_returns_as_is() -> None:
    audio = _tone(0.1)
    result = _fit_length_no_silence(audio, 0)
    assert len(result) == len(audio)


# ── Tích hợp: OLA/WSOLA không còn im lặng đuôi ───────────────────────────


def _tail_silence_ms(audio: np.ndarray) -> float:
    zeros = 0
    for value in reversed(audio):
        if abs(value) < 1e-4:
            zeros += 1
        else:
            break
    return zeros / _SR * 1000.0


def test_ola_no_tail_silence_after_light_compression() -> None:
    audio = _tone(0.8)
    for ratio in (1.03, 1.04, 1.05):
        out = EdgeTTSAdapter._ola_time_stretch(audio, _SR, ratio)
        assert _tail_silence_ms(out) < 2.0  # trước fix: tới 17ms
        assert abs(len(out) - int(len(audio) / ratio)) <= 1


def test_ola_length_matches_target() -> None:
    audio = _tone(0.6)
    out = EdgeTTSAdapter._ola_time_stretch(audio, _SR, 1.04)
    assert abs(len(out) - int(len(audio) / 1.04)) <= 1
