"""[v3.23.180] Test noise gate ngưỡng THÍCH ỨNG theo đỉnh câu (không hạ oan giọng nhỏ).

Bug: noise gate dùng ngưỡng TUYỆT ĐỐI -42dBFS. Câu giọng NHỎ hợp lệ (thì thầm/yếu, đỉnh
envelope ~0.008) nằm trọn dưới ngưỡng -> bị hạ gain oan 36%% -> càng nhỏ đi, mất lời.
Fix: hàm thuần ``_noise_gate_threshold_linear`` lấy MIN giữa ngưỡng tuyệt đối và mức
tương đối -32dB SO đỉnh câu -> câu nhỏ đo theo chính đỉnh của nó (chỉ hạ im lặng giữa
từ), câu to vẫn khử nhiễu nền hiệu quả.
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    EdgeTTSAdapter,
    _noise_gate_threshold_linear,
)

_SR = 24000


def _tone(duration_s: float, amplitude: float, freq: float = 250.0) -> np.ndarray:
    samples = np.arange(int(_SR * duration_s))
    return (amplitude * np.sin(2 * np.pi * freq * samples / _SR)).astype(np.float32)


# ── Hàm thuần _noise_gate_threshold_linear ───────────────────────────────


def test_relative_threshold_used_for_quiet_peak() -> None:
    # Đỉnh nhỏ 0.008 -> ngưỡng tương đối -32dB = 0.008*0.0251 = 0.0002, nhỏ hơn tuyệt
    # đối 0.0079 -> lấy tương đối -> không hạ oan giọng nhỏ.
    threshold = _noise_gate_threshold_linear(0.008)
    assert threshold < 0.008  # ngưỡng dưới đỉnh câu -> giọng được giữ


def test_absolute_threshold_used_for_loud_peak() -> None:
    # Đỉnh lớn 0.8 -> tương đối -32dB = 0.02, lớn hơn tuyệt đối 0.0079 -> lấy tuyệt
    # đối để khử nhiễu nền hiệu quả.
    threshold = _noise_gate_threshold_linear(0.8)
    absolute_linear = 10.0 ** (-42.0 / 20.0)
    assert abs(threshold - absolute_linear) < 1e-9


def test_zero_peak_returns_absolute() -> None:
    threshold = _noise_gate_threshold_linear(0.0)
    assert abs(threshold - 10.0 ** (-42.0 / 20.0)) < 1e-9


def test_custom_thresholds() -> None:
    threshold = _noise_gate_threshold_linear(
        0.01, absolute_threshold_db=-40.0, relative_floor_db=-20.0
    )
    # tương đối -20dB so 0.01 = 0.001; tuyệt đối -40dB = 0.01 -> lấy min 0.001.
    assert abs(threshold - 0.001) < 1e-6


# ── Tích hợp _apply_noise_gate ───────────────────────────────────────────


def test_quiet_voice_not_attenuated() -> None:
    quiet = _tone(1.0, amplitude=0.008)
    out = EdgeTTSAdapter._apply_noise_gate(quiet, _SR)
    rms_before = float(np.sqrt(np.mean(quiet ** 2)))
    rms_after = float(np.sqrt(np.mean(out ** 2)))
    assert rms_after > rms_before * 0.9  # trước fix: giảm 36%


def test_silence_between_words_still_gated() -> None:
    speech = _tone(0.3, amplitude=0.5, freq=300.0)
    rng = np.random.default_rng(0)
    silence = (rng.standard_normal(int(_SR * 0.2)) * 0.001).astype(np.float32)
    mixed = np.concatenate([speech, silence, speech])
    out = EdgeTTSAdapter._apply_noise_gate(mixed, _SR)
    sil_slice = slice(len(speech), len(speech) + len(silence))
    rms_sil_before = float(np.sqrt(np.mean(mixed[sil_slice] ** 2)))
    rms_sil_after = float(np.sqrt(np.mean(out[sil_slice] ** 2)))
    assert rms_sil_after < rms_sil_before * 0.8  # im lặng vẫn được khử


def test_short_audio_returned_unchanged() -> None:
    short = _tone(0.02, amplitude=0.5)
    out = EdgeTTSAdapter._apply_noise_gate(short, _SR)
    assert np.array_equal(out, short)
