"""[v3.23.201] Test bù RMS sau pedalboard stretch (câu nén không còn nghe nhỏ).

Thực nghiệm trên giọng VieNeu thật (第1集.flac, câu 3.96s): Rubber Band làm RMS GIẢM
dần theo mức nén (-16%% @2x, -22%% @3x) trong khi WSOLA giữ nguyên. Master chuẩn hoá
loudness TOÀN file nên câu nén sâu nghe NHỎ tương đối -> giảm truyền đạt nội dung.
Fix: ``match_rms`` bù per-segment ngay trong ``stretch_with_pedalboard`` (nguồn duy
nhất — mọi engine cùng hưởng), trần gain 2x chống khuếch đại nhiễu.

Cùng thực nghiệm: đo độ sắc nét phụ âm (onset clarity) xác nhận trần chất lượng 2.0 là
ĐÚNG (mất ~25%% transient @2x nhưng 32-44%% @2.5-3x) -> GIỮ nguyên, không nới.
"""

from __future__ import annotations

import numpy as np
import pytest

from subtitles_extractor.infrastructure.tts.time_stretch import (
    match_rms,
    stretch_with_pedalboard,
)

_SR = 24000


def _voice_like(duration_s: float = 1.0, amp: float = 0.3) -> np.ndarray:
    # Tín hiệu giàu hài như giọng nói (f0 180Hz + hài) để stretch có ý nghĩa.
    t = np.arange(int(_SR * duration_s)) / _SR
    sig = sum((amp / (k + 1)) * np.sin(2 * np.pi * 180 * (k + 1) * t) for k in range(4))
    return sig.astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt((x.astype(np.float64) ** 2).mean()))


# ── match_rms (hàm thuần) ────────────────────────────────────────────────


def test_match_rms_restores_target_level() -> None:
    quiet = _voice_like(amp=0.1)
    target = 0.2
    out = match_rms(quiet, target, max_gain=4.0)
    assert abs(_rms(out) - target) / target < 0.05


def test_match_rms_caps_gain() -> None:
    quiet = _voice_like(amp=0.01)
    out = match_rms(quiet, 1.0, max_gain=2.0)
    assert _rms(out) / _rms(quiet) == pytest.approx(2.0, rel=0.02)


def test_match_rms_silence_not_amplified() -> None:
    silence = np.zeros(_SR, dtype=np.float32)
    out = match_rms(silence, 0.2)
    assert float(np.abs(out).max()) == 0.0  # không khuếch đại nhiễu nền


def test_match_rms_noop_when_close() -> None:
    sig = _voice_like(amp=0.2)
    out = match_rms(sig, _rms(sig) * 1.01)  # chênh 1% < ngưỡng 2%
    assert out is sig  # trả nguyên bản, không copy thừa


def test_match_rms_zero_target_noop() -> None:
    sig = _voice_like()
    assert match_rms(sig, 0.0) is sig


# ── stretch_with_pedalboard: RMS được bù tại nguồn ───────────────────────


def test_pedalboard_stretch_preserves_rms() -> None:
    pytest.importorskip("pedalboard")
    sig = _voice_like(duration_s=2.0)
    r0 = _rms(sig)
    for ratio in (2.0, 3.0):
        out = stretch_with_pedalboard(sig, _SR, ratio)
        assert out is not None
        # Trước fix: -16%% @2x / -22%% @3x. Sau fix: giữ trong ±5%%.
        assert abs(_rms(out) - r0) / r0 < 0.05
