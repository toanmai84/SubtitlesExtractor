"""[v3.23.178] Test gain chuẩn hoá loudness CÓ TRẦN (chống nổi nhiễu nền).

Bug: pipeline tính ``gain = 10^((target - measured)/20)`` KHÔNG giới hạn trên. Master
loudness rất thấp (phim thoại thưa/nhỏ, vd -40 LUFS) -> gain +26dB -> NỔI nhiễu nền
Edge + đẩy vô số đỉnh vượt trần khiến true-peak limiter ghìm liên tục gây méo/pumping.
Fix: hàm thuần ``_loudness_gain_linear`` chặn trần +15dB (chuẩn phát thanh).
"""

from __future__ import annotations

import math

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    _loudness_gain_linear,
)


def test_gain_capped_at_max_db() -> None:
    # measured -40, target -14 -> cần +26dB nhưng chặn ở +15dB.
    gain = _loudness_gain_linear(-40.0, -14.0, max_gain_db=15.0)
    assert math.isclose(20.0 * math.log10(gain), 15.0, abs_tol=1e-6)


def test_normal_gain_not_capped() -> None:
    # measured -20, target -14 -> +6dB, dưới trần -> giữ nguyên.
    gain = _loudness_gain_linear(-20.0, -14.0, max_gain_db=15.0)
    assert math.isclose(20.0 * math.log10(gain), 6.0, abs_tol=1e-6)


def test_attenuation_not_capped() -> None:
    # measured -10 (to hơn target -14) -> giảm -4dB; trần chỉ áp cho khuếch đại.
    gain = _loudness_gain_linear(-10.0, -14.0, max_gain_db=15.0)
    assert gain < 1.0
    assert math.isclose(20.0 * math.log10(gain), -4.0, abs_tol=1e-6)


def test_silence_returns_unity_gain() -> None:
    assert _loudness_gain_linear(float("-inf"), -14.0) == 1.0


def test_exact_target_returns_unity() -> None:
    gain = _loudness_gain_linear(-14.0, -14.0)
    assert math.isclose(gain, 1.0, abs_tol=1e-9)


def test_custom_max_gain() -> None:
    # Nới trần lên +20dB -> ca -40 nay cho +20dB.
    gain = _loudness_gain_linear(-40.0, -14.0, max_gain_db=20.0)
    assert math.isclose(20.0 * math.log10(gain), 20.0, abs_tol=1e-6)


def test_boundary_exactly_at_cap() -> None:
    # Cần đúng +15dB -> không bị cắt thêm.
    gain = _loudness_gain_linear(-29.0, -14.0, max_gain_db=15.0)
    assert math.isclose(20.0 * math.log10(gain), 15.0, abs_tol=1e-6)
