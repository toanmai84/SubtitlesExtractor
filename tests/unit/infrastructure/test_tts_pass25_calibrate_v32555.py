"""[v3.23.155/162] Test Pass 2.5 — rate Edge hiệu chỉnh ĐỐI XỨNG (nhanh + chậm).

Trả lời "time-stretch cao mà không giảm chất lượng": KHÔNG stretch DSP — nếu audio
lệch khung mà rate Edge còn dư địa thì tổng hợp LẠI với rate hiệu chỉnh (Edge đọc
nhanh/chậm tự nhiên hơn hẳn stretch). v162 bổ sung chiều CHẬM (audio ngắn hơn khung
-> giảm rate) để né cả stretch giãn.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    _CALIBRATE_TRIGGER_FAST,
    _CALIBRATE_TRIGGER_SLOW,
    _EDGE_API_SPEED_MAX,
    _EDGE_API_SPEED_MIN,
    _calibrated_rate,
)

_CAP = _EDGE_API_SPEED_MAX
_FLOOR = _EDGE_API_SPEED_MIN


# ── Chiều NHANH (audio dài hơn khung) ────────────────────────────────────


def test_speeds_up_when_audio_too_long() -> None:
    # Pass 2 ở 1.5x, audio 3.0s cho khung 2.0s (residual 1.5) -> rate mới 2.25.
    rate = _calibrated_rate(1.5, 3.0, 2.0, _CAP, _FLOOR)
    assert rate is not None and abs(rate - 2.25) < 1e-9


def test_skips_small_fast_residual() -> None:
    # residual 1.1 < ngưỡng nhanh 1.15 -> stretch nhẹ chấp nhận, không thêm request.
    assert _calibrated_rate(1.5, 2.2, 2.0, _CAP, _FLOOR) is None
    assert _CALIBRATE_TRIGGER_FAST == 1.15


def test_skips_when_rate_at_cap() -> None:
    assert _calibrated_rate(3.0, 4.0, 2.0, _CAP, _FLOOR) is None


def test_caps_new_rate_at_max() -> None:
    assert _calibrated_rate(2.0, 6.0, 2.0, _CAP, _FLOOR) == _CAP


# ── Chiều CHẬM (audio ngắn hơn khung) — MỚI v162 ─────────────────────────


def test_slows_down_when_audio_too_short() -> None:
    # Pass 2 ở 1.0x, audio 1.6s cho khung 2.0s (residual 0.8) -> rate mới 0.8.
    rate = _calibrated_rate(1.0, 1.6, 2.0, _CAP, _FLOOR)
    assert rate is not None and abs(rate - 0.8) < 1e-9


def test_skips_small_slow_residual() -> None:
    # residual 0.95 > ngưỡng chậm 0.87 -> giãn nhẹ chấp nhận, không thêm request.
    assert _calibrated_rate(1.0, 1.9, 2.0, _CAP, _FLOOR) is None
    assert _CALIBRATE_TRIGGER_SLOW == 0.87


def test_skips_when_rate_at_floor() -> None:
    # Đã kịch sàn 0.5 -> không giảm thêm được.
    assert _calibrated_rate(0.5, 1.0, 2.0, _CAP, _FLOOR) is None


def test_slow_rate_floored_at_min() -> None:
    # residual rất nhỏ -> rate mới bị kẹp sàn 0.5.
    rate = _calibrated_rate(1.0, 0.5, 2.0, _CAP, _FLOOR)
    assert rate == _FLOOR


# ── Bất biến chung ───────────────────────────────────────────────────────


def test_invalid_inputs_return_none() -> None:
    assert _calibrated_rate(1.5, 0.0, 2.0, _CAP, _FLOOR) is None
    assert _calibrated_rate(1.5, 3.0, 0.0, _CAP, _FLOOR) is None


def test_tiny_gain_not_worth_request() -> None:
    # Rate 2.98 -> trần 3.0: mức tăng < 0.04 -> không đáng thêm request.
    assert _calibrated_rate(2.98, 4.0, 2.0, _CAP, _FLOOR) is None
