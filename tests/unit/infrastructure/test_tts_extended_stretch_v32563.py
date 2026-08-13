"""[v3.23.163] Test MƯỢN thời gian tới câu kế để giảm/né time-stretch.

Log thực tế (1454 câu): 7 câu vẫn cần stretch >=1.5x dù Pass 2.5 chỉ cứu được 1 — vì
các câu này đã kịch TRẦN rate Edge (3.0x) mà khung lịch vẫn quá chật. Giải pháp đúng
là mượn thời gian từ khoảng trống tới câu kế (extended window, đã trừ gap_guard nên
không đụng câu sau) làm khung stretch -> ratio giảm mạnh, nhiều câu về dưới 1.5x.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    _stretch_ratio_with_extended,
)

_MAX_SPEED = 3.0


def test_no_stretch_when_fits_extended_window() -> None:
    # Audio 3.0s, khung chặt 2.0s (cần stretch 1.5x) NHƯNG có gap tới 3.1s -> vừa khít
    # khung rộng -> KHÔNG stretch (ratio 1.0).
    ratio = _stretch_ratio_with_extended(3.0, 2.0, 3.1, _MAX_SPEED)
    assert ratio == 1.0


def test_ratio_reduced_by_extended_window() -> None:
    # Audio 3.0s, khung chặt 1.5s (chặt -> stretch 2.0x) nhưng gap tới 2.5s ->
    # chỉ còn 3.0/2.5 = 1.2x (dưới ngưỡng cảnh báo 1.5x).
    ratio = _stretch_ratio_with_extended(3.0, 1.5, 2.5, _MAX_SPEED)
    assert abs(ratio - 1.2) < 1e-9


def test_still_stretches_when_no_gap() -> None:
    # Không có khoảng trống (extended == desired) -> vẫn stretch theo khung chặt.
    ratio = _stretch_ratio_with_extended(3.0, 1.5, 1.5, _MAX_SPEED)
    assert abs(ratio - 2.0) < 1e-9


def test_capped_at_max_speed() -> None:
    ratio = _stretch_ratio_with_extended(10.0, 1.0, 1.0, _MAX_SPEED)
    assert ratio == _MAX_SPEED


def test_extended_never_worse_than_desired() -> None:
    # Extended nhỏ hơn desired (bất thường) -> lấy desired, không làm xấu đi.
    ratio = _stretch_ratio_with_extended(3.0, 2.0, 1.0, _MAX_SPEED)
    assert abs(ratio - 1.5) < 1e-9


def test_tiny_overshoot_not_stretched() -> None:
    # Vượt < 2% -> bỏ qua (ratio 1.0), tránh stretch vô nghĩa.
    assert _stretch_ratio_with_extended(2.01, 2.0, 2.0, _MAX_SPEED) == 1.0


def test_invalid_window_returns_one() -> None:
    assert _stretch_ratio_with_extended(3.0, 0.0, 0.0, _MAX_SPEED) == 1.0
