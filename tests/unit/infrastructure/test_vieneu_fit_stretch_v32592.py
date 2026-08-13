"""[v3.23.192] Test nén thời gian VieNeu vừa khung (giảm cắt đuôi từ 43%% xuống ~0).

Phát hiện từ dữ liệu thật (tts_debug.csv, 498 câu): VieNeu SDK sinh giọng ở tốc độ TỰ
NHIÊN cố định (cột Tốc độ scheduler/Edge API = 0.0, Tốc độ dùng = 1.30 mọi câu — không co
giãn động như EdgeTTS). 212/498 câu (43%%) bị CẮT CỨNG đuôi vì audio dài hơn khung -> mất
chữ cuối. Fix: hàm thuần ``compute_fit_stretch_ratio`` + time-stretch (giữ cao độ) cho
vừa khung thay vì cắt.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    compute_fit_stretch_ratio,
)


def test_no_stretch_when_fits() -> None:
    # Audio ngắn hơn khung -> không nén.
    assert compute_fit_stretch_ratio(1.4, 1.5, max_speed=3.0) == 1.0


def test_no_stretch_within_tolerance() -> None:
    # Vượt ít hơn dung sai -> không xử lý thừa.
    assert compute_fit_stretch_ratio(1.53, 1.5, max_speed=3.0, tolerance_s=0.05) == 1.0


def test_stretch_ratio_when_over() -> None:
    # Audio 2.0s, khung 1.5s -> nén 1.333x.
    ratio = compute_fit_stretch_ratio(2.0, 1.5, max_speed=3.0)
    assert abs(ratio - (2.0 / 1.5)) < 1e-6


def test_ratio_capped_at_max_speed() -> None:
    # Audio dài gấp 6 lần khung -> chặn ở max_speed 3.0 (không méo giọng).
    assert compute_fit_stretch_ratio(6.0, 1.0, max_speed=3.0) == 3.0


def test_zero_available_returns_one() -> None:
    assert compute_fit_stretch_ratio(2.0, 0.0, max_speed=3.0) == 1.0


def test_zero_audio_returns_one() -> None:
    assert compute_fit_stretch_ratio(0.0, 1.5, max_speed=3.0) == 1.0


def test_custom_max_speed() -> None:
    # max_speed 2.0 -> audio gấp 3 lần bị chặn ở 2.0.
    assert compute_fit_stretch_ratio(3.0, 1.0, max_speed=2.0) == 2.0


def test_exactly_at_frame_no_stretch() -> None:
    # Audio đúng bằng khung -> không nén.
    assert compute_fit_stretch_ratio(1.5, 1.5, max_speed=3.0) == 1.0
