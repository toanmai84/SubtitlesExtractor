"""Test cửa sổ "Thử nhanh" OCR — v3.23.320.

Trích xuất một tập 40 phút mất rất nhiều thời gian. Nếu ROI sai hoặc sai ngôn ngữ,
người dùng chỉ biết SAU KHI chạy xong. Thử nhanh bó phạm vi vào ~60 giây để kiểm trước.

Bất biến then chốt: ``skip_intro_sec + skip_outro_sec < duration`` — vi phạm là bộ lấy
mẫu không còn khung hình nào, lần thử ra rỗng và người dùng tưởng OCR hỏng.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.ocr_probe_window import (
    DEFAULT_PROBE_SECONDS,
    compute_probe_window,
    summarise_probe_result,
)


@pytest.mark.parametrize("duration", [2700.0, 1800.0, 600.0, 300.0, 90.0, 61.0])
def test_skips_never_consume_whole_video(duration: float) -> None:
    """BẤT BIẾN: phải còn khung hình để xử lý sau khi cắt hai đầu."""
    window = compute_probe_window(duration)
    assert window.skip_intro_sec + window.skip_outro_sec < duration
    assert window.length_sec > 0


@pytest.mark.parametrize("duration", [2700.0, 1800.0, 300.0, 90.0])
def test_window_stays_inside_video(duration: float) -> None:
    window = compute_probe_window(duration)
    assert window.start_sec >= 0
    assert window.end_sec <= duration + 1e-6
    assert window.start_sec < window.end_sec


def test_window_is_centred_by_default() -> None:
    """Mặc định thử GIỮA phim — đầu là intro/logo, cuối là credits, dễ ra rỗng."""
    duration = 2700.0
    window = compute_probe_window(duration)
    centre = (window.start_sec + window.end_sec) / 2
    assert centre == pytest.approx(duration / 2, abs=1.0)


def test_window_length_matches_request() -> None:
    window = compute_probe_window(2700.0, probe_seconds=90.0)
    assert window.length_sec == pytest.approx(90.0)


@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_any_centre_ratio_stays_in_bounds(ratio: float) -> None:
    """Thử ở đầu/cuối phim cũng không được tràn ra ngoài."""
    duration = 2700.0
    window = compute_probe_window(duration, center_ratio=ratio)
    assert 0 <= window.start_sec
    assert window.end_sec <= duration + 1e-6
    assert window.length_sec == pytest.approx(DEFAULT_PROBE_SECONDS)


@pytest.mark.parametrize("duration", [60.0, 30.0, 10.0, 5.0, 1.0, 0.0])
def test_short_video_is_probed_entirely(duration: float) -> None:
    """Video ngắn hơn cửa sổ -> thử toàn bộ, KHÔNG cắt (cắt chỉ tổ mất dữ liệu)."""
    window = compute_probe_window(duration)
    assert window.skip_intro_sec == 0.0
    assert window.skip_outro_sec == 0.0


def test_negative_duration_is_handled() -> None:
    window = compute_probe_window(-10.0)
    assert window.skip_intro_sec == 0.0
    assert window.skip_outro_sec == 0.0


def test_probe_seconds_larger_than_video() -> None:
    window = compute_probe_window(30.0, probe_seconds=600.0)
    assert window.skip_intro_sec == 0.0
    assert window.skip_outro_sec == 0.0


def test_label_is_human_readable() -> None:
    window = compute_probe_window(2700.0)
    assert "đến" in window.label_vi
    assert ":" in window.label_vi


def test_empty_result_explains_what_to_check() -> None:
    """Không ra câu nào -> phải NÊU RÕ cần kiểm gì, không chỉ báo 'thất bại'."""
    message = summarise_probe_result(0, compute_probe_window(2700.0), [])
    assert "ROI" in message
    assert "ngôn ngữ" in message
    assert "track phụ đề nhúng" in message  # gợi ý đường vào khác


def test_successful_result_shows_sample_text() -> None:
    window = compute_probe_window(2700.0)
    message = summarise_probe_result(12, window, ["你好世界", "我们走吧"])
    assert "12 câu" in message
    assert "你好世界" in message


def test_long_sample_list_is_truncated() -> None:
    window = compute_probe_window(2700.0)
    texts = [f"câu {i}" for i in range(20)]
    message = summarise_probe_result(20, window, texts)
    assert "câu 0" in message
    assert "và 15 câu nữa" in message
