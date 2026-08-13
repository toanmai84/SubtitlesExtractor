"""Test ánh xạ toạ độ ROI giữa khung hiển thị và khung video — v3.23.311.

Vùng ROI quyết định phần ảnh đưa vào OCR nên sai vài điểm ảnh là hỏng chất lượng
trích xuất. Module ``roi_geometry`` là thuần (không Qt) nên kiểm thử được đầy đủ.

Tính chất QUAN TRỌNG NHẤT được kiểm ở đây là **ổn định khứ hồi**: đổi cỡ cửa sổ nhiều
lần không được làm ROI trôi dần.

Ghi chú về giới hạn vật lý: khi khung nhỏ hơn video, một điểm ảnh khung phủ ``1/scale``
điểm ảnh video, nên khứ hồi KHÔNG THỂ chính xác hơn mức đó. Các test dưới đây dùng
đúng bất biến này thay vì đòi khớp tuyệt đối.
"""

from __future__ import annotations

import math

import pytest

from subtitles_extractor.presentation.widgets.roi_geometry import (
    compute_display_geometry,
    video_rect_to_widget,
    widget_rect_to_video,
)

# Kích thước video CJK dọc — đúng định dạng dự án xử lý.
_VIDEO_W, _VIDEO_H = 720, 1280

_WIDGET_SIZES = [
    (1600, 900),
    (800, 600),
    (1920, 1080),
    (400, 300),
    (1000, 1000),
    (3840, 2160),
]

_SAMPLE_ROIS = [
    (0, 0, _VIDEO_W, _VIDEO_H),          # toàn khung
    (100, 1000, 520, 200),               # dải phụ đề điển hình
    (0, 1100, _VIDEO_W, 180),            # sát đáy
    (200, 300, 320, 400),                # giữa khung
]


# ── Thu phóng & căn giữa ─────────────────────────────────────────────────────
def test_geometry_keeps_aspect_ratio() -> None:
    """Video phải giữ nguyên tỉ lệ và nằm gọn trong khung."""
    geometry = compute_display_geometry(
        video_width=_VIDEO_W, video_height=_VIDEO_H,
        widget_width=1600, widget_height=900,
    )
    x, y, width, height = geometry.displayed_rect
    assert geometry.is_valid
    assert width / height == pytest.approx(_VIDEO_W / _VIDEO_H, rel=0.01)
    assert x >= 0 and y >= 0
    assert x + width <= 1600 and y + height <= 900


def test_pillarbox_for_vertical_video_in_wide_widget() -> None:
    """Video dọc trong khung ngang -> viền đen HAI BÊN, không phải trên dưới."""
    geometry = compute_display_geometry(
        video_width=_VIDEO_W, video_height=_VIDEO_H,
        widget_width=1600, widget_height=900,
    )
    assert geometry.offset_x > 0
    assert geometry.offset_y == pytest.approx(0.0, abs=0.5)


def test_letterbox_for_wide_video_in_tall_widget() -> None:
    """Video ngang trong khung cao -> viền đen TRÊN DƯỚI."""
    geometry = compute_display_geometry(
        video_width=1920, video_height=1080,
        widget_width=800, widget_height=600,
    )
    assert geometry.offset_y > 0
    assert geometry.offset_x == pytest.approx(0.0, abs=0.5)


@pytest.mark.parametrize(
    ("video_w", "video_h", "widget_w", "widget_h"),
    [(0, 0, 100, 100), (720, 1280, 0, 0), (-10, 100, 100, 100), (720, 0, 100, 100)],
)
def test_invalid_sizes_are_rejected(
    video_w: int, video_h: int, widget_w: int, widget_h: int
) -> None:
    geometry = compute_display_geometry(
        video_width=video_w, video_height=video_h,
        widget_width=widget_w, widget_height=widget_h,
    )
    assert not geometry.is_valid
    assert widget_rect_to_video((0, 0, 10, 10), geometry) is None
    assert video_rect_to_widget((0, 0, 10, 10), geometry) == (0, 0, 0, 0)


# ── Khứ hồi ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("widget_w", "widget_h"), _WIDGET_SIZES)
@pytest.mark.parametrize("roi", _SAMPLE_ROIS)
def test_round_trip_within_quantisation_limit(
    widget_w: int, widget_h: int, roi: tuple[int, int, int, int]
) -> None:
    """Sai số khứ hồi không vượt quá 1 điểm ảnh khung quy ra điểm ảnh video."""
    geometry = compute_display_geometry(
        video_width=_VIDEO_W, video_height=_VIDEO_H,
        widget_width=widget_w, widget_height=widget_h,
    )
    result = widget_rect_to_video(video_rect_to_widget(roi, geometry), geometry)
    assert result is not None
    limit = math.ceil(1 / geometry.scale) + 1
    assert max(abs(a - b) for a, b in zip(roi, result)) <= limit


@pytest.mark.parametrize(("widget_w", "widget_h"), _WIDGET_SIZES)
@pytest.mark.parametrize("roi", _SAMPLE_ROIS)
def test_round_trip_is_stable_over_many_iterations(
    widget_w: int, widget_h: int, roi: tuple[int, int, int, int]
) -> None:
    """TÍNH CHẤT THEN CHỐT: lặp khứ hồi phải HỘI TỤ, không trôi dần.

    Nếu sai, ROI của người dùng sẽ lệch dần mỗi lần đổi cỡ cửa sổ.
    """
    geometry = compute_display_geometry(
        video_width=_VIDEO_W, video_height=_VIDEO_H,
        widget_width=widget_w, widget_height=widget_h,
    )
    first = widget_rect_to_video(video_rect_to_widget(roi, geometry), geometry)
    assert first is not None
    current = first
    for _ in range(50):
        current = widget_rect_to_video(video_rect_to_widget(current, geometry), geometry)
        assert current is not None
    assert current == first


# ── Cắt biên & kẹp ───────────────────────────────────────────────────────────
def test_drag_entirely_on_black_bar_returns_none() -> None:
    """Kéo hoàn toàn trên viền đen -> không tạo ROI."""
    geometry = compute_display_geometry(
        video_width=_VIDEO_W, video_height=_VIDEO_H,
        widget_width=1600, widget_height=900,
    )
    display_x = geometry.displayed_rect[0]
    assert widget_rect_to_video((0, 0, display_x - 5, 900), geometry) is None


def test_drag_overlapping_black_bar_is_clipped() -> None:
    """Kéo đè một phần lên viền đen -> phần thừa bị cắt, ROI bám mép ảnh."""
    geometry = compute_display_geometry(
        video_width=_VIDEO_W, video_height=_VIDEO_H,
        widget_width=1600, widget_height=900,
    )
    display_x = geometry.displayed_rect[0]
    roi = widget_rect_to_video((display_x - 50, 100, 200, 300), geometry)
    assert roi is not None
    assert roi[0] == 0  # bám mép trái của ảnh


def test_result_always_inside_video_bounds() -> None:
    """Mọi ROI trả về phải nằm trọn trong khung ảnh."""
    geometry = compute_display_geometry(
        video_width=_VIDEO_W, video_height=_VIDEO_H,
        widget_width=1600, widget_height=900,
    )
    display_x, display_y, display_w, display_h = geometry.displayed_rect
    roi = widget_rect_to_video(
        (display_x + display_w - 10, display_y + display_h - 10, 500, 500), geometry
    )
    assert roi is not None
    x, y, width, height = roi
    assert x >= 0 and y >= 0
    assert x + width <= _VIDEO_W
    assert y + height <= _VIDEO_H
    assert width >= 1 and height >= 1


@pytest.mark.parametrize("rect", [(100, 100, 0, 50), (100, 100, 50, 0), (0, 0, -5, -5)])
def test_degenerate_drag_returns_none(rect: tuple[int, int, int, int]) -> None:
    geometry = compute_display_geometry(
        video_width=_VIDEO_W, video_height=_VIDEO_H,
        widget_width=1600, widget_height=900,
    )
    assert widget_rect_to_video(rect, geometry) is None


def test_sub_pixel_roi_returns_none_rather_than_wrong_value() -> None:
    """ROI nhỏ hơn 1 điểm ảnh khung -> trả None, KHÔNG trả ROI sai."""
    geometry = compute_display_geometry(
        video_width=_VIDEO_W, video_height=_VIDEO_H,
        widget_width=400, widget_height=300,
    )
    assert geometry.scale < 0.5  # thu nhỏ mạnh
    widget_rect = video_rect_to_widget((360, 640, 1, 1), geometry)
    assert widget_rect_to_video(widget_rect, geometry) is None


def test_exact_fit_has_no_offset() -> None:
    """Khớp tỉ lệ chính xác -> không có viền đen."""
    geometry = compute_display_geometry(
        video_width=1920, video_height=1080,
        widget_width=960, widget_height=540,
    )
    assert geometry.scale == pytest.approx(0.5)
    assert geometry.offset_x == pytest.approx(0.0)
    assert geometry.offset_y == pytest.approx(0.0)
    assert geometry.displayed_rect == (0, 0, 960, 540)


def test_upscaled_video_still_maps_correctly() -> None:
    """Khung LỚN hơn video (phóng to) vẫn ánh xạ đúng."""
    geometry = compute_display_geometry(
        video_width=_VIDEO_W, video_height=_VIDEO_H,
        widget_width=3840, widget_height=2160,
    )
    assert geometry.scale > 1.0
    roi = (100, 1000, 520, 200)
    result = widget_rect_to_video(video_rect_to_widget(roi, geometry), geometry)
    assert result is not None
    assert max(abs(a - b) for a, b in zip(roi, result)) <= 1
