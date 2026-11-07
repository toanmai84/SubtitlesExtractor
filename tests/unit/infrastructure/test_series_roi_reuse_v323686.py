"""[v3.23.686] Phim bộ: dùng lại ROI của tập trước sau một lượt kiểm nhanh, thay cho dò vùng đầy đủ mỗi tập.

Số liệu thật: tập 1 phim mẫu (720×1280) dò đầy đủ ra ROI x=54, y=818, 607×121; bề ngang câu dài nhất mỗi tập 406–603 px.
"""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.domain.value_objects.roi import Roi, TextOrientation
from subtitles_extractor.infrastructure.video.bbox_analyzer import RawBBox
from subtitles_extractor.infrastructure.video.roi_selection import refit_series_roi

W, H = 720, 1280
ROI = Roi(x=110, y=818, width=500, height=121)


def _box(frame: int, x0: float, x1: float, y0: float = 838, y1: float = 906, conf: float = 0.97) -> RawBBox:
    return RawBBox(
        coord_x_min=x0, coord_y_min=y0, coord_x_max=x1, coord_y_max=y1,
        confidence=conf, frame_idx=frame, timestamp_sec=frame * 5.0,
    )


def test_same_band_is_reused_and_widened_never_narrowed() -> None:
    boxes = [_box(i, 300 - 30 * (i % 4), 420 + 30 * (i % 4)) for i in range(20)]
    boxes.append(_box(20, 70, 650))                                   # câu dài hơn ROI của tập trước
    out = refit_series_roi(ROI, boxes, W, H)
    assert out is not None
    assert (out.x, out.y, out.height) == (70, 818, 121) and out.x + out.width == 650
    # Các câu đều ngắn hơn ROI cũ → giữ nguyên bề ngang cũ.
    short = [_box(i, 300, 420) for i in range(10)]
    assert refit_series_roi(ROI, short, W, H) == ROI


def test_subtitles_moved_or_missing_means_full_detection() -> None:
    # Phụ đề tập này nằm cao hơn hẳn (y 300) → dải đậm đặc nhất ngoài ROI cũ.
    moved = [_box(i, 200, 520, 300, 368) for i in range(20)] + [_box(1, 200, 400)]
    assert refit_series_roi(ROI, moved, W, H) is None
    # Chỉ 2 khung có chữ trong dải → không đủ bằng chứng.
    assert refit_series_roi(ROI, [_box(0, 300, 420), _box(1, 300, 420)], W, H) is None
    # Hộp mờ (độ tin cậy thấp) không tính.
    assert refit_series_roi(ROI, [_box(i, 300, 420, conf=0.5) for i in range(20)], W, H) is None
    assert refit_series_roi(ROI, [], W, H) is None


def test_scene_text_elsewhere_does_not_block_reuse_but_dominance_does() -> None:
    subs = [_box(i, 280, 440) for i in range(12)]
    sign = [_box(i, 100, 300, 100, 160) for i in range(3)]          # biển hiệu ở 3 khung
    assert refit_series_roi(ROI, subs + sign, W, H) is not None
    # Chữ trong dải ROI chỉ ở 3/20 khung có chữ (< 40 %) → không tin.
    other = [_box(i, 100, 300, 500, 560) for i in range(3, 20)]
    assert refit_series_roi(ROI, subs[:3] + other, W, H) is None


def test_band_boxes_must_have_subtitle_height() -> None:
    tall = [_box(i, 0, 720, 700, 1100) for i in range(20)]           # vật thể to phủ qua dải
    assert refit_series_roi(ROI, tall, W, H) is None


def test_vertical_roi_is_not_refit() -> None:
    vertical = Roi(x=600, y=100, width=100, height=900, orientation=TextOrientation.VERTICAL)
    assert refit_series_roi(vertical, [_box(i, 610, 690, 200, 800) for i in range(10)], W, H) is None


def test_page_reuses_roi_only_for_same_frame_size() -> None:
    from subtitles_extractor.presentation.pages.extract_page import ExtractPage

    page = SimpleNamespace(_series_learned_roi=(ROI, (W, H)))
    same = SimpleNamespace(width=W, height=H)
    other = SimpleNamespace(width=1920, height=1080)
    assert ExtractPage._series_learned_roi_for(page, same) == ROI
    assert ExtractPage._series_learned_roi_for(page, other) is None
    assert ExtractPage._series_learned_roi_for(SimpleNamespace(), same) is None


class _FakeDetector:
    def __init__(self, refit: Roi | None) -> None:
        self.calls: list[str] = []
        self._refit = refit

    def refit_series_roi(self, metadata, roi, batch_size, progress_callback, vram_flags=None):
        self.calls.append("refit")
        return self._refit

    def detect_primary_subtitle_roi(self, metadata, step_ms, batch_size, progress_callback, vram_flags=None):
        self.calls.append("full")
        return Roi(x=1, y=2, width=3, height=4)


def test_detect_series_roi_falls_back_to_full_detection_in_one_call() -> None:
    from subtitles_extractor.infrastructure.video.ocr_based_auto_roi_detector import OcrBasedAutoRoiDetector

    for refit, expected_calls, expected in (
        (ROI, ["refit"], ROI),
        (None, ["refit", "full"], Roi(x=1, y=2, width=3, height=4)),
    ):
        fake = _FakeDetector(refit)
        assert OcrBasedAutoRoiDetector.detect_series_roi(fake, SimpleNamespace(), ROI) == expected
        assert fake.calls == expected_calls
    fake = _FakeDetector(ROI)
    OcrBasedAutoRoiDetector.detect_series_roi(fake, SimpleNamespace(), None)        # tập đầu: dò đầy đủ ngay
    assert fake.calls == ["full"]
