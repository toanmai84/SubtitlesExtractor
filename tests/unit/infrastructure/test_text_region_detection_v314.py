"""Test engine phát hiện đa vùng text (tái thiết kế ROI tự động)."""

from __future__ import annotations

from subtitles_extractor.infrastructure.video.bbox_analyzer import RawBBox
from subtitles_extractor.infrastructure.video.text_region_detection import (
    ROLE_PRIMARY,
    detect_text_regions,
)

W, H = 720, 1280


def _box(xmin, ymin, xmax, ymax, fi, conf=0.95):
    return RawBBox(xmin, ymin, xmax, ymax, conf, fi, fi * 0.1)


def _subtitle_band(n_frames, y0, y1, fi_offset=0):
    # Dải phụ đề: cùng vị trí, đổi nội dung qua nhiều frame.
    return [_box(150, y0, 570, y1, fi_offset + f) for f in range(n_frames)]


class TestDetectTextRegions:
    def test_empty(self) -> None:
        assert detect_text_regions([], W, H) == []

    def test_single_bottom_region_is_primary(self) -> None:
        boxes = _subtitle_band(400, 980, 1030)
        regions = detect_text_regions(boxes, W, H)
        assert len(regions) == 1
        assert regions[0].role == ROLE_PRIMARY
        cy = (regions[0].roi.y + regions[0].roi.height / 2) / H
        assert cy > 0.6

    def test_region_is_snug_not_full_frame(self) -> None:
        boxes = _subtitle_band(400, 980, 1030)
        roi = detect_text_regions(boxes, W, H)[0].roi
        # "Vừa đủ": chiều cao nhỏ so với khung, không nuốt cả màn hình.
        assert roi.height < H * 0.28
        # X trimmed bằng percentile → không tràn cả khung.
        assert roi.width <= W

    def test_two_regions_primary_is_denser_bottom(self) -> None:
        # Dải đáy dày (phụ đề chính) + dải trên thưa hơn (thông tin phụ).
        boxes = _subtitle_band(500, 980, 1030) + _subtitle_band(150, 200, 250, fi_offset=2000)
        regions = detect_text_regions(boxes, W, H)
        assert len(regions) == 2
        primary = next(r for r in regions if r.role == ROLE_PRIMARY)
        cy = (primary.roi.y + primary.roi.height / 2) / H
        assert cy > 0.6, "Vùng đáy dày phải là primary"
        # Sắp theo Y: vùng trên đứng trước.
        assert regions[0].roi.y < regions[1].roi.y

    def test_resists_dense_noise_mega_cluster(self) -> None:
        # Phụ đề đáy ổn định + nhiễu cảnh rải khắp khung (mỗi vị trí ít frame).
        boxes = _subtitle_band(600, 980, 1030)
        for f in range(40):
            for y in range(100, 900, 80):
                boxes.append(_box(50, y, 250, y + 40, 5000 + f))
        regions = detect_text_regions(boxes, W, H)
        primary = next(r for r in regions if r.role == ROLE_PRIMARY)
        cy = (primary.roi.y + primary.roi.height / 2) / H
        assert cy > 0.6, "Phải bám dải phụ đề đáy, không bị nhiễu kéo lên"
        assert primary.roi.height < H * 0.28
