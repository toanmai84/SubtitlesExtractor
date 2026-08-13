"""Test [v3.18.1]: biên X bền vững + đối xứng hoá ROI cho phụ đề căn giữa.

Lỗi gốc: biên X của cluster lấy ``min/max`` TUYỆT ĐỐI của box thành viên — chỉ
1-2 box nhiễu (logo, text cảnh lọt vào dải Y, OCR rác conf ~0.4) là đủ kéo lệch
một phía, khiến ROI của phụ đề CĂN GIỮA bị lệch trái/phải (đo được tới ±4.7%W).

Fix: (1) biên trục-phụ theo order-statistic thứ k (``_BOUND_SUPPORT_BOXES=6``) —
biên chỉ mở tới nơi có ≥ k box hỗ trợ; (2) khi ``alignment=CENTER``, đối xứng hoá
quanh median tâm box (chỉ mở rộng phía hẹp — không bao giờ cắt chữ).
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.video.bbox_analyzer import (
    BBoxAnalyzer,
    RawBBox,
)

_FRAME_W = 720
_FRAME_H = 1280


def _box(x0: float, y0: float, x1: float, y1: float, frame: int, conf: float = 0.97) -> RawBBox:
    return RawBBox(
        coord_x_min=x0, coord_y_min=y0, coord_x_max=x1, coord_y_max=y1,
        confidence=conf, frame_idx=frame, timestamp_sec=frame * 0.04,
    )


def _centered_subtitle_boxes(n_frames: int = 400) -> list[RawBBox]:
    """Phụ đề căn giữa quanh x=360: xen kẽ câu dài (120..600) và câu ngắn (260..460)."""
    boxes = []
    for f in range(n_frames):
        if (f // 40) % 2 == 0:
            boxes.append(_box(120, 1100, 600, 1160, f))   # câu dài, tâm 360
        else:
            boxes.append(_box(260, 1100, 460, 1160, f))   # câu ngắn, tâm 360
    return boxes


class TestRobustCenteredRoi:
    def test_single_noise_box_cannot_skew_roi(self) -> None:
        boxes = _centered_subtitle_boxes()
        # 2 box nhiễu conf thấp lọt vào dải Y, kéo về mép PHẢI (mô phỏng file 03).
        boxes.append(_box(451, 1105, 720, 1158, 9001, conf=0.48))
        boxes.append(_box(460, 1105, 718, 1158, 9002, conf=0.48))
        clusters = BBoxAnalyzer(frame_width=_FRAME_W, frame_height=_FRAME_H, padding=0).analyze(boxes)
        primary = max(clusters, key=lambda c: c.frame_count)
        roi_center = (primary.coord_x_min + primary.coord_x_max) / 2.0
        # Tâm ROI phải bám tâm phụ đề thật (360) — không bị nhiễu kéo lệch phải.
        assert abs(roi_center - 360) <= 8, f"ROI lệch tâm: {roi_center} (kỳ vọng ~360)"
        # Biên phải không bị kéo ra mép bởi 2 box nhiễu.
        assert primary.coord_x_max <= 620

    def test_clean_centered_subtitle_stays_symmetric(self) -> None:
        boxes = _centered_subtitle_boxes()
        clusters = BBoxAnalyzer(frame_width=_FRAME_W, frame_height=_FRAME_H, padding=0).analyze(boxes)
        primary = max(clusters, key=lambda c: c.frame_count)
        roi_center = (primary.coord_x_min + primary.coord_x_max) / 2.0
        assert abs(roi_center - 360) <= 5
        # Câu dài thật (120..600, 200 box hỗ trợ) KHÔNG bị order-statistic cắt.
        assert primary.coord_x_min <= 126 and primary.coord_x_max >= 594

    def test_robust_axis_bounds_drops_lone_outlier(self) -> None:
        starts = [100.0] * 50 + [2.0]          # 1 outlier trái
        ends = [600.0] * 50 + [719.0]          # 1 outlier phải
        low, high = BBoxAnalyzer._robust_axis_bounds(starts, ends)
        assert low == 100.0 and high == 600.0

    def test_robust_axis_bounds_keeps_supported_edge(self) -> None:
        # Biên 80 có 10 box hỗ trợ (≥ k=6) → phải GIỮ, không được cắt.
        starts = [80.0] * 10 + [200.0] * 90
        ends = [640.0] * 10 + [520.0] * 90
        low, high = BBoxAnalyzer._robust_axis_bounds(starts, ends)
        assert low == 80.0 and high == 640.0

    def test_symmetrize_only_expands_never_shrinks(self) -> None:
        # Biên lệch trái (100..500) quanh tâm 360 → mở phía trái-? half=max(260,140)=260
        low, high = BBoxAnalyzer._symmetrize_around_center(100.0, 500.0, [360.0], _FRAME_W)
        assert low == 100.0 and high == 620.0   # chỉ MỞ RỘNG phía hẹp
        assert (low + high) / 2.0 == 360.0
