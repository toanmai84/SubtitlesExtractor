"""Test [v3.18.2]: phân loại hướng chữ (orientation) bất biến với jitter OCR.

Lỗi gốc (tái hiện được): orientation quyết định bằng ``var(center_x)`` vs
``var(center_y)`` — khi một khối text đứng yên qua nhiều frame, var chỉ đo NHIỄU
jitter ±2px của OCR, nên cột chữ DỌC 66×300 có thể bị gán ``HORIZONTAL`` hay
``VERTICAL`` tuỳ… seed nhiễu. Hệ quả nhìn thấy: ROI 153×307 của cột chữ dọc
御前太监 hiển thị nhãn "Ngang/Giữa" trong dialog kiểm duyệt.

Fix: orientation theo **median aspect của box thành viên** (dòng ngang w ≫ h,
cột dọc h ≫ w — bất biến jitter); spread không gian (p95−p5 của tâm, phải vượt
kích thước box) chỉ làm tie-break khi box gần vuông; macro aspect là fallback.
"""

from __future__ import annotations

import random

from subtitles_extractor.domain.value_objects.roi import TextOrientation
from subtitles_extractor.infrastructure.video.bbox_analyzer import (
    BBoxAnalyzer,
    RawBBox,
)

_FRAME_W = 1080
_FRAME_H = 1920


def _vertical_column_boxes(seed: int, jitter_x: float, jitter_y: float) -> list[RawBBox]:
    """Cột chữ dọc 66×300 đứng yên 300 frame với jitter OCR cấu hình được."""
    random.seed(seed)
    boxes = []
    for frame in range(300):
        jx = random.uniform(-jitter_x, jitter_x)
        jy = random.uniform(-jitter_y, jitter_y)
        boxes.append(
            RawBBox(
                coord_x_min=620 + jx, coord_y_min=300 + jy,
                coord_x_max=686 + jx, coord_y_max=600 + jy,
                confidence=0.96, frame_idx=frame, timestamp_sec=frame * 0.04,
            )
        )
    return boxes


def _analyze(boxes: list[RawBBox]):
    clusters = BBoxAnalyzer(frame_width=_FRAME_W, frame_height=_FRAME_H, padding=0).analyze(boxes)
    assert clusters
    return max(clusters, key=lambda c: c.frame_count)


class TestOrientationJitterInvariance:
    def test_vertical_column_detected_regardless_of_jitter_direction(self) -> None:
        # Hai cấu hình jitter ĐỐI NGHỊCH phải cho CÙNG kết quả VERTICAL.
        for seed, jx, jy in [(42, 2.0, 1.0), (7, 1.0, 2.0)]:
            primary = _analyze(_vertical_column_boxes(seed, jx, jy))
            assert primary.orientation == TextOrientation.VERTICAL, (
                f"Cột dọc bị gán {primary.orientation.name} (seed={seed})"
            )

    def test_horizontal_line_detected(self) -> None:
        boxes = [
            RawBBox(coord_x_min=300, coord_y_min=1700, coord_x_max=780, coord_y_max=1760,
                    confidence=0.97, frame_idx=f, timestamp_sec=f * 0.04)
            for f in range(300)
        ]
        assert _analyze(boxes).orientation == TextOrientation.HORIZONTAL

    def test_square_glyphs_stacked_vertically_use_spread_tiebreak(self) -> None:
        # 5 ký tự vuông 60×60 xếp dọc thành cột — box vuông nên aspect bất phân,
        # spread dọc (≫ kích thước box) phải quyết định VERTICAL.
        boxes = []
        for frame in range(200):
            for i in range(5):
                y = 300 + i * 66
                boxes.append(
                    RawBBox(coord_x_min=620, coord_y_min=y, coord_x_max=680, coord_y_max=y + 60,
                            confidence=0.95, frame_idx=frame, timestamp_sec=frame * 0.04)
                )
        assert _analyze(boxes).orientation == TextOrientation.VERTICAL
