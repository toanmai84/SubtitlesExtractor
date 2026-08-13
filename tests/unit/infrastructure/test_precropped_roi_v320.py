"""Test [v3.20] xác nhận video ĐÃ CROP sẵn về dải phụ đề không bị xử lý sai.

Điều tra "known issue mega-cluster" của ``haomen_roi_subtitle`` cho thấy đó là
DƯƠNG TÍNH GIẢ: video này đã được cắt sẵn về dải phụ đề (khung rất thấp, vd
1088×144), nên cluster phủ ~0.59× chiều cao khung là ĐÚNG (toàn khung là vùng
phụ đề), không phải mega-cluster nuốt cảnh. Test khoá lại hành vi đúng này.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.video.bbox_analyzer import (
    BBoxAnalyzer,
    RawBBox,
)


def _box(x0: float, y0: float, x1: float, y1: float, frame: int) -> RawBBox:
    return RawBBox(coord_x_min=x0, coord_y_min=y0, coord_x_max=x1, coord_y_max=y1,
                   confidence=0.96, frame_idx=frame, timestamp_sec=frame * 0.04)


def test_precropped_subtitle_strip_roi_tracks_text() -> None:
    # Khung đã crop về dải phụ đề: 1088×144. Dòng phụ đề chiếm phần lớn chiều cao.
    frame_w, frame_h = 1088, 144
    boxes = [_box(120, 40, 960, 100, f) for f in range(2000)]
    clusters = BBoxAnalyzer(frame_width=frame_w, frame_height=frame_h, padding=0).analyze(boxes)
    assert clusters
    primary = max(clusters, key=lambda c: c.frame_count)
    # ROI phải bám đúng dải text (y≈40..100), KHÔNG bị co về một dải hẹp vô lý
    # cũng KHÔNG phình quá biên text.
    assert primary.coord_y_min <= 45
    assert primary.coord_y_max >= 95
    # Trên khung thấp, tỉ lệ chiều cao ROI/khung lớn là HỢP LỆ (không gán "mega").
    height_ratio = (primary.coord_y_max - primary.coord_y_min) / frame_h
    assert height_ratio >= 0.35  # bám text thật, không bị band-refiner cắt nhầm
