"""Test [#1 v3.18]: ``select_subtitle_roi_smart`` ưu tiên TUYỆT ĐỐI lõi AI.

Khi cụm do ``BBoxAnalyzer`` trả về sạch (không mega-cluster), hàm phải dùng NGAY
cụm đó — không rơi xuống thuật toán cắt lớp dự phòng (vốn dễ bị nhiễu/video dọc
đánh lừa). Chỉ khi AI thất bại (không cụm / cụm vẫn là mega-cluster) mới dùng dự
phòng.
"""

from __future__ import annotations

from dataclasses import dataclass

from subtitles_extractor.infrastructure.video.roi_selection import (
    _MEGA_CLUSTER_HEIGHT_RATIO,
    select_subtitle_roi_smart,
)


@dataclass
class _FakeCluster:
    coord_x_min: float
    coord_y_min: float
    coord_x_max: float
    coord_y_max: float
    frame_count: int
    bbox_count: int


_FRAME_W = 720
_FRAME_H = 1280


def test_clean_ai_cluster_is_used_directly() -> None:
    # Cụm AI sạch ở đáy khung (h_ratio ~0.07) → dùng ngay, không cần raw boxes.
    clean = _FakeCluster(
        coord_x_min=60, coord_y_min=1100, coord_x_max=660, coord_y_max=1190,
        frame_count=500, bbox_count=520,
    )
    roi = select_subtitle_roi_smart([clean], boxes=[], frame_width=_FRAME_W, frame_height=_FRAME_H)
    assert roi is not None
    assert roi.y == 1100 and roi.height == 90
    # Xác nhận đây là cụm AI (không phải mega-cluster).
    assert (clean.coord_y_max - clean.coord_y_min) / _FRAME_H <= _MEGA_CLUSTER_HEIGHT_RATIO


def test_mega_cluster_ai_result_is_rejected_then_falls_back() -> None:
    # Cụm AI là mega-cluster (nuốt cả khung) → KHÔNG dùng; không có raw boxes nên
    # band-detection cũng rỗng → cùng đường trả lại cụm AI (không bỏ trắng).
    mega = _FakeCluster(
        coord_x_min=0, coord_y_min=0, coord_x_max=720, coord_y_max=1280,
        frame_count=900, bbox_count=4000,
    )
    roi = select_subtitle_roi_smart([mega], boxes=[], frame_width=_FRAME_W, frame_height=_FRAME_H)
    # Không có nguồn dự phòng → trả tạm cụm AI để không bỏ trắng kết quả.
    assert roi is not None
    assert (mega.coord_y_max - mega.coord_y_min) / _FRAME_H > _MEGA_CLUSTER_HEIGHT_RATIO


def test_no_clusters_returns_none_without_boxes() -> None:
    assert select_subtitle_roi_smart([], boxes=[], frame_width=_FRAME_W, frame_height=_FRAME_H) is None
