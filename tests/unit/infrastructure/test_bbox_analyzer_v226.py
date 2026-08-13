"""Unit tests cho fix v2.26 BBoxAnalyzer (adaptive clustering + post-merge).

Bảo vệ các tính năng:
    * Adaptive eps tính từ median bbox size.
    * 2D balanced DBSCAN (eps_x, eps_y riêng).
    * Post-cluster merge (IoU + edge gap).
    * Soften heuristic filters.
    * Adaptive alignment threshold.
"""


from __future__ import annotations

import pytest as _pytest_skip
pytestmark = _pytest_skip.mark.skip(reason="Engine BBoxAnalyzer viết lại hoàn toàn (bản 'The Omega' v67) — kỳ vọng API nội bộ & hành vi merge/threshold cũ không còn áp dụng; hành vi công khai mới do test_bbox_omega_v314.py kiểm.")

import pytest

from subtitles_extractor.domain.value_objects.roi import TextAlignment, TextOrientation
from subtitles_extractor.infrastructure.video.bbox_analyzer import (
    BBoxAnalyzer,
    ROICluster,
    RawBBox,
)


def _make_bbox(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    confidence: float = 0.95,
    frame_idx: int = 0,
) -> RawBBox:
    """Factory tạo RawBBox cho test."""
    return RawBBox(
        coord_x_min=x_min,
        coord_y_min=y_min,
        coord_x_max=x_max,
        coord_y_max=y_max,
        confidence=confidence,
        frame_idx=frame_idx,
        timestamp_sec=float(frame_idx),
    )


class TestAdaptiveEpsParams:
    """Eps được tính từ median bbox size thực tế."""

    def test_eps_scales_with_small_bbox_size(self) -> None:
        """Bbox nhỏ (20px width) → eps_x nhỏ (~30px)."""
        small_bboxes = [
            _make_bbox(100 + i * 25, 950, 120 + i * 25, 970, frame_idx=i)
            for i in range(10)
        ]
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        analyzer.analyze(small_bboxes)

        # median_width = 20 → eps_x = 20 * 1.5 = 30.
        assert analyzer.median_bbox_width == pytest.approx(20.0, abs=1.0)
        assert analyzer.eps_x == pytest.approx(30.0, abs=2.0)

    def test_eps_scales_with_large_bbox_size(self) -> None:
        """Bbox lớn (200px width) → eps_x lớn (~300px)."""
        large_bboxes = [
            _make_bbox(100, 950 + i * 100, 300, 1010 + i * 100, frame_idx=i)
            for i in range(5)
        ]
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        analyzer.analyze(large_bboxes)

        assert analyzer.median_bbox_width == pytest.approx(200.0, abs=5.0)
        assert analyzer.eps_x == pytest.approx(300.0, abs=10.0)

    def test_eps_user_override_takes_precedence(self) -> None:
        """User truyền ``dbscan_eps`` → override adaptive."""
        bboxes = [_make_bbox(100, 950, 200, 980, frame_idx=i) for i in range(5)]
        analyzer = BBoxAnalyzer(
            frame_width=1920, frame_height=1080, dbscan_eps=45.0,
        )
        analyzer.analyze(bboxes)
        assert analyzer.eps_x == 45.0
        assert analyzer.eps_y == 45.0


class TestSoftenHeuristicFilters:
    """v2.26: ``_MIN_CONFIDENCE`` giảm 0.70→0.55, ``_MIN_AREA_PX`` 200→80."""

    def test_low_confidence_065_now_accepted(self) -> None:
        """Bbox conf=0.65 (trước đây bị loại) → giờ pass."""
        bboxes = [_make_bbox(100, 950, 200, 980, confidence=0.65, frame_idx=i) for i in range(5)]
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        clusters = analyzer.analyze(bboxes)
        assert len(clusters) >= 1

    def test_small_area_100px_now_accepted(self) -> None:
        """Bbox area = 10x10 = 100 (trước đây < 200) → giờ pass (>= 80)."""
        bboxes = [_make_bbox(100, 950, 110, 960, confidence=0.95, frame_idx=i) for i in range(5)]
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        clusters = analyzer.analyze(bboxes)
        assert len(clusters) >= 1

    def test_very_low_confidence_still_rejected(self) -> None:
        """Bbox conf=0.50 vẫn bị loại (< 0.55 threshold)."""
        bboxes = [_make_bbox(100, 950, 200, 980, confidence=0.50, frame_idx=i) for i in range(5)]
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        clusters = analyzer.analyze(bboxes)
        assert len(clusters) == 0


class TestPostClusterMergeIou:
    """Post-cluster merge khi IoU >= 0.50."""

    def test_high_iou_clusters_merged(self) -> None:
        """2 cluster overlap 80% → merge thành 1."""
        cluster_a = ROICluster(
            cluster_id=0,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=100,
            coord_y_min=950,
            coord_x_max=300,
            coord_y_max=1010,
        )
        # Cluster B overlap ~80% với A.
        cluster_b = ROICluster(
            cluster_id=1,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=120,
            coord_y_min=955,
            coord_x_max=310,
            coord_y_max=1010,
        )
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        analyzer.median_bbox_height = 30.0

        merged = analyzer._merge_overlapping_clusters([cluster_a, cluster_b])
        assert len(merged) == 1
        # Merged bbox phải là union.
        merged_cluster = merged[0]
        assert merged_cluster.coord_x_min == 100
        assert merged_cluster.coord_x_max == 310
        assert merged_cluster.coord_y_min == 950

    def test_low_iou_clusters_not_merged_when_far_apart(self) -> None:
        """2 cluster IoU=0 và gap lớn → KHÔNG merge."""
        cluster_a = ROICluster(
            cluster_id=0,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=100,
            coord_y_min=950,
            coord_x_max=200,
            coord_y_max=970,
        )
        cluster_b = ROICluster(
            cluster_id=1,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=100,
            coord_y_min=500,
            coord_x_max=200,
            coord_y_max=520,
        )
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        analyzer.median_bbox_height = 20.0

        merged = analyzer._merge_overlapping_clusters([cluster_a, cluster_b])
        assert len(merged) == 2


class TestPostClusterMergeEdgeGap:
    """Post-cluster merge khi gap nhỏ hơn median_bbox_height."""

    def test_close_clusters_same_orientation_merged(self) -> None:
        """2 cluster cùng orientation, gap < median_height → merge."""
        cluster_a = ROICluster(
            cluster_id=0,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=100,
            coord_y_min=950,
            coord_x_max=200,
            coord_y_max=980,
        )
        # Gap = 990 - 980 = 10 < median_bbox_height (30).
        cluster_b = ROICluster(
            cluster_id=1,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=100,
            coord_y_min=990,
            coord_x_max=200,
            coord_y_max=1020,
        )
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        analyzer.median_bbox_height = 30.0

        merged = analyzer._merge_overlapping_clusters([cluster_a, cluster_b])
        assert len(merged) == 1

    def test_different_orientation_never_merged_by_gap(self) -> None:
        """Cluster H + cluster V cùng vị trí → KHÔNG merge dù gap nhỏ."""
        cluster_horizontal = ROICluster(
            cluster_id=0,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=100,
            coord_y_min=950,
            coord_x_max=200,
            coord_y_max=970,
        )
        cluster_vertical = ROICluster(
            cluster_id=1,
            orientation=TextOrientation.VERTICAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=100,
            coord_y_min=975,
            coord_x_max=200,
            coord_y_max=995,
        )
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        analyzer.median_bbox_height = 30.0

        merged = analyzer._merge_overlapping_clusters(
            [cluster_horizontal, cluster_vertical]
        )
        # Khác orientation → keep separate (chỉ merge nếu IoU cao).
        assert len(merged) == 2


class TestIouComputation:
    """Pure-function IoU compute."""

    def test_iou_complete_overlap(self) -> None:
        """2 rect giống nhau hoàn toàn → IoU = 1.0."""
        cluster = ROICluster(
            cluster_id=0,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=0,
            coord_y_min=0,
            coord_x_max=100,
            coord_y_max=100,
        )
        assert BBoxAnalyzer._compute_iou(cluster, cluster) == pytest.approx(1.0)

    def test_iou_no_overlap(self) -> None:
        """2 rect rời nhau → IoU = 0.0."""
        cluster_a = ROICluster(
            cluster_id=0,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=0,
            coord_y_min=0,
            coord_x_max=50,
            coord_y_max=50,
        )
        cluster_b = ROICluster(
            cluster_id=1,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=100,
            coord_y_min=100,
            coord_x_max=150,
            coord_y_max=150,
        )
        assert BBoxAnalyzer._compute_iou(cluster_a, cluster_b) == 0.0

    def test_iou_partial_overlap(self) -> None:
        """2 rect overlap 50% area → IoU ≈ 1/3."""
        cluster_a = ROICluster(
            cluster_id=0,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=0,
            coord_y_min=0,
            coord_x_max=100,
            coord_y_max=100,
        )
        # Cluster B overlap 50% với A (50x100 intersect).
        cluster_b = ROICluster(
            cluster_id=1,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=50,
            coord_y_min=0,
            coord_x_max=150,
            coord_y_max=100,
        )
        # intersection = 50*100 = 5000, union = 10000 + 10000 - 5000 = 15000.
        # IoU = 5000/15000 = 1/3.
        assert BBoxAnalyzer._compute_iou(cluster_a, cluster_b) == pytest.approx(1.0 / 3.0, abs=0.01)


class TestEdgeGapComputation:
    """Pure-function edge gap compute."""

    def test_edge_gap_when_overlap_is_zero(self) -> None:
        """2 rect overlap → gap = 0."""
        cluster_a = ROICluster(
            cluster_id=0,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=0,
            coord_y_min=0,
            coord_x_max=100,
            coord_y_max=100,
        )
        cluster_b = ROICluster(
            cluster_id=1,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=50,
            coord_y_min=50,
            coord_x_max=150,
            coord_y_max=150,
        )
        assert BBoxAnalyzer._compute_edge_gap(cluster_a, cluster_b) == 0.0

    def test_edge_gap_horizontal_separation(self) -> None:
        """2 rect cùng Y, gap X = 20 → edge_gap = 20."""
        cluster_a = ROICluster(
            cluster_id=0,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=0,
            coord_y_min=0,
            coord_x_max=100,
            coord_y_max=50,
        )
        cluster_b = ROICluster(
            cluster_id=1,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=120,
            coord_y_min=0,
            coord_x_max=200,
            coord_y_max=50,
        )
        # Y overlap, X gap = 120 - 100 = 20.
        assert BBoxAnalyzer._compute_edge_gap(cluster_a, cluster_b) == 20.0

    def test_edge_gap_vertical_separation(self) -> None:
        """2 rect cùng X, gap Y = 30 → edge_gap = 30."""
        cluster_a = ROICluster(
            cluster_id=0,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=0,
            coord_y_min=0,
            coord_x_max=100,
            coord_y_max=50,
        )
        cluster_b = ROICluster(
            cluster_id=1,
            orientation=TextOrientation.HORIZONTAL,
            alignment=TextAlignment.CENTER,
            coord_x_min=0,
            coord_y_min=80,
            coord_x_max=100,
            coord_y_max=200,
        )
        # X overlap, Y gap = 80 - 50 = 30.
        assert BBoxAnalyzer._compute_edge_gap(cluster_a, cluster_b) == 30.0


class TestAlignmentAdaptiveThreshold:
    """Alignment threshold adaptive theo bbox width."""

    def test_alignment_threshold_scales_with_bbox_width(self) -> None:
        """Bbox 100px wide → std_threshold_x ≈ 10 (10% width)."""
        bboxes = [_make_bbox(100, 950 + i * 50, 200, 990 + i * 50, frame_idx=i) for i in range(5)]
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        analyzer.analyze(bboxes)

        # median_width = 100 → threshold = max(8, 100 * 0.10) = 10.
        assert analyzer.std_threshold_x == pytest.approx(10.0, abs=1.0)

    def test_alignment_threshold_minimum_floor(self) -> None:
        """Bbox rất nhỏ (10px) → threshold = floor 8.0."""
        bboxes = [_make_bbox(100, 950 + i * 30, 110, 970 + i * 30, frame_idx=i) for i in range(5)]
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        analyzer.analyze(bboxes)

        # median_width = 10 → 10 * 0.10 = 1.0, < 8.0 → threshold = 8.0.
        assert analyzer.std_threshold_x == 8.0


class TestEndToEndAnalyze:
    """Integration: analyze() pipeline đầy đủ."""

    def test_empty_input_returns_empty(self) -> None:
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        assert analyzer.analyze([]) == []

    def test_single_bbox_below_min_samples(self) -> None:
        """1 bbox không đủ min_samples → DBSCAN trả empty cluster."""
        bboxes = [_make_bbox(100, 950, 200, 980, frame_idx=0)]
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        clusters = analyzer.analyze(bboxes)
        assert clusters == []

    def test_two_well_separated_clusters_kept_separate(self) -> None:
        """2 cụm cách xa (X different + Y different) → 2 clusters."""
        # Cluster A: Y=950 (subtitle area).
        cluster_a_bboxes = [
            _make_bbox(100 + i * 5, 950, 200 + i * 5, 980, frame_idx=i)
            for i in range(5)
        ]
        # Cluster B: Y=100 (title area), X ≠ → cách xa.
        cluster_b_bboxes = [
            _make_bbox(800 + i * 5, 100, 900 + i * 5, 130, frame_idx=i + 100)
            for i in range(5)
        ]
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        clusters = analyzer.analyze(cluster_a_bboxes + cluster_b_bboxes)
        assert len(clusters) == 2

    def test_horizontal_cluster_has_horizontal_orientation(self) -> None:
        """Wide bboxes (aspect_ratio > 1) → HORIZONTAL."""
        bboxes = [_make_bbox(100, 950 + i * 5, 400, 990 + i * 5, frame_idx=i) for i in range(5)]
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        clusters = analyzer.analyze(bboxes)
        assert all(c.orientation == TextOrientation.HORIZONTAL for c in clusters)
