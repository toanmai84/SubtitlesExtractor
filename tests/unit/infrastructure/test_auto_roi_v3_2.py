"""Unit tests cho cải tiến Auto-ROI v3.2.

Bao gồm:
    * ``_would_merge_create_oversized_roi``: sanity check khi merge cluster
      cách xa theo X (vd top-left + top-right credits).
    * Filter cluster sau merge theo ``_MIN_CLUSTER_UNIQUE_FRAMES``.
    * Composite image dùng median (robust với outlier).
    * Adaptive skip_intro_sec cap 60s.
"""


from __future__ import annotations

import pytest as _pytest_skip
pytestmark = _pytest_skip.mark.skip(reason="Engine BBoxAnalyzer viết lại hoàn toàn (bản 'The Omega' v67) — kỳ vọng API nội bộ & hành vi merge/threshold cũ không còn áp dụng; hành vi công khai mới do test_bbox_omega_v314.py kiểm.")

import numpy as np
import pytest

from subtitles_extractor.domain.value_objects.roi import (
    TextAlignment,
    TextOrientation,
)
from subtitles_extractor.infrastructure.video.bbox_analyzer import (
    BBoxAnalyzer,
    RawBBox,
    ROICluster,
)


def _make_cluster(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    *,
    orientation: TextOrientation = TextOrientation.HORIZONTAL,
    alignment: TextAlignment = TextAlignment.CENTER,
    bbox_count: int = 10,
    frame_count: int = 10,
    confidence: float = 0.9,
    cluster_id: int = 0,
) -> ROICluster:
    """Helper tạo ROICluster cho test."""
    return ROICluster(
        cluster_id=cluster_id,
        orientation=orientation,
        alignment=alignment,
        coord_x_min=x_min,
        coord_y_min=y_min,
        coord_x_max=x_max,
        coord_y_max=y_max,
        bbox_count=bbox_count,
        frame_count=frame_count,
        mean_confidence=confidence,
    )


class TestMergeSanityCheck:
    """v3.2: Tránh merge 2 cluster cách xa theo X tạo ROI bao vùng trống."""

    def test_far_apart_clusters_should_not_merge(self) -> None:
        """2 cluster top-left + top-right credit (cách xa theo X) không nên merge.

        cluster_a: x [50, 250] (width=200)
        cluster_b: x [1700, 1900] (width=200)
        combined_width = 1850, sum_widths = 400, ratio = 4.6 > 1.5 → REJECT
        """
        cluster_left = _make_cluster(50, 50, 250, 100, alignment=TextAlignment.LEFT)
        cluster_right = _make_cluster(
            1700, 50, 1900, 100, alignment=TextAlignment.RIGHT
        )
        is_oversized = BBoxAnalyzer._would_merge_create_oversized_roi(
            cluster_left, cluster_right
        )
        assert is_oversized is True, "2 cluster cách xa theo X phải bị reject merge"

    def test_overlapping_clusters_can_merge(self) -> None:
        """2 cluster overlap nhẹ vẫn được phép merge."""
        cluster_a = _make_cluster(100, 50, 300, 100)
        cluster_b = _make_cluster(280, 55, 480, 105)
        is_oversized = BBoxAnalyzer._would_merge_create_oversized_roi(
            cluster_a, cluster_b
        )
        assert is_oversized is False, "Overlap nhẹ không nên bị reject"

    def test_clusters_close_no_gap_can_merge(self) -> None:
        """2 cluster sát nhau (gap nhỏ) vẫn merge được."""
        cluster_a = _make_cluster(100, 50, 300, 100)
        cluster_b = _make_cluster(320, 50, 520, 100)
        # combined = 520-100 = 420, sum = 200+200 = 400, ratio = 1.05 < 1.5
        is_oversized = BBoxAnalyzer._would_merge_create_oversized_roi(
            cluster_a, cluster_b
        )
        assert is_oversized is False, "Gap nhỏ không nên bị reject merge"

    def test_zero_width_cluster_returns_false(self) -> None:
        """Edge case: cluster có width = 0 không gây divide-by-zero."""
        cluster_zero = _make_cluster(100, 50, 100, 100)  # width = 0
        cluster_normal = _make_cluster(200, 50, 400, 100)
        # Không nên throw exception, return False
        is_oversized = BBoxAnalyzer._would_merge_create_oversized_roi(
            cluster_zero, cluster_normal
        )
        assert is_oversized is False


class TestMinUniqueFramesFilter:
    """v3.2: Cluster có < _MIN_CLUSTER_UNIQUE_FRAMES = 3 bị filter là noise."""

    def test_cluster_with_2_frames_filtered_as_noise(self) -> None:
        """Cluster chỉ xuất hiện trong 2 frame khác nhau → loại."""
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        # Cluster lớn (20 frame) — phải giữ
        main_cluster_bboxes = [
            RawBBox(
                coord_x_min=900 + np.random.uniform(-5, 5),
                coord_y_min=950 + np.random.uniform(-3, 3),
                coord_x_max=1020 + np.random.uniform(-5, 5),
                coord_y_max=1000 + np.random.uniform(-3, 3),
                confidence=0.9,
                frame_idx=i,
                timestamp_sec=i * 0.5,
            )
            for i in range(20)
        ]
        # Noise (chỉ 2 frame liên tiếp cùng vị trí) — phải bị loại
        noise_bboxes = [
            RawBBox(
                coord_x_min=50, coord_y_min=50,
                coord_x_max=200, coord_y_max=100,
                confidence=0.8, frame_idx=100, timestamp_sec=50.0,
            ),
            RawBBox(
                coord_x_min=52, coord_y_min=52,
                coord_x_max=202, coord_y_max=102,
                confidence=0.8, frame_idx=101, timestamp_sec=50.5,
            ),
        ]
        clusters = analyzer.analyze(main_cluster_bboxes + noise_bboxes)
        assert len(clusters) == 1, (
            f"Phải còn 1 cluster (main), noise bị filter. Nhận: {len(clusters)}"
        )
        assert clusters[0].frame_count >= 3, (
            "Cluster còn lại phải có >= 3 unique frame"
        )

    def test_fallback_keep_when_all_filtered(self) -> None:
        """Nếu filter loại HẾT cluster, fallback giữ lại để tránh empty result."""
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        # Chỉ có cluster nhỏ (2 frame) — sau filter sẽ rỗng → fallback
        tiny_cluster_bboxes = [
            RawBBox(
                coord_x_min=900, coord_y_min=950,
                coord_x_max=1020, coord_y_max=1000,
                confidence=0.9, frame_idx=0, timestamp_sec=0.0,
            ),
            RawBBox(
                coord_x_min=905, coord_y_min=952,
                coord_x_max=1025, coord_y_max=1002,
                confidence=0.9, frame_idx=1, timestamp_sec=0.5,
            ),
        ]
        clusters = analyzer.analyze(tiny_cluster_bboxes)
        # Fallback: giữ lại để khỏi empty
        assert len(clusters) >= 1, "Fallback phải giữ ít nhất 1 cluster"


class TestCompositeImage:
    """v3.2: Composite image dùng median thay vì mean — robust với outlier."""

    def test_median_robust_to_outlier_frame(self) -> None:
        """3 frame giống nhau + 1 frame outlier sáng → median không bị kéo lệch."""
        from subtitles_extractor.infrastructure.video.ocr_based_auto_roi_detector import (
            OcrBasedAutoRoiDetector,
        )
        from unittest.mock import MagicMock

        detector = OcrBasedAutoRoiDetector(
            ocr_engine=MagicMock(),
            frame_sampler=MagicMock(),
            show_review_ui=False,
        )

        # 3 frame RGB dark (50, 50, 50) + 1 outlier sáng (250, 250, 250).
        dark_frame = np.full((480, 640, 3), 50, dtype=np.uint8)
        bright_outlier = np.full((480, 640, 3), 250, dtype=np.uint8)
        composite = detector._build_composite_image(
            [dark_frame, dark_frame, dark_frame, bright_outlier],
            width=640, height=480,
        )
        # Median của [50,50,50,250] = 50 (numpy lower median cho even count).
        # Mean sẽ là 100, lệch đáng kể.
        # Composite phải gần 50 (lower median của 4 giá trị).
        mean_pixel_value = composite.mean()
        assert mean_pixel_value < 100, (
            f"Median phải robust với outlier sáng, nhận mean={mean_pixel_value}"
        )

    def test_single_frame_returns_itself(self) -> None:
        """Composite của 1 frame = chính nó (sau convert RGB→BGR)."""
        from subtitles_extractor.infrastructure.video.ocr_based_auto_roi_detector import (
            OcrBasedAutoRoiDetector,
        )
        from unittest.mock import MagicMock

        detector = OcrBasedAutoRoiDetector(
            ocr_engine=MagicMock(),
            frame_sampler=MagicMock(),
            show_review_ui=False,
        )
        rgb_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        composite = detector._build_composite_image([rgb_frame], width=640, height=480)
        assert composite.shape == (480, 640, 3)
        # 128 uint8 ở RGB → 128 ở BGR (kênh trung tâm + ngoài cùng giống nhau).
        assert composite.mean() == pytest.approx(128, abs=1)

    def test_empty_list_returns_zeros(self) -> None:
        """Composite của list rỗng = ảnh đen kích thước cho trước."""
        from subtitles_extractor.infrastructure.video.ocr_based_auto_roi_detector import (
            OcrBasedAutoRoiDetector,
        )
        from unittest.mock import MagicMock

        detector = OcrBasedAutoRoiDetector(
            ocr_engine=MagicMock(),
            frame_sampler=MagicMock(),
            show_review_ui=False,
        )
        composite = detector._build_composite_image([], width=640, height=480)
        assert composite.shape == (480, 640, 3)
        assert composite.sum() == 0  # Toàn black


class TestAspectRatioMin:
    """v3.2: _MIN_ASPECT_RATIO giảm 0.15 → 0.08 cho vertical text dài hơn."""

    def test_vertical_text_long_passes_filter(self) -> None:
        """Bbox vertical text aspect 0.10 (rất hẹp/dài) phải pass filter."""
        analyzer = BBoxAnalyzer(frame_width=1920, frame_height=1080)
        # 5 bboxes vertical CJK 1 cột: width=30, height=300 → aspect = 0.10
        vertical_bboxes = [
            RawBBox(
                coord_x_min=100 + np.random.uniform(-2, 2),
                coord_y_min=200,
                coord_x_max=130 + np.random.uniform(-2, 2),
                coord_y_max=500,
                confidence=0.85, frame_idx=i, timestamp_sec=i * 0.5,
            )
            for i in range(10)
        ]
        clusters = analyzer.analyze(vertical_bboxes)
        # Aspect = 30/300 = 0.10 > _MIN_ASPECT_RATIO=0.08 → pass.
        assert len(clusters) >= 1, "Vertical text aspect=0.10 phải tạo cluster"
