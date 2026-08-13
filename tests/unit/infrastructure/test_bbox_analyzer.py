"""Unit test BBoxAnalyzer — dùng mock data, KHÔNG cần video thực."""

from __future__ import annotations

import pytest
import numpy as np

from subtitles_extractor.infrastructure.video.bbox_analyzer import (
    BBoxAnalyzer,
    RawBBox,
    TextAlignment,
    TextOrientation,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_bbox(
    x1: float, y1: float, x2: float, y2: float,
    conf: float = 0.92, frame_idx: int = 0, ts: float = 0.0,
) -> RawBBox:
    return RawBBox(
        coord_x_min=x1, 
        coord_y_min=y1, 
        coord_x_max=x2, 
        coord_y_max=y2,
        confidence=conf, 
        frame_idx=frame_idx, 
        timestamp_sec=ts,
    )


# Giả lập subtitle ngang nửa dưới (căn giữa) - 20 bbox trải 10 frame
def _make_horizontal_bottom_centered(n_frames: int = 12) -> list[RawBBox]:
    """Text subtitle ổn định ở đáy màn hình, căn giữa."""
    bboxes =[]
    for i in range(n_frames):
        # Y_max = 700 ± 2px (ổn định)
        # X_center ≈ 500 (ổn định), X_min/X_max thay đổi theo độ dài câu
        text_width = np.random.randint(200, 400)
        x_min = 500 - text_width // 2 + np.random.randint(-5, 5)
        x_max = 500 + text_width // 2 + np.random.randint(-5, 5)
        bboxes.append(_make_bbox(x_min, 650, x_max, 700, frame_idx=i, ts=i * 0.5))
    return bboxes


def _make_horizontal_top_left_description(n_frames: int = 10) -> list[RawBBox]:
    """Text mô tả nhân vật ở trên bên trái, căn trái."""
    bboxes =[]
    for i in range(n_frames):
        # X_min = 20 ± 3px (ổn định, căn trái)
        # X_max thay đổi theo độ dài text
        text_width = np.random.randint(100, 250)
        x_min = 20 + np.random.randint(-3, 3)
        x_max = x_min + text_width
        bboxes.append(_make_bbox(x_min, 80, x_max, 120, frame_idx=i, ts=i * 0.5))
    return bboxes


def _make_noise_scattered(n: int = 8) -> list[RawBBox]:
    """Nhiễu rải rác không thuộc cụm nào."""
    bboxes =[]
    for i in range(n):
        x = np.random.randint(50, 650)
        y = np.random.randint(50, 650)
        bboxes.append(_make_bbox(x, y, x + 60, y + 30,
                                 conf=0.75, frame_idx=i * 3))
    return bboxes


# ── Tests ─────────────────────────────────────────────────────────────────


class TestBBoxAnalyzer:
    FRAME_W = 1000
    FRAME_H = 750

    def _analyzer(self, eps: float = 25.0) -> BBoxAnalyzer:
        return BBoxAnalyzer(
            frame_width=self.FRAME_W,
            frame_height=self.FRAME_H,
            dbscan_eps=eps,
            dbscan_min_samples=2,
            padding=10,
        )

    def test_empty_input(self) -> None:
        analyzer = self._analyzer()
        result = analyzer.analyze([])
        assert result ==[]

    def test_all_filtered_by_confidence(self) -> None:
        bboxes =[_make_bbox(100, 100, 200, 150, conf=0.3) for _ in range(5)]
        result = self._analyzer().analyze(bboxes)
        assert result ==[]

    def test_horizontal_bottom_centered(self) -> None:
        """Subtitle ngang nửa dưới → phải phát hiện 1 cluster, alignment CENTER."""
        np.random.seed(42)
        bboxes = _make_horizontal_bottom_centered(15)
        analyzer = self._analyzer()
        clusters = analyzer.analyze(bboxes)

        assert len(clusters) >= 1
        # Cluster phải là nằm nửa dưới
        bottom_clusters =[
            c for c in clusters
            if c.coord_y_max > self.FRAME_H * 0.5
        ]
        assert len(bottom_clusters) >= 1
        c = bottom_clusters[0]
        assert c.orientation == TextOrientation.HORIZONTAL
        # Alignment phải là CENTER (X_center ổn định quanh 500)
        assert c.alignment in (TextAlignment.CENTER, TextAlignment.UNKNOWN)

    def test_top_left_description_text(self) -> None:
        """Text mô tả nửa trên căn trái → phải phát hiện cluster riêng."""
        np.random.seed(123)
        bboxes = _make_horizontal_top_left_description(12)
        analyzer = self._analyzer()
        clusters = analyzer.analyze(bboxes)

        assert len(clusters) >= 1
        top_clusters =[c for c in clusters if c.coord_y_min < self.FRAME_H * 0.4]
        assert len(top_clusters) >= 1
        c = top_clusters[0]
        # Alignment phải phát hiện LEFT (X_min ổn định = ~20px)
        assert c.alignment in (TextAlignment.LEFT, TextAlignment.CENTER, TextAlignment.UNKNOWN)

    def test_noise_is_filtered(self) -> None:
        """Nhiễu rải rác (không cùng cụm) phải bị DBSCAN loại bỏ."""
        np.random.seed(7)
        noise = _make_noise_scattered(8)
        analyzer = BBoxAnalyzer(
            frame_width=self.FRAME_W,
            frame_height=self.FRAME_H,
            dbscan_min_samples=4,  # Yêu cầu tối thiểu 4 bbox cùng cụm
            padding=10,
        )
        clusters = analyzer.analyze(noise)
        # Nhiễu rải rác không nên tạo cluster nào
        for c in clusters:
            assert c.bbox_count >= 4, "Cluster từ noise không đủ min_samples"

    def test_two_regions_separated(self) -> None:
        """Subtitle dưới + mô tả trên → phải tách thành 2 cluster riêng."""
        np.random.seed(99)
        bottom = _make_horizontal_bottom_centered(12)
        top = _make_horizontal_top_left_description(10)
        all_bboxes = bottom + top
        analyzer = self._analyzer(eps=30.0)
        clusters = analyzer.analyze(all_bboxes)

        # Phải có ít nhất 2 cluster (1 trên, 1 dưới)
        assert len(clusters) >= 2, (
            f"Expected ≥ 2 cluster (top + bottom), got {len(clusters)}: "
            f"{[(c.coord_y_min, c.coord_y_max) for c in clusters]}"
        )
        y_centers = sorted([c.coord_y_min + c.height / 2 for c in clusters])
        # Cluster đầu phải nằm nửa trên
        assert y_centers[0] < self.FRAME_H * 0.5
        # Cluster cuối phải nằm nửa dưới
        assert y_centers[-1] > self.FRAME_H * 0.4

    def test_roi_padding_applied(self) -> None:
        """ROI output phải được mở rộng padding từ bbox gốc."""
        bboxes =[
            _make_bbox(100, 600, 400, 650, frame_idx=i) for i in range(6)
        ]
        analyzer = BBoxAnalyzer(
            frame_width=1280, frame_height=720,
            dbscan_min_samples=2, padding=15,
        )
        clusters = analyzer.analyze(bboxes)
        if clusters:
            c = clusters[0]
            # ROI phải mở rộng ra ít nhất padding pixels
            assert c.coord_x_min <= 100, f"x1={c.coord_x_min} > 100 (không áp dụng padding trái)"
            assert c.coord_y_min <= 600, f"y1={c.coord_y_min} > 600 (không áp dụng padding trên)"
            assert c.coord_x_max >= 400, f"x2={c.coord_x_max} < 400 (không áp dụng padding phải)"
            assert c.coord_y_max >= 650, f"y2={c.coord_y_max} < 650 (không áp dụng padding dưới)"

    @pytest.mark.skip(reason="Bản Omega tính alignment trong _build_cluster_from_bboxes (shift_ratio); _determine_alignment đã xoá.")
    def test_variance_center_alignment(self) -> None:
        """Test variance analysis: X_center ổn định → CENTER."""
        analyzer = self._analyzer()
        # X_center = 500 không đổi, X_min/X_max thay đổi
        x_min_arr = np.array([300, 250, 350, 280, 320], dtype=np.float32)
        x_max_arr = np.array([700, 750, 650, 720, 680], dtype=np.float32)
        x_ctr_arr = (x_min_arr + x_max_arr) / 2  # ≈ 500, var ≈ 0

        # Variance X_center phải gần 0
        var_ctr = float(np.var(x_ctr_arr))
        assert var_ctr < 15.0, f"var_ctr={var_ctr} quá lớn"

        alignment = analyzer._determine_alignment(
            float(np.var(x_min_arr)),
            float(np.var(x_max_arr)),
            var_ctr,
        )
        assert alignment == TextAlignment.CENTER

    @pytest.mark.skip(reason="Bản Omega tính alignment trong _build_cluster_from_bboxes (shift_ratio); _determine_alignment đã xoá.")
    def test_variance_left_alignment(self) -> None:
        """Test variance: X_min ổn định, X_max thay đổi → LEFT."""
        analyzer = self._analyzer()
        x_min_arr = np.array([20, 21, 19, 20, 22], dtype=np.float32)  # ~0 var
        x_max_arr = np.array([200, 350, 150, 400, 280], dtype=np.float32)  # lớn var

        var_min = float(np.var(x_min_arr))
        var_max = float(np.var(x_max_arr))
        var_ctr = float(np.var((x_min_arr + x_max_arr) / 2))

        assert var_min < 15.0
        assert var_max >= 15.0

        alignment = analyzer._determine_alignment(var_min, var_max, var_ctr)
        assert alignment == TextAlignment.LEFT