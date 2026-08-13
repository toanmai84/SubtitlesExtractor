"""Unit tests cho fix v2.25 density-aware Y-position filter.

Mỗi test bảo vệ một khía cạnh của thuật toán 3-pass mới trong
``filter_y_position_outliers``:
    * Pass 1: cluster discovery với histogram bins.
    * Pass 2: density-based filter (giữ box trong dense cluster, drop ngoài).
    * Pass 3: fallback MAD (khi không phát hiện cụm).

Test data được thiết kế để mô phỏng các scenarios thực tế:
    * Multi-mode distribution (test1): 2+ cụm Y với density rất chênh lệch.
    * Single-mode distribution (file 1): 1 cụm chính + một vài outliers.
    * Sparse data (< 6 boxes): fallback về behavior gốc (keep all).
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.outlier_detection import (
    filter_y_position_outliers,
)


class TestSparseDataReturnsAllKeep:
    """Khi số boxes < ``_MIN_SAMPLES_FOR_OUTLIER`` (6) → return [True]*N."""

    def test_empty_input_returns_empty_list(self) -> None:
        assert filter_y_position_outliers([]) == []

    def test_single_box_keeps(self) -> None:
        assert filter_y_position_outliers([70.0]) == [True]

    def test_below_min_samples_keeps_all(self) -> None:
        """5 boxes < ngưỡng 6 → keep all (kể cả Y khác nhau xa)."""
        y_centers = [70.0, 70.0, 70.0, 200.0, 200.0]
        result_mask = filter_y_position_outliers(y_centers)
        assert all(result_mask)
        assert len(result_mask) == 5


class TestDenseClusterDiscovery:
    """Pass 1: phát hiện cụm Y dày đặc.

    Bins rộng 5 pixel. Density threshold = ``max(30, len(y_centers) // 100)``.
    """

    def test_single_dense_cluster_keeps_all_in_cluster(self) -> None:
        """Tất cả box ở Y≈70 (cụm dày đặc) → keep all."""
        y_centers = [70.0 + (i % 3) * 0.5 for i in range(100)]  # 100 boxes Y=70-71
        result_mask = filter_y_position_outliers(y_centers)
        assert all(result_mask)

    def test_multimode_distribution_keeps_both_dense_clusters(self) -> None:
        """Scenario test1: 2 cụm dày Y=70 (90 boxes) và Y=110 (40 boxes)."""
        y_centers_cluster_1 = [70.0 + (i % 5) * 0.5 for i in range(90)]
        y_centers_cluster_2 = [110.0 + (i % 5) * 0.5 for i in range(40)]
        all_y_centers = y_centers_cluster_1 + y_centers_cluster_2

        result_mask = filter_y_position_outliers(all_y_centers)

        # Cả 2 cụm đều dense (>= 30) → cả 2 đều được giữ.
        assert sum(result_mask) == len(all_y_centers)

    def test_sparse_outliers_outside_dense_cluster_dropped(self) -> None:
        """Scenario file 1: 1 cụm Y=70 (100 boxes) + 5 outliers Y=110 (rác)."""
        dense_cluster_y_centers = [70.0 + (i % 3) * 0.5 for i in range(100)]
        sparse_outlier_y_centers = [110.0, 110.0, 109.0, 108.0, 111.0]
        all_y_centers = dense_cluster_y_centers + sparse_outlier_y_centers

        result_mask = filter_y_position_outliers(all_y_centers)

        # Cluster Y=70 giữ hết.
        assert all(result_mask[:100])
        # Outliers Y=110 (chỉ 5 boxes, dưới ngưỡng density) → drop.
        assert not any(result_mask[100:])


class TestBinExpansionTwoNeighbors:
    """Pass 2: mở rộng cụm dày đặc ra ±2 bins lân cận."""

    def test_boxes_one_bin_away_from_dense_cluster_kept(self) -> None:
        """Box Y=68 (cách dense cluster Y=70 ~ 0.4 bin) → vẫn keep."""
        dense_cluster_y_centers = [70.0] * 50
        slightly_offset_y_centers = [68.0, 73.0]  # ±1 bin
        all_y_centers = dense_cluster_y_centers + slightly_offset_y_centers

        result_mask = filter_y_position_outliers(all_y_centers)

        # Cả slightly_offset boxes đều phải keep (trong vùng ±2 bins lân cận).
        assert result_mask[-1] is True
        assert result_mask[-2] is True

    def test_box_three_bins_away_dropped(self) -> None:
        """Box Y=90 (cách dense cluster Y=70 ~ 4 bins = 20px) → drop."""
        dense_cluster_y_centers = [70.0] * 50
        far_offset_y_centers = [90.0, 95.0]  # +4, +5 bins
        all_y_centers = dense_cluster_y_centers + far_offset_y_centers

        result_mask = filter_y_position_outliers(all_y_centers)

        # 2 box Y=90, 95 ngoài expansion → drop.
        assert result_mask[-1] is False
        assert result_mask[-2] is False


class TestFallbackMad:
    """Pass 3: fallback về MAD khi KHÔNG có cụm dày đặc nào (data ít)."""

    def test_few_boxes_below_density_threshold_uses_mad(self) -> None:
        """Chỉ 10 boxes (< ngưỡng 30) ở 1 cụm → fallback MAD, keep all."""
        y_centers = [70.0 + i * 0.5 for i in range(10)]  # 10 boxes Y=70-74.5
        result_mask = filter_y_position_outliers(y_centers)
        # MAD: median≈72, deviations nhỏ, all within threshold → all True.
        assert all(result_mask)

    def test_fallback_mad_with_explicit_threshold_floor(self) -> None:
        """Fallback MAD nhận ``minimum_threshold_distance`` (sàn ROI)."""
        y_centers = [70.0, 71.0, 72.0, 73.0, 74.0, 75.0]  # tight cluster
        result_with_default_floor = filter_y_position_outliers(y_centers)
        result_with_large_floor = filter_y_position_outliers(
            y_centers, minimum_threshold_distance=100.0
        )
        # Cả 2 phải keep all (tight cluster).
        assert all(result_with_default_floor)
        assert all(result_with_large_floor)


class TestEdgeCasesV25:
    """Test các edge cases cụ thể."""

    def test_negative_y_values_treated_consistently(self) -> None:
        """Y values âm (lý thuyết không xảy ra, nhưng để defensive)."""
        y_centers = [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]
        # Tất cả cùng giá trị → 1 cụm duy nhất → keep all.
        result_mask = filter_y_position_outliers(y_centers)
        assert all(result_mask)

    def test_three_dense_clusters_all_kept(self) -> None:
        """3 cụm Y cách xa: Y=50 (40 boxes), Y=120 (40 boxes), Y=200 (40 boxes)."""
        cluster_a = [50.0 + (i % 3) * 0.5 for i in range(40)]
        cluster_b = [120.0 + (i % 3) * 0.5 for i in range(40)]
        cluster_c = [200.0 + (i % 3) * 0.5 for i in range(40)]
        all_y_centers = cluster_a + cluster_b + cluster_c

        result_mask = filter_y_position_outliers(all_y_centers)
        # Tất cả 3 cụm đều dense → keep tất cả 120 boxes.
        assert sum(result_mask) == 120

    def test_density_threshold_scales_with_input_size(self) -> None:
        """Cluster có 50 boxes trong 200 tổng (25%) → dense (>= 30).

        Cluster có 5 boxes trong 1000 tổng (0.5%) → không dense (< 30 và < 1%).
        """
        # Scenario 1: 200 boxes, cluster B 50 boxes (dense), 150 boxes ngoài (dense).
        main_y = [70.0] * 150
        secondary_y = [110.0] * 50
        all_y = main_y + secondary_y
        mask_1 = filter_y_position_outliers(all_y)
        assert sum(mask_1) == 200  # cả 2 đều dense

        # Scenario 2: 1000 boxes, 995 main + 5 sparse outliers.
        main_y_large = [70.0] * 995
        sparse_y = [110.0, 111.0, 109.0, 110.0, 110.0]
        all_y_large = main_y_large + sparse_y
        mask_2 = filter_y_position_outliers(all_y_large)
        # 995 main → keep, 5 sparse → drop.
        assert all(mask_2[:995])
        assert not any(mask_2[995:])
