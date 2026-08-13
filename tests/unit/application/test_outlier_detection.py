"""Tests cho :mod:`outlier_detection` — Tukey IQR + MAD-based detection."""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.outlier_detection import (
    filter_confidence_outliers,
    filter_y_position_outliers,
    mad_score,
    median,
    percentile,
    tukey_iqr_bounds,
)


class TestMedian:
    def test_odd_length(self) -> None:
        assert median([1.0, 3.0, 2.0]) == 2.0

    def test_even_length(self) -> None:
        assert median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_single(self) -> None:
        assert median([5.0]) == 5.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            median([])


class TestPercentile:
    def test_q1_q3(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(values, 25.0) == 2.0
        assert percentile(values, 75.0) == 4.0

    def test_min_max(self) -> None:
        values = [10.0, 20.0, 30.0]
        assert percentile(values, 0.0) == 10.0
        assert percentile(values, 100.0) == 30.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            percentile([], 50.0)

    def test_invalid_p_raises(self) -> None:
        with pytest.raises(ValueError):
            percentile([1.0, 2.0], 150.0)


class TestTukeyIqrBounds:
    def test_uniform_distribution(self) -> None:
        # IQR = 50, fence ±2*50 = ±100.
        values = list(range(0, 100))
        lower, upper = tukey_iqr_bounds([float(v) for v in values], k=2.0)
        # Q1 ≈ 24.75, Q3 ≈ 74.25, IQR ≈ 49.5.
        # Lower ≈ 24.75 - 99 = -74.25, upper ≈ 173.25.
        assert lower < 0
        assert upper > 100

    def test_too_few_samples_returns_inf(self) -> None:
        # < 6 sample → return ±inf để không filter ai.
        lower, upper = tukey_iqr_bounds([1.0, 2.0, 3.0])
        assert lower == float("-inf")
        assert upper == float("inf")


class TestMadScore:
    def test_value_at_median_is_zero(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert mad_score(values, 3.0) == 0.0

    def test_outlier_high_score(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        # 100 cách rất xa median → high score.
        assert mad_score(values, 100.0) > 5.0

    def test_zero_mad_returns_high_score_for_distant_target(self) -> None:
        """Khi tất cả values bằng nhau (MAD=0), target rất xa median phải
        cho score CAO (đúng nghĩa là outlier rõ rệt).

        Lý do: trước v2.x cũ, MAD=0 → return 0.0 → không phát hiện được
        outlier khi data quá đều ('Outlier tàng hình bug'). [CRITICAL FIX]
        đã ép ``effective_mad = max(mad, 1e-6)`` nên giờ ``|target - median|
        / (1.4826 * 1e-6)`` cho giá trị rất lớn → phát hiện được outlier.
        """
        values = [5.0] * 10
        score = mad_score(values, 100.0)
        # 100 - 5 = 95; 95 / (1.4826 * 1e-6) ≈ 6.4e7 — rất lớn.
        assert score > 1e6, f"Score phải rất lớn khi data đều và target xa, nhận {score}"

    def test_zero_mad_target_equals_median_returns_zero(self) -> None:
        """Khi MAD=0 và target = median (không xa) → score gần 0."""
        values = [5.0] * 10
        score = mad_score(values, 5.0)
        assert score == 0.0, f"Score phải = 0 khi target trùng median, nhận {score}"

    def test_empty_returns_zero(self) -> None:
        assert mad_score([], 5.0) == 0.0


class TestFilterConfidenceOutliers:
    def test_keeps_uniform_data(self) -> None:
        confs = [0.85, 0.86, 0.87, 0.85, 0.86, 0.87, 0.85, 0.86]
        mask = filter_confidence_outliers(confs)
        assert all(mask)

    def test_drops_low_outlier(self) -> None:
        confs = [0.95, 0.94, 0.95, 0.94, 0.95, 0.94, 0.95, 0.10]
        mask = filter_confidence_outliers(confs, k=1.5)
        # Outlier 0.10 phải bị drop.
        assert not mask[-1]

    def test_does_not_drop_high_outlier(self) -> None:
        # High conf không phải noise — không drop.
        confs = [0.50, 0.51, 0.52, 0.50, 0.51, 0.52, 0.99, 0.50]
        mask = filter_confidence_outliers(confs)
        # 0.99 phải được giữ.
        assert mask[6]

    def test_too_few_samples_keeps_all(self) -> None:
        mask = filter_confidence_outliers([0.1, 0.95, 0.96])
        assert all(mask)


class TestFilterYPositionOutliers:
    def test_keeps_consistent_y(self) -> None:
        # Subtitle ổn định ở Y ≈ 950.
        ys = [950.0, 952.0, 948.0, 951.0, 949.0, 950.0, 951.0]
        mask = filter_y_position_outliers(ys)
        assert all(mask)

    def test_drops_far_outlier(self) -> None:
        # 6 box ở Y ≈ 950, 1 box ở Y = 100 (logo cắt nhầm).
        ys = [950.0, 952.0, 948.0, 951.0, 949.0, 950.0, 100.0]
        mask = filter_y_position_outliers(ys)
        # Index 6 (Y=100) phải bị drop.
        assert not mask[6]

    def test_too_few_keeps_all(self) -> None:
        mask = filter_y_position_outliers([100.0, 950.0])
        assert all(mask)
