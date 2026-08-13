"""Module phát hiện outlier robust — Tukey IQR cho confidence, MAD cho Y-position.

Lý do dùng phương pháp robust:
    Mean/standard deviation **rất nhạy** với outlier — chính cái thứ ta
    đang muốn loại bỏ. Median và MAD (Median Absolute Deviation) là
    những thống kê **resistant**: dù 25-49% dữ liệu là outlier, kết
    quả vẫn ổn định.

CẢI TIẾN HIỆU NĂNG & LOGIC:
    * Giảm thiểu Redundant Sorting: Sort mảng 1 lần duy nhất cho mỗi chu trình lọc.
    * Khắc phục lỗi MAD=0: Ngăn chặn hiện tượng tê liệt bộ lọc khi dữ liệu quá đồng đều.
"""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger

_TUKEY_K: float = 2.0
_MAD_CONSISTENCY: float = 1.4826
_MAD_K: float = 4.0
_MIN_SAMPLES_FOR_OUTLIER: int = 6



# ==========================================
# INTERNAL FAST HELPERS (Tránh sort nhiều lần)
# ==========================================
def _median_sorted(sorted_vals: list[float]) -> float:
    """Tính median siêu tốc trên mảng đã được sort."""
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return sorted_vals[mid]


def _percentile_sorted(sorted_vals: list[float], p: float) -> float:
    """Tính percentile siêu tốc trên mảng đã được sort."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = (p / 100.0) * (n - 1)
    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, n - 1)
    fraction = rank - lower_idx
    return sorted_vals[lower_idx] + fraction * (
        sorted_vals[upper_idx] - sorted_vals[lower_idx]
    )


# ==========================================
# PUBLIC API (Đảm bảo tương thích ngược)
# ==========================================
def median(values: Sequence[float]) -> float:
    """Median không cần numpy. ``ValueError`` nếu input rỗng."""
    if not values:
        raise ValueError("Không tính được median từ list rỗng.")
    return _median_sorted(sorted(values))


def percentile(values: Sequence[float], p: float) -> float:
    """Percentile interpolation tuyến tính. ``p`` ∈ [0, 100]."""
    if not values:
        raise ValueError("Không tính được percentile từ list rỗng.")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"Percentile phải trong [0, 100], nhận {p}.")
    return _percentile_sorted(sorted(values), p)


def tukey_iqr_bounds(
    values: Sequence[float], k: float = _TUKEY_K
) -> tuple[float, float]:
    """Tính ngưỡng Tukey IQR ``(lower, upper)``."""
    if len(values) < _MIN_SAMPLES_FOR_OUTLIER:
        return (float("-inf"), float("inf"))

    sorted_vals = sorted(values)
    q1 = _percentile_sorted(sorted_vals, 25.0)
    q3 = _percentile_sorted(sorted_vals, 75.0)
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)


def mad_score(values: Sequence[float], target: float) -> float:
    """Modified Z-score dùng MAD thay cho SD."""
    if not values:
        return 0.0

    sorted_vals = sorted(values)
    med = _median_sorted(sorted_vals)

    deviations = sorted(abs(v - med) for v in values)
    mad = _median_sorted(deviations)

    # [CRITICAL FIX] Ép MAD tối thiểu để không bị lỗi tàng hình Outlier khi data quá đều
    effective_mad = max(mad, 1e-6)

    return abs(target - med) / (_MAD_CONSISTENCY * effective_mad)


def filter_confidence_outliers(
    confidences: Sequence[float],
    k: float = _TUKEY_K,
) -> list[bool]:
    """Đánh dấu outlier trong dãy confidence (``True`` = giữ, ``False`` = drop)."""
    if len(confidences) < _MIN_SAMPLES_FOR_OUTLIER:
        return [True] * len(confidences)

    # [PERFORMANCE FIX] Chỉ sort 1 lần duy nhất
    sorted_confs = sorted(confidences)
    q1 = _percentile_sorted(sorted_confs, 25.0)
    q3 = _percentile_sorted(sorted_confs, 75.0)
    iqr = q3 - q1
    med = _median_sorted(sorted_confs)

    # IQR rất nhỏ → fallback dùng median - 0.20 thay vì dựa vào IQR.
    lower_bound = med - 0.20 if iqr < 0.05 else q1 - k * iqr

    mask = [conf >= lower_bound for conf in confidences]
    dropped = mask.count(False)

    if dropped > 0:
        logger.debug(
            "Tukey IQR confidence filter: drop {}/{} frame "
            "(threshold={:.3f}, k={:.1f}, iqr={:.3f}).",
            dropped, len(confidences), lower_bound, k, iqr,
        )
    return mask


def filter_y_position_outliers(
    y_centers: Sequence[float],
    k: float = _MAD_K,
    minimum_threshold_distance: float = 0.0,
) -> list[bool]:
    """Đánh dấu outlier theo Y-center, ưu tiên density-aware cluster detection.

    [FIX v2.25 CRITICAL]: Thuật toán mới ưu tiên **density clustering** thay
    vì MAD-thuần. Khi phụ đề có nhiều cụm Y khác nhau (multi-mode), MAD
    filter lọc nhầm cụm thiểu số (vd test1: cụm Y=110 có 7000+ boxes là phụ
    đề 2-3 ký tự ngắn, MAD coi là outlier dù số lượng lớn).

    Thuật toán 3-pass:
        1. **Pass 1 — Cluster discovery**: chia Y range thành bins rộng 5px,
           xác định bins có density >= max(30, 1% tổng) là "dense clusters".
        2. **Pass 2 — Multi-mode detection**: nếu phát hiện >= 1 dense
           cluster, áp dụng logic density-based. Box thuộc dense cluster
           (hoặc ±2 bins lân cận) → KEEP. Box ngoài → DROP.
        3. **Pass 3 — Fallback MAD**: nếu KHÔNG phát hiện dense cluster nào
           (data ít), dùng MAD threshold cũ.

    Lợi ích:
        * **Test1** (`'豪门ZW'`): cụm Y=110 đông (7000+ boxes) → keep phụ đề
          ngắn `'死变态'`, `'变态'`, ... ở vùng dưới.
        * **File 1** (echo trail): cụm Y=107 thưa (chỉ 17 boxes rác) → drop
          rác `'GOR'`, `'ZGOS'` mà giữ phụ đề chính Y=63.

    Args:
        y_centers: Danh sách Y-center các box.
        k: Hệ số MAD threshold cho fallback.
        minimum_threshold_distance: Sàn tối thiểu cho MAD fallback (legacy).

    Returns:
        Danh sách boolean: True = giữ, False = drop.
    """
    if len(y_centers) < _MIN_SAMPLES_FOR_OUTLIER:
        return [True] * len(y_centers)

    # [FIX v2.25] Pass 1: cluster discovery bằng histogram bins.
    histogram_bin_width_pixels = 5.0
    min_cluster_density_count = max(30, len(y_centers) // 100)

    box_count_by_bin: dict[int, int] = {}
    for y_value in y_centers:
        bin_index = int(y_value // histogram_bin_width_pixels)
        box_count_by_bin[bin_index] = box_count_by_bin.get(bin_index, 0) + 1

    dense_cluster_bin_indices: set[int] = {
        bin_index
        for bin_index, box_count in box_count_by_bin.items()
        if box_count >= min_cluster_density_count
    }

    # Pass 2: Density-based filter (nếu phát hiện được cluster).
    if dense_cluster_bin_indices:
        # Mở rộng cụm dày đặc ra ±2 bins lân cận (10 pixel) — chống artifact
        # biên (box vừa lệch sang bin kế bên dù vẫn cùng dòng phụ đề).
        expanded_keep_bins: set[int] = set()
        for dense_bin in dense_cluster_bin_indices:
            for offset in range(-2, 3):
                expanded_keep_bins.add(dense_bin + offset)

        mask: list[bool] = []
        for y_value in y_centers:
            bin_index = int(y_value // histogram_bin_width_pixels)
            mask.append(bin_index in expanded_keep_bins)

        dropped_count = mask.count(False)
        if dropped_count > 0:
            logger.debug(
                "Density Y filter: drop {}/{} box (dense_clusters={}, "
                "min_density={}, bin_width={}px)",
                dropped_count, len(y_centers), len(dense_cluster_bin_indices),
                min_cluster_density_count, int(histogram_bin_width_pixels),
            )
        return mask

    # Pass 3: Fallback MAD (data ít, không phát hiện cluster rõ).
    sorted_y = sorted(y_centers)
    med = _median_sorted(sorted_y)

    deviations = sorted(abs(y - med) for y in y_centers)
    mad = _median_sorted(deviations)
    effective_mad = max(mad, 2.0)

    threshold_distance = max(
        k * _MAD_CONSISTENCY * effective_mad,
        10.0,
        float(minimum_threshold_distance),
    )

    mask = [abs(y - med) <= threshold_distance for y in y_centers]
    dropped = mask.count(False)

    if dropped > 0:
        logger.debug(
            "MAD Y-position filter (fallback): drop {}/{} box (median={:.1f}, "
            "effective_mad={:.1f}, threshold_dist={:.1f})",
            dropped, len(y_centers), med, effective_mad, threshold_distance
        )
    return mask


__all__ = [
    "filter_confidence_outliers",
    "filter_y_position_outliers",
    "mad_score",
    "median",
    "percentile",
    "tukey_iqr_bounds",
]
