"""Engine phát hiện ĐA VÙNG text từ OCR thô (tái thiết kế ROI tự động).

Mục tiêu: vẽ ĐÚNG và ĐỦ mọi vùng chữ ổn định mà video hiển thị — dải phụ đề chính
(thường nửa dưới) và các vùng thông tin phụ (tiêu đề/tên tập/dòng phụ trên cao) —
mỗi vùng "vừa đủ" bao quanh chữ, không nuốt cả khung như mega-cluster.

Nguyên lý (rút ra từ phân tích dữ liệu OCR thực tế):
  * **Frame-presence** (số khung hình phân biệt có chữ ở một dải Y) là tín hiệu mạnh
    nhất: phụ đề/thông tin ổn định xuất hiện ở dải cố định qua RẤT NHIỀU khung, trong
    khi chữ trong cảnh (nhiễu) rải rác, frame-presence thấp.
  * **Số câu khác nhau (unique text)** giúp loại nhiễu lác đác: vùng text thật có
    nhiều nội dung khác nhau theo thời gian.
  * **Bề rộng X theo percentile** cho ROI "vừa đủ": cắt 2 đuôi outlier (một box lạc ra
    biên) thay vì min-max thô khiến ROI rộng cả khung.

Hàm này độc lập hoàn toàn với phép gom cụm hình thái học (vốn dễ tạo "mega-cluster"
khi OCR fullscreen dày đặc), nên kháng được hiện tượng nuốt cả khung.
"""

from __future__ import annotations

from dataclasses import dataclass

from subtitles_extractor.domain.value_objects.roi import Roi

#: Vai trò vùng text.
ROLE_PRIMARY = "primary"      # Dải phụ đề chính (ưu tiên OCR liên tục).
ROLE_SECONDARY = "secondary"  # Vùng thông tin phụ (tiêu đề, dòng phụ…).


@dataclass(frozen=True)
class TextRegion:
    """Một vùng chữ ổn định phát hiện được từ OCR thô.

    Attributes:
        roi: Hộp bao "vừa đủ" quanh chữ của vùng.
        frame_presence_ratio: Tỉ lệ khung hình có chữ ở vùng này (0..1).
        unique_text_count: Số câu khác nhau quan sát được (đo độ "động").
        box_count: Tổng số bbox rơi vào vùng.
        role: ``ROLE_PRIMARY`` hoặc ``ROLE_SECONDARY``.
    """

    roi: Roi
    frame_presence_ratio: float
    unique_text_count: int
    box_count: int
    role: str


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Percentile tuyến tính trên list ĐÃ sắp xếp (tránh phụ thuộc numpy ở runtime)."""
    if not sorted_values:
        return 0.0
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def detect_text_regions(
    boxes: list,
    frame_width: int,
    frame_height: int,
    *,
    conf_threshold: float = 0.80,
    num_bins: int | None = None,
    min_presence_ratio: float = 0.06,
    min_unique_texts: int = 12,
    min_width_ratio: float = 0.12,
    x_trim_percentile: float = 2.0,
    merge_gap_bins: int = 1,
    max_height_ratio: float = 0.28,
) -> list[TextRegion]:
    """Phát hiện mọi vùng chữ ổn định trong khung.

    Args:
        boxes: Danh sách ``RawBBox`` (cần ``coord_*``, ``confidence``, ``frame_idx``,
            và tuỳ chọn ``text``).
        frame_width, frame_height: Kích thước khung.
        conf_threshold: Bỏ qua box độ tin cậy thấp hơn.
        num_bins: Số dải Y; mặc định ~ chiều cao / 24px (cỡ một dòng chữ).
        min_presence_ratio: Ngưỡng tối thiểu (theo tỉ lệ tổng khung) để giữ một vùng.
        min_unique_texts: Số câu khác nhau tối thiểu để coi là vùng text thật.
        min_width_ratio: Bề rộng tối thiểu (tỉ lệ khung) để loại logo/nhiễu hẹp.
        x_trim_percentile: Cắt mỗi đuôi X bao nhiêu % để có bề rộng "vừa đủ".
        merge_gap_bins: Gộp các dải cách nhau ≤ số bin này (phụ đề nhiều dòng).

    Returns:
        Danh sách :class:`TextRegion` đã sắp theo Y; vùng frame-presence cao nhất ở
        nửa dưới được gán ``ROLE_PRIMARY``. Rỗng nếu không có vùng đạt ngưỡng.
    """
    if not boxes or frame_height <= 0 or frame_width <= 0:
        return []

    bins = num_bins if num_bins and num_bins > 0 else max(20, frame_height // 24)
    bin_height = frame_height / bins

    frame_sets: list[set] = [set() for _ in range(bins)]
    x_mins: list[list[float]] = [[] for _ in range(bins)]
    x_maxs: list[list[float]] = [[] for _ in range(bins)]
    y_mins: list[list[float]] = [[] for _ in range(bins)]
    y_maxs: list[list[float]] = [[] for _ in range(bins)]
    texts: list[list[str]] = [[] for _ in range(bins)]

    total_frames: set = set()
    for box in boxes:
        total_frames.add(box.frame_idx)
        if box.confidence < conf_threshold:
            continue
        center_y = (box.coord_y_min + box.coord_y_max) / 2.0
        bi = int(center_y / bin_height) if bin_height > 0 else 0
        if not 0 <= bi < bins:
            continue
        frame_sets[bi].add(box.frame_idx)
        x_mins[bi].append(box.coord_x_min)
        x_maxs[bi].append(box.coord_x_max)
        y_mins[bi].append(box.coord_y_min)
        y_maxs[bi].append(box.coord_y_max)
        texts[bi].append(getattr(box, "text", "") or "")

    num_frames = len(total_frames)
    if num_frames == 0:
        return []

    presence = [len(s) for s in frame_sets]
    peak = max(presence)
    if peak == 0:
        return []

    # Ngưỡng phân đoạn theo đỉnh (không dùng sàn tuyệt đối — ở video rất dài, sàn
    # thấp dễ nối các dải cảnh rời rạc thành một khối lớn).
    seg_threshold = peak * 0.15

    # Phân đoạn các dải liên tục, cho phép khe hở ≤ merge_gap_bins.
    raw_segments: list[tuple[int, int]] = []
    i = 0
    while i < bins:
        if presence[i] >= seg_threshold:
            j = i
            gap = 0
            k = i + 1
            while k < bins:
                if presence[k] >= seg_threshold:
                    j = k
                    gap = 0
                elif gap < merge_gap_bins:
                    gap += 1
                else:
                    break
                k += 1
            raw_segments.append((i, j))
            i = k
        else:
            i += 1

    # Refine: co mỗi dải về "lõi đậm" quanh đỉnh frame-presence nội bộ, để ROI vừa đủ
    # (không phình theo các bin rìa thưa). Giữ vùng liên tục quanh đỉnh có mật độ ≥
    # 45% đỉnh nội bộ.
    segments: list[tuple[int, int]] = []
    for start, end in raw_segments:
        local_peak_bin = max(range(start, end + 1), key=lambda b: presence[b])
        local_peak = presence[local_peak_bin]
        core_threshold = local_peak * 0.45
        lo = hi = local_peak_bin
        while lo - 1 >= start and presence[lo - 1] >= core_threshold:
            lo -= 1
        while hi + 1 <= end and presence[hi + 1] >= core_threshold:
            hi += 1
        # Cap chiều cao: nếu dải vẫn quá cao (text dày trải rộng), ép về cửa sổ quanh
        # đỉnh để ROI "vừa đủ" cho phụ đề, không bao trùm cả nửa khung.
        max_bins = max(1, int(max_height_ratio * bins))
        if hi - lo + 1 > max_bins:
            half = max_bins // 2
            lo = max(start, local_peak_bin - half)
            hi = min(end, lo + max_bins - 1)
        segments.append((lo, hi))

    regions: list[TextRegion] = []
    for start, end in segments:
        frames: set = set()
        xs_min: list[float] = []
        xs_max: list[float] = []
        ys_min: list[float] = []
        ys_max: list[float] = []
        seg_texts: list[str] = []
        for b in range(start, end + 1):
            frames |= frame_sets[b]
            xs_min += x_mins[b]
            xs_max += x_maxs[b]
            ys_min += y_mins[b]
            ys_max += y_maxs[b]
            seg_texts += texts[b]
        if not frames or not xs_min:
            continue

        fp_ratio = len(frames) / num_frames
        unique_texts = len({t for t in seg_texts if t})
        # Nếu OCR không kèm text, không thể đo unique → bỏ điều kiện uniq.
        has_text = any(seg_texts)

        xs_min_sorted = sorted(xs_min)
        xs_max_sorted = sorted(xs_max)
        x_left = _percentile(xs_min_sorted, x_trim_percentile)
        x_right = _percentile(xs_max_sorted, 100 - x_trim_percentile)
        # Y theo dải bin đã refine/cap (vừa đủ), kẹp trong thực tế box để không hụt.
        band_top = start * bin_height
        band_bottom = (end + 1) * bin_height
        y_top = max(band_top, min(ys_min)) if ys_min else band_top
        y_bottom = min(band_bottom, max(ys_max)) if ys_max else band_bottom
        if y_bottom <= y_top:
            y_top, y_bottom = band_top, band_bottom
        # Đệm dọc nhẹ để bao trọn nét chữ + chừa lề (phòng phụ đề 2 dòng/dấu phụ).
        vertical_pad = frame_height * 0.015
        y_top = max(0.0, y_top - vertical_pad)
        y_bottom = min(float(frame_height), y_bottom + vertical_pad)
        width = x_right - x_left

        if fp_ratio < min_presence_ratio:
            continue
        if has_text and unique_texts < min_unique_texts:
            continue
        if width < min_width_ratio * frame_width:
            continue

        roi = Roi(
            x=max(0, int(x_left)),
            y=max(0, int(y_top)),
            width=max(1, int(width)),
            height=max(1, int(y_bottom - y_top)),
        )
        regions.append(
            TextRegion(
                roi=roi,
                frame_presence_ratio=fp_ratio,
                unique_text_count=unique_texts,
                box_count=len(xs_min),
                role=ROLE_SECONDARY,
            )
        )

    if not regions:
        return []

    # Chọn PRIMARY: frame-presence cao nhất, ưu tiên vùng có tâm ở nửa dưới khung.
    def _primary_key(region: TextRegion) -> tuple:
        center_y_ratio = (region.roi.y + region.roi.height / 2) / frame_height
        lower_half_bonus = 1 if center_y_ratio >= 0.4 else 0
        return (lower_half_bonus, region.frame_presence_ratio)

    primary = max(regions, key=_primary_key)
    regions = [
        TextRegion(
            roi=r.roi,
            frame_presence_ratio=r.frame_presence_ratio,
            unique_text_count=r.unique_text_count,
            box_count=r.box_count,
            role=ROLE_PRIMARY if r is primary else ROLE_SECONDARY,
        )
        for r in regions
    ]
    regions.sort(key=lambda r: r.roi.y)
    return regions
