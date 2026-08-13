"""Chọn vùng ROI phụ đề CHÍNH từ kết quả phân tích bbox (logic thuần, không Qt).

Dùng cho preset "Tự nhận diện vùng phụ đề": sau khi engine phân tích toàn khung,
chọn MỘT cụm "đậm đặc nhất" làm ROI duy nhất rồi trích xuất hardsub.

Cải tiến cách chọn (đo trên 5 bộ dữ liệu fullscreen thật): tiêu chí "đậm đặc"
KHÔNG nên chỉ là "nhiều box nhất" — nhiễu cảnh động có thể tạo nhiều box trong ít
khung hình. Phụ đề thật là dải có **frame-presence áp đảo** (xuất hiện ổn định ở
RẤT nhiều khung hình khác nhau, đo được ~25× so với nhiễu). Vì vậy ta xếp hạng
theo ``frame_count`` trước, rồi tới tổng số box, và ưu tiên nhẹ dải nằm ở nửa dưới
khung (nơi phụ đề hardsub thường nằm) để phá thế hoà.
"""

from __future__ import annotations

from subtitles_extractor.domain.value_objects.roi import Roi, TextOrientation
from subtitles_extractor.infrastructure.video.bbox_analyzer import ROICluster


def score_cluster_density(cluster: ROICluster, frame_height: int) -> tuple[float, float, float]:
    """Khoá sắp xếp đánh giá độ "đậm đặc phụ đề" của một cụm (cao = phụ đề hơn).

    [v3.19] Điểm chính là TỔNG HỢP kháng watermark/logo:
        ``frame_count × trọng_số_bề_rộng × trọng_số_vị_trí``
    - **Trọng số bề rộng**: phụ đề là dòng chữ DÀI (câu), logo/watermark là khối
      NHỎ — kích thước theo trục văn bản ≥ 30% khung đạt trọng số 1.0, nhỏ hơn bị
      giảm tuyến tính (sàn 0.2). Chống logo "chết" hiện diện 100% phim nhưng bé.
    - **Trọng số vị trí** (text ngang): phụ đề hardsub nằm nửa dưới khung; khối ở
      nửa trên (logo kênh, tiêu đề) bị giảm còn 0.45.
    Tie-break: ``bbox_count`` rồi vị trí Y.

    Returns:
        Tuple khoá sắp xếp (dùng với ``max``/``sorted reverse=True``).
    """
    center_y_ratio = 0.0
    if frame_height > 0:
        center_y_ratio = ((cluster.coord_y_min + cluster.coord_y_max) / 2.0) / frame_height

    cluster_width = float(cluster.coord_x_max - cluster.coord_x_min)
    cluster_height = float(cluster.coord_y_max - cluster.coord_y_min)
    is_vertical = getattr(cluster, "orientation", None) == TextOrientation.VERTICAL
    size_along_text = cluster_height if is_vertical else cluster_width
    # Trục chuẩn hoá: text dọc so với chiều cao khung; text ngang cần bề rộng khung —
    # ROICluster không mang frame_width nên ước lượng qua frame_height × 9/16 là
    # không tin cậy; dùng mốc tuyệt đối an toàn: coi 0.30 × frame_height là "dài".
    reference_span = 0.30 * float(max(1, frame_height))
    width_weight = min(1.0, max(0.2, size_along_text / max(1.0, reference_span)))

    position_weight = 1.0
    if not is_vertical and center_y_ratio < 0.5:
        position_weight = 0.45

    composite = float(cluster.frame_count) * width_weight * position_weight
    return (composite, float(cluster.bbox_count), center_y_ratio)


def select_primary_subtitle_cluster(
    clusters: list[ROICluster], frame_height: int
) -> ROICluster | None:
    """Chọn cụm phụ đề chính (đậm đặc nhất) trong danh sách cụm.

    Args:
        clusters: Các cụm ROI do engine phân tích trả về.
        frame_height: Chiều cao khung video.

    Returns:
        Cụm đậm đặc nhất, hoặc ``None`` nếu danh sách rỗng.
    """
    if not clusters:
        return None
    return max(clusters, key=lambda cluster: score_cluster_density(cluster, frame_height))


def cluster_to_roi(cluster: ROICluster) -> Roi:
    """Chuyển một :class:`ROICluster` sang :class:`Roi` (kẹp biên không âm)."""
    x = max(0, int(cluster.coord_x_min))
    y = max(0, int(cluster.coord_y_min))
    width = max(1, int(cluster.coord_x_max - cluster.coord_x_min))
    height = max(1, int(cluster.coord_y_max - cluster.coord_y_min))
    return Roi(x=x, y=y, width=width, height=height)


def select_primary_subtitle_roi(
    clusters: list[ROICluster], frame_height: int
) -> Roi | None:
    """Tiện ích: chọn cụm phụ đề chính và trả về luôn dưới dạng :class:`Roi`.

    Returns:
        ROI duy nhất bao quanh vùng phụ đề chính, hoặc ``None`` nếu không có cụm.
    """
    primary = select_primary_subtitle_cluster(clusters, frame_height)
    return cluster_to_roi(primary) if primary is not None else None


#: Ngưỡng coi một cụm là "mega-cluster" (nuốt cả khung) — chiều cao vượt tỉ lệ này.
_MEGA_CLUSTER_HEIGHT_RATIO = 0.55


def select_subtitle_roi_smart(
    clusters: list[ROICluster],
    boxes: "list",
    frame_width: int,
    frame_height: int,
) -> Roi | None:
    """Chọn ROI phụ đề cho preset Tự nhận diện — ƯU TIÊN TUYỆT ĐỐI lõi AI.

    [v3.17] Thứ tự ưu tiên (theo yêu cầu kháng mega-cluster cho video dọc/nhiễu):

    1. **Lõi AI ``BBoxAnalyzer``** (gom cụm không gian DBSCAN + tinh chỉnh dải theo
       mật độ thời gian của v3.16): lấy cụm đậm đặc nhất. Nếu cụm KHÔNG phải
       mega-cluster (``height ≤ _MEGA_CLUSTER_HEIGHT_RATIO``) → DÙNG NGAY. Đây là
       đường đi mặc định và mạnh nhất: cụm AI đã được khử mega-cluster + bám dải
       phụ đề thật, nên không bị logo/text-dán đánh lừa như thuật toán cắt lớp cũ.
    2. **Phương án dự phòng** (chỉ khi AI thất bại — không có cụm hoặc cụm vẫn là
       mega-cluster): thử bộ phát hiện vùng đa-dải ``detect_text_regions``, rồi tới
       ``detect_subtitle_band_from_boxes`` (đo frame-presence theo trục Y, robust).
    3. Cùng đường: nếu mọi cách trên đều rỗng nhưng vẫn có cụm AI (kể cả mega), trả
       tạm cụm đó để không bỏ trắng kết quả.

    Args:
        clusters: Cụm ROI do lõi AI ``BBoxAnalyzer`` trả về.
        boxes: Danh sách :class:`RawBBox` thô (cho phương án dự phòng).
        frame_width, frame_height: Kích thước khung video.

    Returns:
        ROI phụ đề chính, hoặc ``None`` nếu không thể xác định.
    """
    primary = select_primary_subtitle_cluster(clusters, frame_height)

    # (1) ƯU TIÊN TUYỆT ĐỐI lõi AI: cụm chính sạch (không mega-cluster) → dùng ngay.
    if primary is not None and frame_height > 0:
        height_ratio = (primary.coord_y_max - primary.coord_y_min) / frame_height
        if height_ratio <= _MEGA_CLUSTER_HEIGHT_RATIO:
            return cluster_to_roi(primary)

    # (2) PHƯƠNG ÁN DỰ PHÒNG — chỉ chạy khi AI thất bại (không cụm / còn mega-cluster).
    if boxes:
        from subtitles_extractor.infrastructure.video.text_region_detection import (
            ROLE_PRIMARY,
            detect_text_regions,
        )

        regions = detect_text_regions(boxes, frame_width, frame_height)
        primary_region = next(
            (region for region in regions if region.role == ROLE_PRIMARY), None
        )
        if primary_region is not None:
            return primary_region.roi

    band_roi = detect_subtitle_band_from_boxes(boxes, frame_width, frame_height)
    if band_roi is not None:
        return band_roi

    # (3) Cùng đường: trả tạm cụm AI nếu có (kể cả mega) để không bỏ trắng.
    return cluster_to_roi(primary) if primary is not None else None


def detect_subtitle_band_from_boxes(
    boxes: "list",
    frame_width: int,
    frame_height: int,
    *,
    conf_threshold: float = 0.85,
    num_bins: int = 60,
    presence_ratio: float = 0.30,
    dominance_ratio: float = 0.80,
) -> Roi | None:
    """Phát hiện dải phụ đề bằng mật độ FRAME theo trục Y (robust với mega-cluster).

    Khác cách dựa cụm hình thái học (dễ bị "mega-cluster" nuốt cả khung khi OCR
    fullscreen dày đặc), hàm này đo trực tiếp **frame-presence** của từng dải ngang:
    phụ đề thật là dải có số khung hình xuất hiện áp đảo (đo được ~25× nhiễu cảnh).

    Thuật toán: chia trục Y thành ``num_bins`` dải, đếm số frame phân biệt có box độ
    tin cậy cao ở mỗi dải; lấy đỉnh frame-presence (ưu tiên dải THẤP nhất trong các
    đỉnh tương đương — phụ đề hardsub ở đáy), rồi mở rộng quanh đỉnh trong khi mật độ
    còn ≥ ``presence_ratio`` lần đỉnh.

    Args:
        boxes: Danh sách :class:`RawBBox`.
        frame_width, frame_height: Kích thước khung.
        conf_threshold: Ngưỡng độ tin cậy box được tính.
        num_bins: Số dải chia trục Y.
        presence_ratio: Tỉ lệ tối thiểu so với đỉnh để gộp dải lân cận.
        dominance_ratio: Các dải có mật độ ≥ tỉ lệ này của đỉnh được coi là "đồng
            hạng" → trong nhóm đó chọn dải thấp nhất (gần đáy).

    Returns:
        ROI bao quanh dải phụ đề, hoặc ``None`` nếu không đủ dữ liệu.
    """
    if not boxes or frame_height <= 0 or num_bins <= 0:
        return None

    bin_height = frame_height / num_bins
    presence: list[set] = [set() for _ in range(num_bins)]
    extents: list[list[tuple[float, float, float, float]]] = [[] for _ in range(num_bins)]

    for box in boxes:
        if box.confidence < conf_threshold:
            continue
        center_y = (box.coord_y_min + box.coord_y_max) / 2.0
        bin_index = int(center_y / bin_height) if bin_height > 0 else 0
        if 0 <= bin_index < num_bins:
            presence[bin_index].add(box.frame_idx)
            extents[bin_index].append(
                (box.coord_x_min, box.coord_x_max, box.coord_y_min, box.coord_y_max)
            )

    counts = [len(frame_set) for frame_set in presence]
    peak_count = max(counts)
    if peak_count == 0:
        return None

    # Ưu tiên dải đáy trong các đỉnh "đồng hạng" (phụ đề hardsub nằm dưới).
    dominant_floor = peak_count * dominance_ratio
    peak_bin = max(
        (i for i, c in enumerate(counts) if c >= dominant_floor),
        key=lambda i: i,
    )

    # Mở rộng dải liên tục quanh đỉnh khi mật độ còn đủ cao.
    threshold = peak_count * presence_ratio
    low = high = peak_bin
    while low - 1 >= 0 and counts[low - 1] >= threshold:
        low -= 1
    while high + 1 < num_bins and counts[high + 1] >= threshold:
        high += 1

    selected = [extent for i in range(low, high + 1) for extent in extents[i]]
    if not selected:
        return None

    x_min = min(e[0] for e in selected)
    x_max = max(e[1] for e in selected)
    y_min = min(e[2] for e in selected)
    y_max = max(e[3] for e in selected)
    return Roi(
        x=max(0, int(x_min)),
        y=max(0, int(y_min)),
        width=max(1, int(x_max - x_min)),
        height=max(1, int(y_max - y_min)),
    )
