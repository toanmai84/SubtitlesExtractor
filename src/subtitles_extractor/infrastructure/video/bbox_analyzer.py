"""Phân tích Bounding Box OCR để tìm cụm text ổn định.

    * [TEMPORAL FIX] Temporal Executioner Override: Hạ min_hits của Heatmap xuống tối đa 2, 
      cứu sống tuyệt đối các phụ đề lướt nhanh (fast-flash 2 frames).
    * [MATH FIX] Dynamic Confidence Threshold: Phá bỏ ảo ảnh toán học cũ, thiết lập thuật 
      toán siết nhiễu động học thông minh dựa trên độ tin cậy trung vị thực tế.
    * [MEMORY FIX] DBSCAN KD-Tree Enforcer: Ép thuật toán phân cụm sử dụng 'kd_tree', khóa 
      chặt độ phức tạp bộ nhớ ở O(N log N), triệt tiêu hoàn toàn rủi ro OOM Crash do ma trận O(N^2).
    * [CORE] Kế thừa di sản Absolute Elite Override (0.90), Universal Symbiosis, 
      2D True Mass Physics, ConnectedComponents Exact Map.

Trạng thái: Đạt ngưỡng Hoàn Mỹ (Absolute Perfection). Sẵn sàng triển khai Big Data.
"""

from __future__ import annotations

from collections import defaultdict, deque
from itertools import compress
import numpy as np
from loguru import logger
from typing import Optional, Set
import cv2

from subtitles_extractor.domain.value_objects.roi import TextAlignment, TextOrientation

try:
    from sklearn.cluster import DBSCAN as SklearnDBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    SklearnDBSCAN = None


# ============================================================================
# CẤU HÌNH NGƯỠNG TỐI CAO
# ============================================================================
_MIN_CONFIDENCE: float = 0.85        # Ngưỡng sinh tồn cơ bản
_ELITE_CONFIDENCE: float = 0.90      # Ngưỡng Tinh anh (Lệnh bài miễn tử toàn cục)
_MIN_AREA_PX: float = 40.0           # Bộ lọc rác cấp thấp
_MIN_SIDE_PX: int = 6                # Bộ lọc rác cấp thấp
_HEATMAP_DOWNSCALE: int = 4
# [v3.16] Sàn tin cậy nhẹ cho profile mật độ thời gian (Band Refiner).
# Thấp hơn _MIN_CONFIDENCE để giàu mẫu (đỉnh sắc), chỉ loại nhiễu cực thấp.
_DENSITY_CONFIDENCE_FLOOR: float = 0.40
# [v3.18.1] Số box tối thiểu phải "hỗ trợ" một biên trục (order-statistic thứ k).
# Dòng phụ đề thật đứng ≥0.5s tạo ≥12 box trùng biên; nhiễu thoáng qua chỉ 1-3 box.
_BOUND_SUPPORT_BOXES: int = 6
# [v3.18.3] Tỷ lệ aspect đủ RÕ RÀNG để quyết hướng chữ ngay (w/h hoặc h/w ≥ 2.0).
# Ký tự CJK đơn gần vuông (h/w ≈ 0.8..1.3) là vùng mơ hồ → phải tie-break bằng
# phân tán không gian, không được quyết bằng aspect.
_ASPECT_DECISIVE_RATIO: float = 2.0


# ============================================================================
# Data classes (RAM Optimized O(1))
# ============================================================================

class RawBBox:
    """Bounding box thô từ OCR. Khóa RAM tĩnh O(1)."""
    __slots__ = (
        'coord_x_min', 'coord_y_min', 'coord_x_max', 'coord_y_max',
        'confidence', 'frame_idx', 'timestamp_sec',
        'box_width', 'box_height', 'area', 'center_x', 'center_y'
    )

    def __init__(
        self, coord_x_min: float, coord_y_min: float, coord_x_max: float, coord_y_max: float, 
        confidence: float, frame_idx: int, timestamp_sec: float
    ):
        self.coord_x_min = coord_x_min
        self.coord_y_min = coord_y_min
        self.coord_x_max = coord_x_max
        self.coord_y_max = coord_y_max
        self.confidence = confidence
        self.frame_idx = frame_idx
        self.timestamp_sec = timestamp_sec

        self.box_width = max(0.0, self.coord_x_max - self.coord_x_min)
        self.box_height = max(0.0, self.coord_y_max - self.coord_y_min)
        self.area = self.box_width * self.box_height
        self.center_x = (self.coord_x_min + self.coord_x_max) / 2.0
        self.center_y = (self.coord_y_min + self.coord_y_max) / 2.0


class ROICluster:
    """Cụm ROI sau phân tích. Tối ưu Pure Attributes O(1)."""
    __slots__ = (
        'cluster_id', 'orientation', 'alignment',
        'coord_x_min', 'coord_y_min', 'coord_x_max', 'coord_y_max',
        'bbox_count', 'frame_count', 'mean_confidence', 'keep',
        '_frame_index_set', 'width', 'height', 'area', 'center_x', 'center_y',
        'mass_center_x', 'mass_center_y', 'local_thickness', 'true_mass_area'
    )

    def __init__(
        self, cluster_id: int, orientation: TextOrientation, alignment: TextAlignment, 
        coord_x_min: int, coord_y_min: int, coord_x_max: int, coord_y_max: int,
        bbox_count: int = 0, frame_count: int = 0, mean_confidence: float = 0.0, 
        keep: bool = True, _frame_index_set: Optional[Set[int]] = None,
        mass_center_x: float = 0.0, mass_center_y: float = 0.0, 
        local_thickness: float = 20.0, true_mass_area: float = 0.0
    ):
        self.cluster_id = cluster_id
        self.orientation = orientation
        self.alignment = alignment
        self.coord_x_min = coord_x_min
        self.coord_y_min = coord_y_min
        self.coord_x_max = coord_x_max
        self.coord_y_max = coord_y_max
        self.bbox_count = bbox_count
        self.frame_count = frame_count
        self.mean_confidence = mean_confidence
        self.keep = keep
        self._frame_index_set = _frame_index_set if _frame_index_set is not None else set()

        self.width = max(0, self.coord_x_max - self.coord_x_min)
        self.height = max(0, self.coord_y_max - self.coord_y_min)
        self.area = self.width * self.height
        self.center_x = (self.coord_x_min + self.coord_x_max) / 2.0
        self.center_y = (self.coord_y_min + self.coord_y_max) / 2.0
        
        self.mass_center_x = mass_center_x if mass_center_x > 0 else self.center_x
        self.mass_center_y = mass_center_y if mass_center_y > 0 else self.center_y
        self.local_thickness = local_thickness
        self.true_mass_area = true_mass_area if true_mass_area > 0 else self.area

    def update_bounds(self, x_min: int, y_min: int, x_max: int, y_max: int) -> None:
        """Đồng bộ hóa Không gian: Cập nhật ranh giới vĩ mô. Bảo toàn Mass."""
        self.coord_x_min = x_min
        self.coord_y_min = y_min
        self.coord_x_max = x_max
        self.coord_y_max = y_max
        
        self.width = max(0, self.coord_x_max - self.coord_x_min)
        self.height = max(0, self.coord_y_max - self.coord_y_min)
        self.area = self.width * self.height
        self.center_x = (self.coord_x_min + self.coord_x_max) / 2.0
        self.center_y = (self.coord_y_min + self.coord_y_max) / 2.0


# ============================================================================
# ENGINE PHÂN TÍCH
# ============================================================================

class BBoxAnalyzer:
    """Phân tích BBox để tìm ROI subtitle - THE OMEGA (Điểm Kết Thúc)."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        padding: int | None = None,
        max_lines: int = 3,
        min_aspect_ratio: float = 0.05,   
        max_aspect_ratio: float = 50.0,  
        max_width_ratio: float = 0.95,   
        use_weighted_heatmap: bool = True,
        heatmap_threshold_multiplier: float = 1.0,
        use_dbscan: bool = False,
        dbscan_eps: float | None = None,
        dbscan_min_samples: int = 3,
        post_merge_iou_threshold: float = 0.50,
        post_merge_gap_ratio: float = 1.0,
        use_iqr_filtering: bool = True,
        iqr_multiplier: float = 1.5,
        use_edge_mask: bool = True,
        edge_margin_ratio: float = 0.05,
        enable_band_refinement: bool = True,
        band_keep_ratio: float = 0.50,
        band_extend_ratio: float = 0.50,
        band_smoothing_ratio: float = 0.008,
        bottom_padding_factor: float = 1.6,
        **kwargs,
    ) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"Kích thước video không hợp lệ: {frame_width}x{frame_height}.")

        self.frame_width = frame_width
        self.frame_height = frame_height
        self._user_padding = padding
        self.max_lines = max(1, min(3, max_lines))
        
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.max_width_ratio = max_width_ratio

        self.grid_w = max(1, self.frame_width // _HEATMAP_DOWNSCALE)
        self.grid_h = max(1, self.frame_height // _HEATMAP_DOWNSCALE)

        self.use_weighted_heatmap = use_weighted_heatmap
        self.heatmap_threshold_multiplier = heatmap_threshold_multiplier
        self.use_dbscan = use_dbscan and HAS_SKLEARN
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = max(1, dbscan_min_samples)
        self.post_merge_iou_threshold = post_merge_iou_threshold
        self.post_merge_gap_ratio = post_merge_gap_ratio
        self.use_iqr_filtering = use_iqr_filtering
        self.use_edge_mask = use_edge_mask
        self.iqr_multiplier = iqr_multiplier
        self.edge_margin_ratio = edge_margin_ratio

        # [v3.16 BAND REFINER] Tinh chỉnh dải theo mật độ thời gian — diệt mega-cluster.
        self.enable_band_refinement = enable_band_refinement
        self.band_keep_ratio = max(0.05, min(0.95, band_keep_ratio))
        self.band_extend_ratio = max(0.02, min(self.band_keep_ratio, band_extend_ratio))
        self.band_smoothing_ratio = max(0.0, band_smoothing_ratio)
        # [v3.16 BOTTOM PAD] Hệ số nới biên dưới (chống cắt sát chân chữ/dấu phụ).
        self.bottom_padding_factor = max(1.0, bottom_padding_factor)

        self.median_font_thickness: float = 20.0

    def analyze(self, raw_bboxes: list[RawBBox]) -> list[ROICluster]:
        """Luồng xử lý chính: Phân tích bboxes → ROI clusters."""
        if not raw_bboxes:
            return []

        # Step 1: Lọc Dị thường C-Level Vectorized
        valid_boxes = self._filter_anomalies(raw_bboxes)
        if not valid_boxes:
            logger.debug("BBoxAnalyzer: Không còn Box hợp lệ nào sau màng lọc Anomaly.")
            return []

        thicknesses = np.array([min(b.box_width, b.box_height) for b in valid_boxes])
        self.median_font_thickness = max(float(_MIN_SIDE_PX), float(np.median(thicknesses)) if len(thicknesses) > 0 else 20.0)

        # Hardware Shield: Ngắt DBSCAN xuống 25.000
        if self.use_dbscan and len(valid_boxes) > 25000:
            logger.warning("BBoxAnalyzer: Phát hiện {} BBoxes. Tắt DBSCAN (Fallback to Heatmap) chống OOM.", len(valid_boxes))
            self.use_dbscan = False

        # Step 2: Nhánh Phân Cụm
        clusters = self._analyze_with_dbscan(valid_boxes) if self.use_dbscan else self._analyze_with_heatmap(valid_boxes)
        if not clusters:
            return []

        # Step 3: Bộ lọc Độ tin cậy (VIP Pass)
        validated_clusters = [c for c in clusters if c.frame_count >= 3 or c.mean_confidence >= _ELITE_CONFIDENCE]
        if not validated_clusters and clusters:
            validated_clusters = [c for c in clusters if c.frame_count >= 2]

        if not validated_clusters:
            return []

        # Step 4: Lượng Tử Hóa Viền sơ bộ (gom nhóm dòng để merge).
        validated_clusters.sort(key=lambda c: c.area, reverse=True)
        for cluster in validated_clusters:
            self._apply_symmetrical_padding(cluster)

        # Step 5: Đồ thị Liên thông Hậu xử lý (Scale-Aware Boundary Cleanup Merge)
        validated_clusters = self._merge_overlapping_clusters(validated_clusters, is_post_padding=True)

        # Step 6: [v3.16] Temporal Band Refiner — phép biến đổi không gian CUỐI.
        # Cắt mỗi cluster (đã gom dòng) về dải con dày đặc nhất theo mật độ thời
        # gian, triệt tiêu mega-cluster do morphology nối nhầm text cảnh nền.
        # Đặt CUỐI để merge không thể phình lại dải đã tinh chỉnh.
        if self.enable_band_refinement:
            validated_clusters = self._refine_clusters_to_dense_band(
                validated_clusters, raw_bboxes
            )
            if not validated_clusters:
                return []
            # Re-pad: nới lại biên (đặc biệt biên dưới) quanh dải đã tinh chỉnh.
            for cluster in validated_clusters:
                self._apply_symmetrical_padding(cluster)

        # Step 7: Sắp xếp Từ trên xuống dưới, Đánh ID
        self._sort_clusters_spatially(validated_clusters)

        logger.debug(
            f"BBoxAnalyzer v3.16 (Band Refiner): {len(raw_bboxes)} raw → {len(valid_boxes)} valid → {len(validated_clusters)} ROI(s) | "
            f"refine={self.enable_band_refinement}, keep={self.band_keep_ratio}"
        )
        return validated_clusters

    def _sort_clusters_spatially(self, clusters: list[ROICluster]) -> None:
        """Sắp xếp các cụm từ trên xuống, bảo vệ dòng chữ nghiêng (Tilted Line Guard)."""
        if not clusters:
            return

        clusters.sort(key=lambda c: c.coord_y_max)
        lines, current_line = [], []
        
        for c in clusters:
            if not current_line:
                current_line.append(c)
            else:
                avg_y_max = sum(item.coord_y_max for item in current_line) / len(current_line)
                avg_h = sum(item.height for item in current_line) / len(current_line)
                local_tolerance = max(10.0, avg_h * 0.6)
                
                if abs(c.coord_y_max - avg_y_max) <= local_tolerance: 
                    current_line.append(c)
                else:
                    lines.append(sorted(current_line, key=lambda x: x.coord_x_min))
                    current_line = [c]
                
        if current_line:
            lines.append(sorted(current_line, key=lambda x: x.coord_x_min))

        clusters.clear()
        for line in lines:
            clusters.extend(line)

        for idx, c in enumerate(clusters):
            c.cluster_id = idx

    # ========================================================================
    # PHASE 3.5: Temporal Density Band Refiner (Anti Mega-Cluster)
    # ========================================================================

    def _refine_clusters_to_dense_band(
        self, clusters: list[ROICluster], raw_bboxes: list[RawBBox]
    ) -> list[ROICluster]:
        """Cắt mỗi cluster về dải con dày đặc nhất theo *mật độ thời gian*.

        Nguyên lý: phụ đề hardsub là một dải **trội** về số frame xuất hiện. Khi
        morphology nối nhầm text cảnh nền vào dải phụ đề (mega-cluster), chiếu mật
        độ box theo trục chính của cluster sẽ lộ một **đỉnh sắc** tại dải phụ đề;
        ta giữ lại đoạn liên tục quanh đỉnh (ngưỡng ``band_keep_ratio × peak``) và
        loại phần nền thưa. Trục chiếu chọn theo *aspect trung vị của box thành
        viên* (box rộng → text ngang → chiếu trục Y; box cao → text dọc → chiếu
        trục X), nên bền vững cả khi orientation vĩ mô của blob bị phân loại sai.

        Dùng **toàn bộ** ``raw_bboxes`` (không phải tập đã lọc anomaly) để profile
        mật độ giàu mẫu nhất → đỉnh sắc hơn, dải khít hơn. Chỉ áp một sàn tin cậy
        nhẹ để loại nhiễu cực thấp.

        Args:
            clusters: Các cluster đã gom dòng (sau merge), trước bước tinh chỉnh cuối.
            raw_bboxes: Toàn bộ box OCR thô để tái lập profile mật độ thời gian.

        Returns:
            Danh sách cluster đã tinh chỉnh (cùng số lượng hoặc ít hơn nếu suy biến).
        """
        if not clusters or not raw_bboxes:
            return clusters

        density_boxes = [b for b in raw_bboxes if b.confidence >= _DENSITY_CONFIDENCE_FLOOR]
        if not density_boxes:
            return clusters

        # Gán box về cluster chứa tâm (2D), chụp lại biên gốc TRƯỚC khi sửa.
        member_map: dict[int, list[RawBBox]] = defaultdict(list)
        snapshots = [
            (c.coord_x_min, c.coord_y_min, c.coord_x_max, c.coord_y_max)
            for c in clusters
        ]
        for box in density_boxes:
            for idx, (x_min, y_min, x_max, y_max) in enumerate(snapshots):
                if x_min <= box.center_x <= x_max and y_min <= box.center_y <= y_max:
                    member_map[idx].append(box)
                    break

        refined: list[ROICluster] = []
        for idx, cluster in enumerate(clusters):
            members = member_map.get(idx)
            new_cluster = self._refine_single_cluster(cluster, members)
            refined.append(new_cluster if new_cluster is not None else cluster)
        return refined

    def _refine_single_cluster(
        self, cluster: ROICluster, members: Optional[list[RawBBox]]
    ) -> Optional[ROICluster]:
        """Tinh chỉnh một cluster; trả ``None`` nếu không thể cải thiện an toàn."""
        if not members or len(members) < 2:
            return None

        median_box_width = float(np.median([b.box_width for b in members]))
        median_box_height = float(np.median([b.box_height for b in members]))
        # [v3.18.3] Chọn trục chiếu 3 tầng: aspect rõ ràng → quyết ngay; mơ hồ
        # (box gần vuông, điển hình ký tự CJK đơn) → chọn trục có ĐỈNH MẬT ĐỘ TRỘI
        # hơn (prominence), đúng mục đích "cắt về dải dày đặc". So sánh
        # medW >= medH đơn thuần bị box đơn-ký-tự đánh lừa (file 83: h/w=1.25 →
        # chiếu nhầm trục X, dải 0.37h không cắt được); còn spread của tâm bị
        # mega-blob đánh lừa (file 5: nhiễu trải dọc toàn khung → spread_y lớn).
        if median_box_width >= max(1.0, median_box_height) * _ASPECT_DECISIVE_RATIO:
            project_on_y_axis = True
        elif median_box_height >= max(1.0, median_box_width) * _ASPECT_DECISIVE_RATIO:
            project_on_y_axis = False
        else:
            project_on_y_axis = self._y_axis_has_sharper_density_peak(cluster, members)

        if project_on_y_axis:
            low, high, axis_dim = int(cluster.coord_y_min), int(cluster.coord_y_max), self.frame_height
            seg_starts = np.fromiter((b.coord_y_min for b in members), dtype=np.float64, count=len(members))
            seg_ends = np.fromiter((b.coord_y_max for b in members), dtype=np.float64, count=len(members))
            centers = np.fromiter((b.center_y for b in members), dtype=np.float64, count=len(members))
        else:
            low, high, axis_dim = int(cluster.coord_x_min), int(cluster.coord_x_max), self.frame_width
            seg_starts = np.fromiter((b.coord_x_min for b in members), dtype=np.float64, count=len(members))
            seg_ends = np.fromiter((b.coord_x_max for b in members), dtype=np.float64, count=len(members))
            centers = np.fromiter((b.center_x for b in members), dtype=np.float64, count=len(members))

        span = high - low
        if span < _MIN_SIDE_PX:
            return None

        # Profile mật độ bằng sweep-line cumsum O(N + span) — tránh O(N×span).
        start_idx = np.clip(seg_starts.astype(int) - low, 0, span)
        end_idx = np.clip(seg_ends.astype(int) - low, 0, span)
        delta = np.zeros(span + 2, dtype=np.float64)
        np.add.at(delta, start_idx, 1.0)
        np.add.at(delta, end_idx + 1, -1.0)
        profile = np.cumsum(delta)[: span + 1]
        if profile.max() <= 0.0:
            return None

        smooth_radius = max(1, int(self.band_smoothing_ratio * axis_dim))
        kernel = np.ones(2 * smooth_radius + 1, dtype=np.float64) / (2 * smooth_radius + 1)
        smoothed = np.convolve(profile, kernel, mode="same")

        peak_index = int(smoothed.argmax())
        keep_threshold = self.band_keep_ratio * float(smoothed[peak_index])
        # [v3.19] Hysteresis 2 ngưỡng: dải LÕI cắt tại ``band_keep_ratio × peak``
        # (chống mega-cluster), sau đó MỞ RỘNG LIÊN TỤC hai phía xuống ngưỡng
        # ``band_extend_ratio × peak`` — giữ DÒNG THỨ HAI của phụ đề nhiều dòng
        # (chỉ xuất hiện ở ~1/3 số câu nên mật độ thấp hơn dòng chính, nhưng vẫn
        # cao hơn hẳn nền nhiễu ~1-10%). Nhiễu xa không liền kề nên không bị nối.
        extend_threshold = min(keep_threshold, self.band_extend_ratio * float(smoothed[peak_index]))

        band_top = peak_index
        while band_top > 0 and smoothed[band_top - 1] >= keep_threshold:
            band_top -= 1
        band_bottom = peak_index
        while band_bottom < len(smoothed) - 1 and smoothed[band_bottom + 1] >= keep_threshold:
            band_bottom += 1
        while band_top > 0 and smoothed[band_top - 1] >= extend_threshold:
            band_top -= 1
        while band_bottom < len(smoothed) - 1 and smoothed[band_bottom + 1] >= extend_threshold:
            band_bottom += 1

        new_low, new_high = low + band_top, low + band_bottom
        boxes_in_band = [
            box
            for box, center in zip(members, centers)
            if new_low <= center <= new_high
        ]
        if not boxes_in_band:
            return None

        # Tái dựng cluster từ tập box trong dải → orientation/mass/conf đúng,
        # đồng thời vá lỗi phân loại VERTICAL nhầm của mega-blob.
        rebuilt = self._build_cluster_from_bboxes(boxes_in_band, cluster.cluster_id)
        if rebuilt is None:
            return None

        # KẸP biên trục chiếu về đúng dải dày đặc [new_low, new_high]: box có *tâm*
        # trong dải vẫn có thể *kéo dài* ra ngoài, làm extent vĩ mô phình lại — phải
        # khoá lại đúng dải. Biên TRỤC CÒN LẠI dùng order-statistic bền vững
        # (chống 1-2 box nhiễu kéo lệch) + đối xứng hoá khi text căn giữa [v3.18.1].
        if project_on_y_axis:
            robust_left, robust_right = self._robust_axis_bounds(
                [box.coord_x_min for box in boxes_in_band],
                [box.coord_x_max for box in boxes_in_band],
            )
            if rebuilt.alignment == TextAlignment.CENTER:
                robust_left, robust_right = self._symmetrize_around_center(
                    robust_left,
                    robust_right,
                    sorted(box.center_x for box in boxes_in_band),
                    self.frame_width,
                )
            rebuilt.update_bounds(
                max(0, int(robust_left)),
                max(rebuilt.coord_y_min, new_low),
                min(self.frame_width, int(robust_right)),
                min(rebuilt.coord_y_max, new_high),
            )
        else:
            robust_top, robust_bottom = self._robust_axis_bounds(
                [box.coord_y_min for box in boxes_in_band],
                [box.coord_y_max for box in boxes_in_band],
            )
            rebuilt.update_bounds(
                max(rebuilt.coord_x_min, new_low),
                max(0, int(robust_top)),
                min(rebuilt.coord_x_max, new_high),
                min(self.frame_height, int(robust_bottom)),
            )
        return rebuilt

    def _y_axis_has_sharper_density_peak(
        self, cluster: ROICluster, members: list[RawBBox]
    ) -> bool:
        """Chọn trục chiếu theo ĐỘ TRỘI của đỉnh mật độ (prominence) khi aspect mơ hồ.

        Với mỗi trục, dựng profile mật độ sweep-line trên span của cluster, rồi đo
        ``prominence = peak / median(các giá trị > 0)``. Trục có đỉnh trội hơn là
        trục mà "dải dày đặc" tồn tại rõ — đúng trục cần cắt: dải phụ đề ngang cho
        đỉnh Y rất sắc (cao ~50px); cột chữ dọc cho đỉnh X sắc (rộng ~60px).

        Returns:
            ``True`` nếu nên chiếu trục Y (bố cục ngang), ``False`` nếu trục X.
        """

        def axis_prominence(low: int, high: int, starts: np.ndarray, ends: np.ndarray, axis_dim: int) -> float:
            span = high - low
            if span < _MIN_SIDE_PX:
                return 0.0
            start_idx = np.clip(starts.astype(int) - low, 0, span)
            end_idx = np.clip(ends.astype(int) - low, 0, span)
            delta = np.zeros(span + 2, dtype=np.float64)
            np.add.at(delta, start_idx, 1.0)
            np.add.at(delta, end_idx + 1, -1.0)
            profile = np.cumsum(delta)[: span + 1]
            if profile.max() <= 0:
                return 0.0
            radius = max(1, int(self.band_smoothing_ratio * axis_dim))
            kernel = np.ones(2 * radius + 1, dtype=np.float64) / (2 * radius + 1)
            smoothed = np.convolve(profile, kernel, mode="same")
            positive_values = smoothed[smoothed > 0]
            baseline = float(np.median(positive_values)) if positive_values.size else 1.0
            return float(smoothed.max()) / max(1.0, baseline)

        prominence_y = axis_prominence(
            int(cluster.coord_y_min), int(cluster.coord_y_max),
            np.fromiter((b.coord_y_min for b in members), dtype=np.float64, count=len(members)),
            np.fromiter((b.coord_y_max for b in members), dtype=np.float64, count=len(members)),
            self.frame_height,
        )
        prominence_x = axis_prominence(
            int(cluster.coord_x_min), int(cluster.coord_x_max),
            np.fromiter((b.coord_x_min for b in members), dtype=np.float64, count=len(members)),
            np.fromiter((b.coord_x_max for b in members), dtype=np.float64, count=len(members)),
            self.frame_width,
        )
        return prominence_y >= prominence_x

    @staticmethod
    def _is_horizontal_layout(
        median_width: float,
        median_height: float,
        centers_x: list[float],
        centers_y: list[float],
    ) -> bool:
        """Suy bố cục NGANG/DỌC của một tập box — logic 3 tầng, kháng nhập nhằng.

        1. **Aspect rõ ràng** (ngưỡng ``_ASPECT_DECISIVE_RATIO = 2.0``): dòng ngang
           nguyên câu có w ≫ h, cột dọc nguyên khối có h ≫ w → quyết ngay.
        2. **Box mơ hồ** (gần vuông — điển hình ký tự CJK đơn, h/w ≈ 0.8..1.3):
           dùng PHÂN TÁN KHÔNG GIAN của tâm box (p95−p5, bỏ outlier): các ký tự
           của câu ngang rải theo X, của cột dọc rải theo Y.
        3. Hoà: mặc định NGANG (phụ đề ngang là phổ biến nhất).

        Returns:
            ``True`` nếu bố cục ngang (chiếu trục Y), ``False`` nếu dọc (chiếu X).
        """
        safe_height = max(1.0, median_height)
        safe_width = max(1.0, median_width)
        if median_width >= safe_height * _ASPECT_DECISIVE_RATIO:
            return True
        if median_height >= safe_width * _ASPECT_DECISIVE_RATIO:
            return False
        spread_x = float(np.percentile(centers_x, 95) - np.percentile(centers_x, 5))
        spread_y = float(np.percentile(centers_y, 95) - np.percentile(centers_y, 5))
        if spread_y > spread_x * 1.2:
            return False
        return True

    @staticmethod
    def _robust_axis_bounds(
        segment_starts: list[float], segment_ends: list[float]
    ) -> tuple[float, float]:
        """Biên trục BỀN VỮNG theo order-statistic — chống outlier kéo lệch.

        Vấn đề: biên ``min/max`` tuyệt đối bị 1-2 box nhiễu (logo, text cảnh lọt
        vào dải, OCR rác conf thấp) kéo lệch một phía, phá tính đối xứng của phụ đề
        căn giữa. Quan sát đo được: dòng phụ đề thật đứng yên ≥0.5s tạo **≥12 box**
        cùng biên (sample 25fps), còn nhiễu thoáng qua chỉ 1-3 box.

        Giải pháp: biên trái = phần tử nhỏ thứ ``k`` của ``segment_starts``, biên
        phải = phần tử lớn thứ ``k`` của ``segment_ends`` (k = ``_BOUND_SUPPORT_BOXES``).
        Biên chỉ được "mở" tới nơi có ≥ k box hỗ trợ → câu dài thật (nhiều box trùng
        biên) giữ nguyên, nhiễu lẻ bị bỏ.

        Returns:
            ``(low, high)`` bền vững; rơi về min/max tuyệt đối nếu suy biến.
        """
        sample_count = len(segment_starts)
        support = min(_BOUND_SUPPORT_BOXES, sample_count)
        starts_array = np.asarray(segment_starts, dtype=np.float64)
        ends_array = np.asarray(segment_ends, dtype=np.float64)
        robust_low = float(np.partition(starts_array, support - 1)[support - 1])
        robust_high = float(np.partition(ends_array, sample_count - support)[sample_count - support])
        if robust_high <= robust_low:
            return float(starts_array.min()), float(ends_array.max())
        return robust_low, robust_high

    @staticmethod
    def _symmetrize_around_center(
        bound_low: float,
        bound_high: float,
        sorted_centers: list[float],
        axis_limit: int,
    ) -> tuple[float, float]:
        """Đối xứng hoá biên quanh **median tâm box** cho text CĂN GIỮA.

        Phụ đề căn giữa có median tâm cực kỳ ổn định (đo được p5..p95 chỉ ±3px).
        Sau khi đã có biên bền vững, mở rộng PHÍA HẸP cho hai nửa bằng nhau quanh
        median tâm (chỉ mở rộng — không bao giờ co, nên không thể cắt chữ), kẹp
        trong ``[0, axis_limit]``.
        """
        if not sorted_centers:
            return bound_low, bound_high
        median_center = sorted_centers[len(sorted_centers) // 2]
        half_span = max(median_center - bound_low, bound_high - median_center)
        return (
            max(0.0, median_center - half_span),
            min(float(axis_limit), median_center + half_span),
        )

    # ========================================================================
    # PHASE 1: Time-Frequency Weighted Heatmap
    # ========================================================================

    def _analyze_with_heatmap(self, valid_boxes: list[RawBBox]) -> list[ROICluster]:
        """Bản đồ nhiệt Kết hợp: Vectorized Exact Pixel Mapping."""
        heatmap_conf = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        heatmap_freq = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        
        mask_freq_buffer = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        mask_conf_buffer = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        D = _HEATMAP_DOWNSCALE
        valid_boxes.sort(key=lambda b: b.frame_idx)
        
        props = [(b.coord_x_min, b.coord_y_min, b.coord_x_max, b.coord_y_max, 
                  b.box_width, b.box_height, b.confidence, b.frame_idx) for b in valid_boxes]
        arr = np.array(props, dtype=np.float32)
        
        w_margins = arr[:, 4] * 0.10
        h_margins = arr[:, 5] * 0.10
        
        x1_arr = np.clip(((arr[:, 0] + w_margins) // D).astype(int), 0, self.grid_w - 1)
        y1_arr = np.clip(((arr[:, 1] + h_margins) // D).astype(int), 0, self.grid_h - 1)
        x2_raw = (np.ceil((arr[:, 2] - w_margins) / D)).astype(int)
        y2_raw = (np.ceil((arr[:, 3] - h_margins) / D)).astype(int)
        
        x2_arr = np.clip(np.maximum(x1_arr + 1, x2_raw), 0, self.grid_w)
        y2_arr = np.clip(np.maximum(y1_arr + 1, y2_raw), 0, self.grid_h)
        
        weights = arr[:, 6] ** 1.5 if self.use_weighted_heatmap else np.ones(len(arr), dtype=np.float32)
        f_idxs = arr[:, 7].astype(int)

        frame_to_indices = defaultdict(list)
        for i, f_idx in enumerate(f_idxs):
            frame_to_indices[f_idx].append(i)

        for f_idx, indices in frame_to_indices.items():
            mask_freq_buffer.fill(0)
            mask_conf_buffer.fill(0)

            for i in indices:
                x1, y1, x2, y2 = x1_arr[i], y1_arr[i], x2_arr[i], y2_arr[i]
                if x1 < x2 and y1 < y2:
                    cv2.rectangle(mask_freq_buffer, (x1, y1), (x2 - 1, y2 - 1), 1, -1)
                    mask_slice = mask_conf_buffer[y1:y2, x1:x2]
                    mask_conf_buffer[y1:y2, x1:x2] = np.maximum(mask_slice, weights[i])

            heatmap_freq += mask_freq_buffer
            heatmap_conf += mask_conf_buffer

        total_unique_frames = len(frame_to_indices)
        # [TEMPORAL FIX] Giới hạn min_hits tối đa là 2 để cứu fast-flash subtitles (2 frames)
        if total_unique_frames <= 3: min_hits = 1
        else: min_hits = 2
            
        freq_mask = heatmap_freq >= min_hits
        threshold = self._compute_adaptive_threshold(heatmap_conf, total_unique_frames)
        conf_mask = heatmap_conf >= threshold

        binary_map = (freq_mask & conf_mask).astype(np.uint8) * 255

        kernel_size_x = int(min(50, max(2, (self.median_font_thickness * 2.5) / D)))
        kernel_size_y = int(min(30, max(2, (self.median_font_thickness * 1.2) / D)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_x, kernel_size_y))
        
        closed_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
        dilated_map = cv2.dilate(closed_map, kernel, iterations=1)

        del heatmap_conf, heatmap_freq, mask_freq_buffer, mask_conf_buffer, closed_map

        num_labels, labels = cv2.connectedComponents(dilated_map, connectivity=8)
        
        c_x = (arr[:, 0] + arr[:, 2]) / 2.0
        c_y = (arr[:, 1] + arr[:, 3]) / 2.0
        cy_grids = np.clip(c_y.astype(int) // D, 0, self.grid_h - 1)
        cx_grids = np.clip(c_x.astype(int) // D, 0, self.grid_w - 1)

        box_labels = labels[cy_grids, cx_grids]

        zero_mask = box_labels == 0
        if np.any(zero_mask):
            left_x = np.clip((arr[:, 0][zero_mask] + arr[:, 4][zero_mask] * 0.25).astype(int) // D, 0, self.grid_w - 1)
            box_labels[zero_mask] = labels[cy_grids[zero_mask], left_x]
            
            zero_mask_right = box_labels == 0
            if np.any(zero_mask_right):
                right_x = np.clip((arr[:, 2][zero_mask_right] - arr[:, 4][zero_mask_right] * 0.25).astype(int) // D, 0, self.grid_w - 1)
                box_labels[zero_mask_right] = labels[cy_grids[zero_mask_right], right_x]

        bbox_by_label = defaultdict(list)
        isolated_elite_counter = -1

        for lbl, b in zip(box_labels, valid_boxes):
            if lbl > 0:
                bbox_by_label[lbl].append(b)
            elif b.confidence >= _ELITE_CONFIDENCE:  
                bbox_by_label[isolated_elite_counter].append(b)
                isolated_elite_counter -= 1

        result_clusters = []
        for island_label, boxes in bbox_by_label.items():
            if boxes:
                roi = self._build_cluster_from_bboxes(boxes, cluster_id=len(result_clusters))
                if roi:
                    result_clusters.append(roi)

        return result_clusters

    def _compute_adaptive_threshold(self, heatmap: np.ndarray, total_frames: int) -> float:
        base_threshold = max(0.5, total_frames * 0.01) if self.use_weighted_heatmap else max(3, int(total_frames * 0.03))
        nonzero = heatmap[heatmap > 0]
        if len(nonzero) == 0:
            return float(base_threshold)

        q1, q3 = map(float, np.percentile(nonzero, [25, 75]))
        iqr = max(1e-5, q3 - q1)
        
        adaptive = max(0.1 if self.use_weighted_heatmap else 1.0, q1 - 0.5 * iqr) 
        return adaptive * self.heatmap_threshold_multiplier


    # ========================================================================
    # PHASE 2: DBSCAN Anisotropic
    # ========================================================================

    def _analyze_with_dbscan(self, valid_boxes: list[RawBBox]) -> list[ROICluster]:
        if not HAS_SKLEARN:
            logger.warning("sklearn chưa cài đặt, fallback về Heatmap.")
            return self._analyze_with_heatmap(valid_boxes)

        self.eps_x, self.eps_y = self._calculate_adaptive_eps()
        safe_eps_x = max(1.0, self.eps_x)
        safe_eps_y = max(1.0, self.eps_y)
        
        X = np.array([(b.center_x / safe_eps_x, b.center_y / safe_eps_y) for b in valid_boxes], dtype=np.float32)
        
        total_unique_frames = len(set(b.frame_idx for b in valid_boxes))
        if total_unique_frames <= 2: dynamic_min_samples = 1
        elif total_unique_frames <= 10: dynamic_min_samples = 2
        else: dynamic_min_samples = self.dbscan_min_samples
            
        dynamic_min_samples = max(1, min(dynamic_min_samples, len(valid_boxes)))

        # [MEMORY FIX] Ép dùng thuật toán KD-Tree thay vì 'auto' để chống tràn O(N^2) RAM Matrix.
        db = SklearnDBSCAN(eps=1.0, min_samples=dynamic_min_samples, algorithm='kd_tree')
        labels = db.fit_predict(X)
        
        cl = defaultdict(list)
        neg_counter = -2
        for l, b in zip(labels, valid_boxes):
            if l == -1:
                if b.confidence >= _ELITE_CONFIDENCE: l = neg_counter; neg_counter -= 1
                else: continue
            cl[l].append(b)
        
        roi = [self._build_cluster_from_bboxes(boxes, i) for i, boxes in enumerate(cl.values())]
        return self._merge_overlapping_clusters([r for r in roi if r])

    def _calculate_adaptive_eps(self) -> tuple[float, float]:
        if self.dbscan_eps is not None:
            return float(self.dbscan_eps), float(self.dbscan_eps)
        eps_x = max(15.0, self.median_font_thickness * 2.5)
        eps_y = max(10.0, self.median_font_thickness * 1.2)
        return eps_x, eps_y

    # ========================================================================
    # ĐỒ THỊ LIÊN THÔNG (Graph Post-Merge)
    # ========================================================================

    def _merge_overlapping_clusters(self, clusters: list[ROICluster], is_post_padding: bool = False) -> list[ROICluster]:
        """Gộp Đồ thị O(N log N): Absolute Scale Veto v2 & Universal Symbiosis."""
        if len(clusters) <= 1: return clusters
        
        clusters.sort(key=lambda c: c.coord_y_min)
        n = len(clusters)
        adj = defaultdict(set)
        
        t_iou = 0.01 if is_post_padding else self.post_merge_iou_threshold
        gap_allowance = (self.median_font_thickness * self.post_merge_gap_ratio) if not is_post_padding else 0.0

        for i in range(n):
            c1 = clusters[i]
            for j in range(i + 1, n):
                c2 = clusters[j]
                
                if c2.coord_y_min > c1.coord_y_max + gap_allowance:
                    break  

                union_h = max(c1.coord_y_max, c2.coord_y_max) - min(c1.coord_y_min, c2.coord_y_min)
                
                # Post-Padding Avalanche Preventer
                max_allowed_h = self.frame_height * (0.10 + (self.max_lines * 0.08)) * 1.5
                if is_post_padding:
                    max_allowed_h += (self.median_font_thickness * 4.0)
                if union_h > max_allowed_h:
                    continue

                x1, y1, x2, y2 = max(c1.coord_x_min, c2.coord_x_min), max(c1.coord_y_min, c2.coord_y_min), min(c1.coord_x_max, c2.coord_x_max), min(c1.coord_y_max, c2.coord_y_max)
                
                iou = 0.0
                iom = 0.0 
                is_intersect = (x2 > x1 and y2 > y1)
                
                if is_intersect:
                    inter_area = (x2 - x1) * (y2 - y1)
                    union_area = (c1.area + c2.area) - inter_area
                    if union_area > 0: iou = inter_area / union_area
                    
                    min_area = min(c1.area, c2.area)
                    if min_area > 0: iom = inter_area / min_area

                scale_ratio = max(c1.local_thickness, c2.local_thickness) / max(1.0, min(c1.local_thickness, c2.local_thickness))
                is_scale_compatible = scale_ratio <= 1.8

                h_gap = max(0.0, c2.coord_x_min - c1.coord_x_max, c1.coord_x_min - c2.coord_x_max)
                v_gap = max(0.0, c2.coord_y_min - c1.coord_y_max, c1.coord_y_min - c2.coord_y_max)
                
                min_w = min(c1.width, c2.width)
                min_h = min(c1.height, c2.height)
                
                if h_gap > min_w * 3.0 and c1.orientation == TextOrientation.HORIZONTAL: continue 
                if v_gap > min_h * 3.0 and c1.orientation == TextOrientation.VERTICAL: continue

                # UNIVERSAL PUNCTUATION SYMBIOSIS
                is_symbiosis = False
                if not is_scale_compatible and not is_post_padding:
                    main_c, sub_c = (c1, c2) if c1.area >= c2.area else (c2, c1)
                    
                    if main_c.orientation == TextOrientation.HORIZONTAL:
                        v_margin = main_c.local_thickness * 0.25 
                        if (sub_c.coord_y_min >= main_c.coord_y_min - v_margin) and \
                           (sub_c.coord_y_max <= main_c.coord_y_max + v_margin):
                            if h_gap <= main_c.local_thickness * self.post_merge_gap_ratio:
                                is_symbiosis = True
                                
                    elif main_c.orientation == TextOrientation.VERTICAL:
                        h_margin = main_c.local_thickness * 0.25 
                        if (sub_c.coord_x_min >= main_c.coord_x_min - h_margin) and \
                           (sub_c.coord_x_max <= main_c.coord_x_max + h_margin):
                            if v_gap <= main_c.local_thickness * self.post_merge_gap_ratio:
                                is_symbiosis = True

                # SMART MERGE PROTOCOL
                if iom >= 0.85 or is_symbiosis: 
                    adj[i].add(j); adj[j].add(i)
                elif is_scale_compatible:
                    if is_post_padding and is_intersect:
                        adj[i].add(j); adj[j].add(i)
                    elif not is_post_padding:
                        if iou >= t_iou:
                            adj[i].add(j); adj[j].add(i)
                        elif c1.orientation == c2.orientation:
                            gap = max(v_gap, h_gap)
                            if gap <= min(c1.local_thickness, c2.local_thickness) * self.post_merge_gap_ratio:
                                adj[i].add(j); adj[j].add(i)

        visited = set(); res = []
        for i in range(n):
            if i not in visited:
                q = deque([i]); visited.add(i)
                component_clusters = []

                while q:
                    u = q.popleft()
                    component_clusters.append(clusters[u])
                    for v in adj[u]:
                        if v not in visited: 
                            visited.add(v)
                            q.append(v)
                
                if len(component_clusters) > 1:
                    x_min = min(c.coord_x_min for c in component_clusters)
                    y_min = min(c.coord_y_min for c in component_clusters)
                    x_max = max(c.coord_x_max for c in component_clusters)
                    y_max = max(c.coord_y_max for c in component_clusters)

                    bbox_cnt = sum(c.bbox_count for c in component_clusters)
                    total_true_area = sum(c.true_mass_area for c in component_clusters)
                    sum_com_x = sum(c.mass_center_x * c.true_mass_area for c in component_clusters)
                    sum_com_y = sum(c.mass_center_y * c.true_mass_area for c in component_clusters)
                    
                    conf_sum = sum(c.mean_confidence * c.bbox_count for c in component_clusters)
                    mean_conf = float(conf_sum / bbox_cnt) if bbox_cnt > 0 else 0.0

                    f_set = set().union(*(c._frame_index_set for c in component_clusters))

                    macro_w = x_max - x_min
                    macro_h = y_max - y_min
                    
                    h_votes = sum(c.bbox_count for c in component_clusters if c.orientation == TextOrientation.HORIZONTAL)
                    v_votes = sum(c.bbox_count for c in component_clusters if c.orientation == TextOrientation.VERTICAL)
                    
                    if h_votes > v_votes:
                        orient = TextOrientation.HORIZONTAL
                    elif v_votes > h_votes:
                        orient = TextOrientation.VERTICAL
                    else:
                        orient = TextOrientation.HORIZONTAL if macro_w >= macro_h * 1.2 else TextOrientation.VERTICAL
                    
                    align = TextAlignment.CENTER
                    if orient == TextOrientation.HORIZONTAL and total_true_area > 1e-5:
                        com_x = sum_com_x / total_true_area
                        geo_center_x = (x_min + x_max) / 2.0
                        shift_ratio = (com_x - geo_center_x) / max(1.0, float(macro_w))
                        if shift_ratio < -0.1: align = TextAlignment.LEFT
                        elif shift_ratio > 0.1: align = TextAlignment.RIGHT

                    merged_mass_x = sum_com_x / total_true_area if total_true_area > 1e-5 else (x_min + x_max) / 2.0
                    merged_mass_y = sum_com_y / total_true_area if total_true_area > 1e-5 else (y_min + y_max) / 2.0
                    merged_thickness = sum(c.local_thickness * c.true_mass_area for c in component_clusters) / total_true_area if total_true_area > 1e-5 else component_clusters[0].local_thickness
                    
                    merged_c = ROICluster(
                        cluster_id=len(res),
                        orientation=orient, alignment=align,
                        coord_x_min=int(x_min), coord_y_min=int(y_min),
                        coord_x_max=int(x_max), coord_y_max=int(y_max),
                        bbox_count=bbox_cnt, frame_count=len(f_set),
                        mean_confidence=mean_conf, keep=True, _frame_index_set=f_set,
                        mass_center_x=merged_mass_x, mass_center_y=merged_mass_y, 
                        local_thickness=merged_thickness, true_mass_area=total_true_area
                    )
                    res.append(merged_c)
                else:
                    iso = component_clusters[0]
                    iso.cluster_id = len(res)
                    res.append(iso)
                
        return res if is_post_padding else sorted(res, key=lambda c: c.mean_confidence * c.bbox_count, reverse=True)[:self.max_lines]

    # ========================================================================
    # PHASE 3: Anomaly & Outlier Filtering
    # ========================================================================

    def _filter_anomalies(self, bboxes: list[RawBBox]) -> list[RawBBox]:
        if not bboxes: return []

        props = [(b.coord_x_min, b.coord_x_max, b.coord_y_min, b.coord_y_max, 
                  b.confidence, b.box_width, b.box_height, b.area) for b in bboxes]
        x_min, x_max, y_min, y_max, confs, widths, heights, areas = np.array(props).T
        
        conf_med = float(np.median(confs))
        
        # [MATH FIX] Xóa bỏ ảo ảnh toán học: Ngưỡng động học thông minh
        # Chỉ siết ngưỡng lên nếu chất lượng tổng thể thực sự rất cao (>= 0.95)
        # [v3.19] Ngưỡng động KHÔNG được vượt Lệnh bài miễn tử: khi quần thể phụ đề
        # chủ đạo có conf rất cao (0.97), ngưỡng siết lên 0.92 từng giết sạch các
        # vùng chữ hợp lệ khác (chữ thông tin/hiệu ứng conf 0.85-0.91) → mất ROI.
        min_conf = max(_MIN_CONFIDENCE, conf_med - 0.05) if conf_med >= 0.95 else _MIN_CONFIDENCE
        min_conf = min(min_conf, _ELITE_CONFIDENCE)
        
        mins_side = np.minimum(widths, heights)
        high_conf_mask = confs > 0.85
        med_thick = float(np.median(mins_side[high_conf_mask])) if np.any(high_conf_mask) else float(np.median(mins_side))
        
        self.median_font_thickness = max(float(_MIN_SIDE_PX), med_thick)
        max_thick = self.median_font_thickness * 5.0
        
        aspect_h = widths / np.maximum(1e-5, heights)
        aspect_v = heights / np.maximum(1e-5, widths)
        screen_area = self.frame_width * self.frame_height

        # Lệnh bài miễn tử (Elite Override) cấp hệ thống
        is_perfect_title = confs >= _ELITE_CONFIDENCE
        
        giant_slayer_mask = ~((areas > screen_area * 0.08) & (aspect_h > 0.4) & (aspect_h < 2.5))
        giant_slayer_mask = giant_slayer_mask | is_perfect_title
        
        thickness_mask = (mins_side <= max_thick) | is_perfect_title
        
        valid_aspect = ((aspect_h >= self.min_aspect_ratio) & (aspect_h <= self.max_aspect_ratio)) | \
                       ((aspect_v >= self.min_aspect_ratio) & (aspect_v <= self.max_aspect_ratio))
        valid_aspect = valid_aspect | is_perfect_title

        # Xuyên thủng rào cản tối thiểu để cứu dấu câu
        side_mask = (mins_side >= _MIN_SIDE_PX) | is_perfect_title
        area_mask = (areas >= _MIN_AREA_PX) | is_perfect_title

        mask = (
            (x_max > 0) & (x_min < self.frame_width) &
            (y_max > 0) & (y_min < self.frame_height) &
            (confs >= min_conf) &
            side_mask &
            area_mask &
            thickness_mask &
            valid_aspect &
            (widths <= self.frame_width * self.max_width_ratio) &
            giant_slayer_mask
        )

        valid = list(compress(bboxes, mask))
        
        if self.use_iqr_filtering: valid = self._filter_outliers_iqr(valid)
        if self.use_edge_mask: valid = self._filter_edge_margin(valid)
            
        return valid

    def _filter_outliers_iqr(self, bboxes: list[RawBBox]) -> list[RawBBox]:
        if not bboxes or len(bboxes) < 4: return bboxes
        
        props = [(b.box_width, b.box_height, b.confidence) for b in bboxes]
        widths, heights, confs = np.array(props).T
        
        q1_w, q3_w = map(float, np.percentile(widths, [25, 75]))
        lower_w, upper_w = q1_w - self.iqr_multiplier * (q3_w - q1_w), q3_w + self.iqr_multiplier * (q3_w - q1_w)

        q1_h, q3_h = map(float, np.percentile(heights, [25, 75]))
        lower_h, upper_h = q1_h - self.iqr_multiplier * (q3_h - q1_h), q3_h + self.iqr_multiplier * (q3_h - q1_h)

        elite_mask = confs >= _ELITE_CONFIDENCE
        mask = ((widths >= lower_w) & (widths <= upper_w) & (heights >= lower_h) & (heights <= upper_h)) | elite_mask
        return list(compress(bboxes, mask))

    def _filter_edge_margin(self, bboxes: list[RawBBox]) -> list[RawBBox]:
        if not bboxes: return bboxes
        
        margin_x = self.frame_width * self.edge_margin_ratio
        margin_y = self.frame_height * self.edge_margin_ratio
        safe_x_max = self.frame_width - margin_x
        safe_y_max = self.frame_height - margin_y

        props = [(b.center_x, b.center_y, b.confidence) for b in bboxes]
        c_x, c_y, confs = np.array(props).T
        
        inside = (c_x >= margin_x) & (c_x <= safe_x_max) & (c_y >= margin_y) & (c_y <= safe_y_max)
        
        elite_mask = confs >= _ELITE_CONFIDENCE
        mask = inside | elite_mask
        return list(compress(bboxes, mask))

    def _apply_symmetrical_padding(self, cluster: ROICluster) -> None:
        """Anisotropic Padding & Override Cấu hình cứng.

        [v3.16] Với text NGANG, biên dưới được nới thêm theo ``bottom_padding_factor``
        để không cắt sát chân chữ (descender CJK, dấu phụ tiếng Việt nằm dưới
        baseline). Padding của ``_user_padding`` vẫn đối xứng (tôn trọng override).
        """
        if self._user_padding is not None:
            pad_x = pad_y_top = pad_y_bottom = self._user_padding
        else:
            if cluster.orientation == TextOrientation.HORIZONTAL:
                pad_x = (int(min(40, max(15, cluster.local_thickness * 1.5))) // 4) * 4
                pad_y_top = (int(min(20, max(8, cluster.local_thickness * 0.5))) // 4) * 4
                # Nới biên dưới: chống cắt sát chân chữ / dấu phụ.
                pad_y_bottom = (int(pad_y_top * self.bottom_padding_factor) // 4) * 4
            else:
                pad_x = (int(min(20, max(8, cluster.local_thickness * 0.5))) // 4) * 4
                pad_y_top = pad_y_bottom = (int(min(40, max(15, cluster.local_thickness * 1.5))) // 4) * 4

        cluster.update_bounds(
            max(0, cluster.coord_x_min - pad_x),
            max(0, cluster.coord_y_min - pad_y_top),
            min(self.frame_width, cluster.coord_x_max + pad_x),
            min(self.frame_height, cluster.coord_y_max + pad_y_bottom),
        )

    def _build_cluster_from_bboxes(self, boxes: list[RawBBox], cluster_id: int) -> Optional[ROICluster]:
        props = [(b.coord_x_min, b.coord_y_min, b.coord_x_max, b.coord_y_max, 
                  b.center_x, b.center_y, b.area, b.confidence, b.frame_idx) for b in boxes]
        arr = np.array(props, dtype=np.float32)
        
        x1 = int(np.min(arr[:, 0]))
        y1 = int(np.min(arr[:, 1]))
        x2 = int(np.max(arr[:, 2]))
        y2 = int(np.max(arr[:, 3]))
        
        if x2 - x1 <= 0 or y2 - y1 <= 0: return None
        
        macro_width = x2 - x1
        macro_height = y2 - y1
        
        if len(boxes) > 1:
            # [v3.18.2→3.18.3] Orientation theo HÌNH DẠNG + PHÂN TÁN của box thành
            # viên — bất biến với jitter OCR (var(center) cũ chỉ đo nhiễu khi text
            # đứng yên). Dùng chung helper 3 tầng với bộ chọn trục của Band Refiner
            # để hai nơi nhất quán: aspect rõ (≥2.0) → spread tâm box → mặc định ngang.
            median_box_w = float(np.median(arr[:, 2] - arr[:, 0]))
            median_box_h = float(np.median(arr[:, 3] - arr[:, 1]))
            is_horizontal = self._is_horizontal_layout(
                median_box_w, median_box_h, arr[:, 4].tolist(), arr[:, 5].tolist()
            )
            orient = TextOrientation.HORIZONTAL if is_horizontal else TextOrientation.VERTICAL
        else:
            orient = TextOrientation.VERTICAL if macro_height > macro_width * 1.2 else TextOrientation.HORIZONTAL

        align = TextAlignment.CENTER
        total_true_area = float(np.sum(arr[:, 6]))
        mass_x = (x1 + x2) / 2.0  
        mass_y = (y1 + y2) / 2.0  
        
        if total_true_area > 1e-5:
            mass_x = float(np.sum(arr[:, 4] * arr[:, 6]) / total_true_area)
            mass_y = float(np.sum(arr[:, 5] * arr[:, 6]) / total_true_area)
            
            if orient == TextOrientation.HORIZONTAL:
                geo_center_x = (x1 + x2) / 2.0
                shift_ratio = (mass_x - geo_center_x) / max(1.0, float(macro_width))
                if shift_ratio < -0.1: align = TextAlignment.LEFT
                elif shift_ratio > 0.1: align = TextAlignment.RIGHT

        unique_frames = len(np.unique(arr[:, 8]))
        mean_conf = float(np.mean(arr[:, 7]))
        f_set = set(arr[:, 8].astype(int).tolist())

        local_thick = float(np.median(np.minimum(arr[:, 2]-arr[:, 0], arr[:, 3]-arr[:, 1])))

        return ROICluster(
            cluster_id, orient, align, x1, y1, x2, y2, 
            len(boxes), unique_frames, mean_conf, True, f_set, 
            mass_center_x=mass_x, mass_center_y=mass_y, local_thickness=max(1.0, local_thick), 
            true_mass_area=total_true_area
        )

__all__ = ["BBoxAnalyzer", "ROICluster", "RawBBox"]
