"""Đánh giá chất lượng phát hiện dải ROI phụ đề từ dữ liệu OCR thô.

``BBoxAnalyzer`` thuần CPU trên toạ độ box, nên việc hiệu chuẩn ROI có thể chạy
*offline* từ ``*_seraw.json`` mà không cần video hay GPU. Hai chế độ chấm điểm:

* **Có nhãn** (``labeled_band_ratio``): tính IoU 1-D trên trục Y giữa dải phát
  hiện chính và dải phụ đề thật do người dùng cung cấp — chính xác nhất.
* **Không nhãn** (proxy): dải tốt = bao phủ nhiều box tin cậy cao *và* đủ gọn
  (phụ đề hardsub là một dải hẹp), điểm = ``coverage × compactness``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger

BBoxFactory = Callable[..., object]
AnalyzerFactory = Callable[[int, int, dict[str, float | int | bool]], object]


def derive_temporal_density_band(
    boxes: list[tuple[float, float, float, float, float, int, float]],
    frame_height: int,
    frame_width: int,
    line_width_ratio: float = 0.12,
    keep_ratio: float = 0.5,
    smoothing_ratio: float = 0.01,
) -> "RoiBand | None":
    """Suy ra dải phụ đề *ground-truth độc lập* theo mật độ thời gian.

    Với mỗi hàng Y, đếm số frame riêng biệt có box "dạng dòng" (rộng) phủ hàng đó;
    dải phụ đề là đỉnh trội nhất. Đây là tham chiếu khách quan để chấm IoU mà không
    cần người gán nhãn — dùng khi không có ``labeled_band``.

    Args:
        boxes: Danh sách ``(x_min, y_min, x_max, y_max, conf, frame_idx, ts)``.
        frame_height: Chiều cao khung (px).
        frame_width: Chiều rộng khung (px).
        line_width_ratio: Box hẹp hơn tỷ lệ này so với bề rộng khung bị bỏ (icon/logo).
        keep_ratio: Giữ đoạn liên tục quanh đỉnh có mật độ ≥ ``keep_ratio × peak``.
        smoothing_ratio: Bán kính làm mượt theo tỷ lệ chiều cao khung.

    Returns:
        :class:`RoiBand` của dải trội, hoặc ``None`` nếu không đủ dữ liệu.
    """
    if not boxes or frame_height <= 0:
        return None
    min_line_width = line_width_ratio * max(1, frame_width)
    rows: list[set[int]] = [set() for _ in range(frame_height)]
    used = False
    for x_min, y_min, x_max, y_max, _conf, frame_idx, _ts in boxes:
        if (x_max - x_min) < min_line_width:
            continue
        used = True
        top = max(0, int(y_min))
        bottom = min(frame_height - 1, int(y_max))
        for row in range(top, bottom + 1):
            rows[row].add(frame_idx)
    if not used:
        return None
    profile = np.fromiter((len(frame_set) for frame_set in rows), dtype=np.float64, count=frame_height)
    if profile.max() <= 0:
        return None
    radius = max(1, int(smoothing_ratio * frame_height))
    kernel = np.ones(2 * radius + 1, dtype=np.float64) / (2 * radius + 1)
    smoothed = np.convolve(profile, kernel, mode="same")
    peak = int(smoothed.argmax())
    threshold = keep_ratio * float(smoothed[peak])
    top = peak
    while top > 0 and smoothed[top - 1] >= threshold:
        top -= 1
    bottom = peak
    while bottom < frame_height - 1 and smoothed[bottom + 1] >= threshold:
        bottom += 1
    return RoiBand(top_ratio=top / frame_height, bottom_ratio=bottom / frame_height)


@dataclass(frozen=True, slots=True)
class RoiBand:
    """Một dải ROI theo trục Y (đơn vị: tỷ lệ so với chiều cao khung).

    Attributes:
        top_ratio: Mép trên (∈ [0, 1]).
        bottom_ratio: Mép dưới (∈ [0, 1]).
    """

    top_ratio: float
    bottom_ratio: float

    @property
    def height_ratio(self) -> float:
        return max(0.0, self.bottom_ratio - self.top_ratio)

    def iou_1d(self, other: "RoiBand") -> float:
        """IoU 1-D trên trục Y với dải khác."""
        intersection = max(
            0.0, min(self.bottom_ratio, other.bottom_ratio)
            - max(self.top_ratio, other.top_ratio)
        )
        union = (
            max(self.bottom_ratio, other.bottom_ratio)
            - min(self.top_ratio, other.top_ratio)
        )
        return intersection / union if union > 0 else 0.0


@dataclass(slots=True)
class _PreparedRoiData:
    """Box thô + kích thước khung đã suy ra cho một seraw."""

    label: str
    boxes: list[tuple[float, float, float, float, float, int, float]]
    frame_width: int
    frame_height: int
    labeled_band: RoiBand | None
    auto_gt_band: RoiBand | None = None


def _extract_boxes(
    seraw_path: Path, confidence_floor: float
) -> tuple[list[tuple[float, float, float, float, float, int, float]], int, int]:
    """Trích (x_min,y_min,x_max,y_max,conf,frame_idx,ts) + suy kích thước khung."""
    with seraw_path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    meta = document.get("meta", {})
    boxes: list[tuple[float, float, float, float, float, int, float]] = []
    max_x = float(meta.get("frame_width", 0) or 0)
    max_y = float(meta.get("frame_height", 0) or 0)
    for frame in document.get("frames", []):
        frame_idx = int(frame.get("fi", 0))
        timestamp = float(frame.get("ts", 0.0))
        for box in frame.get("boxes", []):
            confidence = float(box.get("c", 0.0))
            polygon = box.get("p", [])
            if confidence < confidence_floor or len(polygon) < 4:
                continue
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            boxes.append((x_min, y_min, x_max, y_max, confidence, frame_idx, timestamp))
            max_x = max(max_x, x_max)
            max_y = max(max_y, y_max)
    frame_width = int(meta.get("frame_width", 0) or max_x) or 1
    frame_height = int(meta.get("frame_height", 0) or max_y) or 1
    return boxes, frame_width, frame_height


class RoiCalibrationEvaluator:
    """Hàm mục tiêu hiệu chuẩn phát hiện dải ROI phụ đề.

    Args:
        seraw_paths: Danh sách ``(label, path)`` dữ liệu OCR thô.
        bbox_factory: Hàm dựng ``RawBBox(coord_x_min=..., ...)``.
        analyzer_factory: Hàm dựng analyzer ``(frame_w, frame_h, params) -> analyzer``
            có ``.analyze(raw_bboxes) -> clusters`` với mỗi cluster lộ
            ``coord_y_min``, ``coord_y_max``, ``frame_count``.
        labeled_bands: Tùy chọn nhãn ``label -> RoiBand`` (dải phụ đề thật).
        confidence_floor: Ngưỡng lọc box đầu vào.
        compactness_weight: Trọng số phạt dải quá cao trong chế độ proxy.
    """

    def __init__(
        self,
        *,
        seraw_paths: list[tuple[str, Path]],
        bbox_factory: BBoxFactory,
        analyzer_factory: AnalyzerFactory,
        labeled_bands: dict[str, RoiBand] | None = None,
        confidence_floor: float = 0.55,
        compactness_weight: float = 0.5,
        auto_derive_gt_band: bool = True,
    ) -> None:
        self._bbox_factory = bbox_factory
        self._analyzer_factory = analyzer_factory
        self._compactness_weight = compactness_weight
        labeled_bands = labeled_bands or {}
        self._prepared: list[_PreparedRoiData] = []
        for label, path in seraw_paths:
            boxes, frame_width, frame_height = _extract_boxes(path, confidence_floor)
            has_label = label in labeled_bands
            auto_band = (
                derive_temporal_density_band(boxes, frame_height, frame_width)
                if auto_derive_gt_band and not has_label
                else None
            )
            self._prepared.append(
                _PreparedRoiData(
                    label=label,
                    boxes=boxes,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    labeled_band=labeled_bands.get(label),
                    auto_gt_band=auto_band,
                )
            )
            band_note = (
                "  [nhãn band]" if has_label
                else ("  [GT tự suy ra]" if auto_band else "  [proxy]")
            )
            logger.info(
                "Nạp ROI '{}': {} box (conf≥{}), khung≈{}x{}{}",
                label, len(boxes), confidence_floor, frame_width, frame_height,
                band_note,
            )

    def _detect_primary_band(
        self, prepared: _PreparedRoiData, params: dict[str, float | int | bool]
    ) -> RoiBand | None:
        """Chạy analyzer, trả dải có frame_count cao nhất (dòng phụ đề chính)."""
        raw_bboxes = [
            self._bbox_factory(
                coord_x_min=x_min, coord_y_min=y_min,
                coord_x_max=x_max, coord_y_max=y_max,
                confidence=conf, frame_idx=frame_idx, timestamp_sec=ts,
            )
            for (x_min, y_min, x_max, y_max, conf, frame_idx, ts) in prepared.boxes
        ]
        if not raw_bboxes:
            return None
        analyzer = self._analyzer_factory(
            prepared.frame_width, prepared.frame_height, params
        )
        clusters = analyzer.analyze(raw_bboxes)
        if not clusters:
            return None
        primary = max(clusters, key=lambda cluster: getattr(cluster, "frame_count", 0))
        height = max(1, prepared.frame_height)
        return RoiBand(
            top_ratio=float(primary.coord_y_min) / height,
            bottom_ratio=float(primary.coord_y_max) / height,
        )

    def _score_one(
        self, prepared: _PreparedRoiData, params: dict[str, float | int | bool]
    ) -> float:
        band = self._detect_primary_band(prepared, params)
        if band is None:
            return 0.0
        reference_band = prepared.labeled_band or prepared.auto_gt_band
        if reference_band is not None:
            return band.iou_1d(reference_band)
        # Proxy không nhãn: coverage (box nằm trong dải) × compactness (dải hẹp).
        height = max(1, prepared.frame_height)
        inside = sum(
            1
            for (_x0, y_min, _x1, y_max, *_rest) in prepared.boxes
            if band.top_ratio <= ((y_min + y_max) / 2.0) / height <= band.bottom_ratio
        )
        coverage = inside / len(prepared.boxes) if prepared.boxes else 0.0
        compactness = max(0.0, 1.0 - band.height_ratio)
        return coverage * (
            (1.0 - self._compactness_weight) + self._compactness_weight * compactness
        )

    def objective(self, assignment: dict[str, float]) -> float:
        """Điểm trung bình trên toàn bộ seraw (càng cao càng tốt)."""
        if not self._prepared:
            return 0.0
        scores = [self._score_one(prepared, assignment) for prepared in self._prepared]
        return sum(scores) / len(scores)
