"""[v3.23.680] Ghi dải ranh giới trong lúc giải mã: mỗi khung KHÔNG OCR → (thời điểm, số ô chữ sáng, giống mẫu trước,
giống mẫu sau). Dùng chung cho các bộ lấy mẫu (PyAV, NVDEC…). Xem ``domain/value_objects/boundary_track``.

Mặt nạ chữ: độ sáng (Y) vùng phụ đề → lấy max từng ô 4×4 → ≥ 240. Đo trên tập 1–12 phim mẫu (chỉnh 190–240): 240 cho
ít lỗi nhất (chữ trắng phụ đề sáng hơn hẳn nền); phụ đề xám (ít ô sáng) thì không dời mốc — an toàn.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from subtitles_extractor.domain.value_objects.boundary_track import (
    BoundaryProbe,
    BoundaryTrack,
)
from subtitles_extractor.domain.value_objects.roi import Roi

logger = logging.getLogger(__name__)

#: Cạnh ô thu nhỏ (điểm ảnh).
_CELL = 4
#: Ngưỡng độ sáng coi là chữ.
_TEXT_LUMA = 240
#: Giữ tối đa chừng này khung bỏ qua giữa hai mẫu (bước 0,5 s ở 60 fps = 30 khung).
_MAX_GAP_FRAMES = 64
#: Bước lấy mẫu từ mức này trở lên (chế độ khung khoá) thì không ghi dải ranh giới.
PROBE_MAX_STEP_SEC = 5.0


def probing_enabled(roi: Roi | None, step_sec: float) -> bool:
    """Chỉ ghi khi có ROI (mặt nạ đo trên vùng phụ đề) và bước không thưa."""
    return roi is not None and step_sec < PROBE_MAX_STEP_SEC


def expand_studio_luma(y: Any) -> Any:
    """Độ sáng dải hẹp (16–235, mặt phẳng Y gốc) → dải đầy (0–255) như ``frame.to_ndarray(format="gray")`` của PyAV.

    Khớp từng điểm ảnh với PyAV (đo trên tập 1). Dùng được cho cả mảng CuPy (chỉ dùng phép toán chung).
    """
    v = (y.astype("int32") - 16) * 510 + 219
    return (v // 438).clip(0, 255).astype("uint8")


def text_mask(luma: np.ndarray) -> np.ndarray:
    """Mặt nạ chữ sáng thu nhỏ của một vùng độ sáng 2 chiều (uint8)."""
    h, w = luma.shape[:2]
    hh, ww = (h // _CELL) * _CELL, (w // _CELL) * _CELL
    if hh == 0 or ww == 0:
        return np.zeros((0, 0), dtype=bool)
    pooled = luma[:hh, :ww].reshape(hh // _CELL, _CELL, ww // _CELL, _CELL).max(axis=(1, 3))
    return pooled >= _TEXT_LUMA


def mask_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Giao/hợp hai mặt nạ; cả hai trống → 1.0 (cùng "không chữ")."""
    if a.shape != b.shape:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return 1.0 if union == 0 else int(np.count_nonzero(a & b)) / union


def probe_av_frame(record: Callable[[np.ndarray, float], None], frame: Any, timestamp_sec: float) -> None:
    """Ghi một khung PyAV vào bộ dò từ mặt phẳng độ sáng (rẻ, không đổi sang RGB)."""
    try:
        record(frame.to_ndarray(format="gray"), timestamp_sec)
    except (ValueError, AttributeError) as exc:   # định dạng điểm ảnh lạ: bỏ đầu dò khung này, không ảnh hưởng OCR
        logger.debug("Bỏ đầu dò ranh giới khung %.3f: %s", timestamp_sec, exc)


class BoundaryProber:
    """Gom dải ranh giới theo thứ tự giải mã. Gọi :meth:`skipped` cho khung bỏ qua, :meth:`sampled` cho khung mẫu."""

    def __init__(self, roi: Roi | None) -> None:
        self._roi = roi
        self._gap: list[tuple[float, np.ndarray]] = []
        self._prev_mask: np.ndarray | None = None
        self._sample_times: list[float] = []
        self._sample_pixels: list[int] = []
        self._probes: list[BoundaryProbe] = []
        self._frame_times: list[float] = []

    def _crop(self, luma: np.ndarray) -> np.ndarray:
        if self._roi is None:
            return luma
        h, w = luma.shape[:2]
        r = self._roi.clip_to(w, h)
        return luma[r.y:r.y2, r.x:r.x2]

    def skipped(self, luma_frame: np.ndarray, timestamp_sec: float) -> None:
        self._frame_times.append(timestamp_sec)
        self._gap.append((timestamp_sec, text_mask(self._crop(luma_frame))))
        if len(self._gap) > _MAX_GAP_FRAMES:
            self._gap.pop(0)

    def sampled(self, luma_frame: np.ndarray, timestamp_sec: float) -> None:
        self._frame_times.append(timestamp_sec)
        mask = text_mask(self._crop(luma_frame))
        prev = self._prev_mask
        for t, gap_mask in self._gap:
            self._probes.append(BoundaryProbe(
                t, int(np.count_nonzero(gap_mask)),
                mask_similarity(gap_mask, prev) if prev is not None else 0.0,
                mask_similarity(gap_mask, mask),
            ))
        self._gap.clear()
        self._prev_mask = mask
        self._sample_times.append(timestamp_sec)
        self._sample_pixels.append(int(np.count_nonzero(mask)))

    def finish(self) -> BoundaryTrack:
        """Khép dải (khung bỏ qua sau mẫu cuối: chỉ so với mẫu trước)."""
        prev = self._prev_mask
        for t, gap_mask in self._gap:
            self._probes.append(BoundaryProbe(
                t, int(np.count_nonzero(gap_mask)),
                mask_similarity(gap_mask, prev) if prev is not None else 0.0, 0.0,
            ))
        self._gap.clear()
        times = sorted(self._frame_times)
        diffs = np.diff(np.asarray(times)) if len(times) > 1 else np.asarray([])
        diffs = diffs[diffs > 1e-6]
        period = float(np.median(diffs)) if diffs.size else 0.0
        return BoundaryTrack(tuple(self._sample_times), tuple(self._sample_pixels), tuple(self._probes), period)


__all__ = [
    "PROBE_MAX_STEP_SEC", "BoundaryProber", "expand_studio_luma", "mask_similarity", "probe_av_frame", "probing_enabled", "text_mask",
]
