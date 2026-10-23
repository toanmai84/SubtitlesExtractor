"""[v3.23.644] Quyết định khung nào cần OCR — dùng chung cho mọi bộ lấy mẫu (PyAV, NVDEC).

Hai chế độ:

* **Cổ điển** (mặc định, giữ NGUYÊN hành vi cũ): so pHash + tỉ lệ điểm ảnh khác với khung đã giữ gần nhất.
* **Chữ trắng** (``white_text_gate``): chỉ nhìn các điểm ảnh SÁNG (chữ phụ đề trắng/vàng nhạt, độ chói
  > ``white_text_luma``). Đo thật trên tập 1 phim mẫu (4 421 khung, ROI 603×119): chế độ cổ điển gửi 71,8% khung đi
  OCR vì chuyển động nền làm đổi cả vùng ROI, trong khi 96,4% khung liền nhau có cùng chữ và chỉ ~158 lần đổi chữ.
  Với ngưỡng chói 200: 30,9% khung OCR, đủ 95/95 câu phụ đề; ngưỡng 220: 10,9% nhưng rơi 1 câu xám nhạt
  (xem ``tools/bench_ocr.py`` để đo lại trên phim của bạn).

Hàm thuần theo trạng thái nội bộ, không I/O — kiểm thử được không cần video.
"""

from __future__ import annotations

import enum

import cv2
import numpy as np

from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig
from subtitles_extractor.infrastructure.video.perceptual_hash import (
    compute_phash,
    hamming_distance,
    pixel_diff_ratio,
)

__all__ = ["FrameDeduper", "FrameVerdict"]

#: Khung có ít hơn chừng này (tỉ lệ diện tích ROI) điểm ảnh sáng thì coi là KHÔNG có chữ.
BLANK_FRACTION = 0.0007
BLANK_MIN_PIXELS = 8
#: Chữ đổi khi số điểm ảnh sáng mới/mất đi (sau khi nở 3×3 chống rung 1 px) vượt cả hai mức.
CHANGE_FRACTION_OF_AREA = 0.0004
CHANGE_FRACTION_OF_TEXT = 0.03
#: Quá chừng này giây không OCR thì ép OCR lại (chống trôi từ thay đổi quá nhỏ).
MAX_GAP_SEC = 1.5

_KERNEL = np.ones((3, 3), np.uint8)


class FrameVerdict(enum.Enum):
    """Kết luận cho một khung đã lấy mẫu."""

    OCR = "ocr"
    DUPLICATE = "duplicate"     # giống khung đã OCR gần nhất → sao chép kết quả
    BLANK = "blank"             # không có chữ → kết quả rỗng


class FrameDeduper:
    """Giữ trạng thái "khung đã OCR gần nhất" và phán từng khung mới.

    Cách dùng: ``verdict = deduper.verdict(image, ts)``; nếu ``OCR`` thì sau khi phát khung đi gọi
    ``deduper.accept(image, ts)`` để ghi nhớ làm mốc so sánh.
    """

    def __init__(self, config: FrameSamplingConfig) -> None:
        self._config = config
        self._gate = bool(getattr(config, "white_text_gate", False))
        self._luma = int(min(254, max(1, getattr(config, "white_text_luma", 200))))
        # cổ điển
        self._last_hash: int | None = None
        self._last_image: np.ndarray | None = None
        self._pending_hash: int | None = None
        # chữ trắng
        self._last_mask: np.ndarray | None = None
        self._last_text_px = 0
        self._last_ocr_ts = 0.0
        self._pending_mask: np.ndarray | None = None
        self._pending_text_px = 0

    # ── API ────────────────────────────────────────────────────────────────
    def verdict(self, image_rgb: np.ndarray, timestamp_sec: float) -> FrameVerdict:
        """Khung này cần OCR, là bản sao của khung trước, hay trống?"""
        if self._gate:
            return self._gate_verdict(image_rgb, timestamp_sec)
        return self._classic_verdict(image_rgb)

    def accept(self, image_rgb: np.ndarray, timestamp_sec: float) -> None:
        """Ghi nhớ khung vừa được gửi đi OCR làm mốc so sánh cho các khung sau."""
        if self._gate:
            if self._pending_mask is not None:
                self._last_mask = self._pending_mask
                self._last_text_px = self._pending_text_px
            else:                                # chưa gọi verdict (an toàn): tự tính
                self._last_mask, self._last_text_px = self._text_mask(image_rgb)
            self._last_ocr_ts = timestamp_sec
            self._pending_mask = None
            return
        self._last_hash = (
            self._pending_hash if self._pending_hash is not None else compute_phash(image_rgb)
        )
        self._last_image = image_rgb
        self._pending_hash = None

    # ── Cổ điển (y hệt hành vi cũ của cả hai bộ lấy mẫu) ──────────────────
    def _classic_verdict(self, image_rgb: np.ndarray) -> FrameVerdict:
        if self._last_hash is None or self._last_image is None:
            self._pending_hash = None
            return FrameVerdict.OCR
        current_hash = compute_phash(image_rgb)
        self._pending_hash = current_hash                       # tính MỘT lần, dùng lại ở accept()
        config = self._config
        if hamming_distance(current_hash, self._last_hash) > config.phash_distance_threshold:
            return FrameVerdict.OCR
        if pixel_diff_ratio(image_rgb, self._last_image) > config.pixel_diff_threshold:
            return FrameVerdict.OCR
        return FrameVerdict.DUPLICATE

    # ── Chữ trắng ──────────────────────────────────────────────────────────
    def _text_mask(self, image_rgb: np.ndarray) -> tuple[np.ndarray, int]:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
        _, mask = cv2.threshold(gray, self._luma, 255, cv2.THRESH_BINARY)
        return mask, int(cv2.countNonZero(mask))

    def _gate_verdict(self, image_rgb: np.ndarray, timestamp_sec: float) -> FrameVerdict:
        mask, text_px = self._text_mask(image_rgb)
        area = int(mask.size)
        if text_px < max(BLANK_MIN_PIXELS, int(BLANK_FRACTION * area)):
            self._last_mask = None            # lần có chữ kế tiếp luôn được OCR
            self._pending_mask = None
            return FrameVerdict.BLANK
        self._pending_mask, self._pending_text_px = mask, text_px
        last = self._last_mask
        if last is None or last.shape != mask.shape:
            return FrameVerdict.OCR
        if timestamp_sec - self._last_ocr_ts >= MAX_GAP_SEC:
            return FrameVerdict.OCR
        grown_last = cv2.dilate(last, _KERNEL)
        grown_now = cv2.dilate(mask, _KERNEL)
        appeared = int(cv2.countNonZero(cv2.bitwise_and(mask, cv2.bitwise_not(grown_last))))
        vanished = int(cv2.countNonZero(cv2.bitwise_and(last, cv2.bitwise_not(grown_now))))
        threshold = max(
            CHANGE_FRACTION_OF_AREA * area,
            CHANGE_FRACTION_OF_TEXT * max(text_px, self._last_text_px),
        )
        if appeared + vanished >= threshold:
            return FrameVerdict.OCR
        return FrameVerdict.DUPLICATE
