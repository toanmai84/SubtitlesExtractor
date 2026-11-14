"""[v3.23.644] Quyết định khung nào cần OCR — dùng chung cho mọi bộ lấy mẫu (PyAV, NVDEC).

Hai chế độ:

* **Cổ điển** (mặc định, giữ NGUYÊN hành vi cũ): so pHash + tỉ lệ điểm ảnh khác với khung đã giữ gần nhất.
* **Chữ trắng** (``white_text_gate``): chỉ nhìn các điểm ảnh SÁNG (chữ phụ đề trắng/vàng nhạt, độ chói
  > ``white_text_luma``). Đo thật trên tập 1 phim mẫu (4 421 khung, ROI 603×119): chế độ cổ điển gửi 71,8% khung đi
  OCR vì chuyển động nền làm đổi cả vùng ROI, trong khi 96,4% khung liền nhau có cùng chữ và chỉ ~158 lần đổi chữ.
  Với ngưỡng chói 200: 30,9% khung OCR, đủ 95/95 câu phụ đề; ngưỡng 220: 10,9% nhưng rơi 1 câu xám nhạt
  (xem ``tools/bench_ocr.py`` để đo lại trên phim của bạn).

[v3.23.697] Chế độ chữ trắng so "chữ đã đổi chưa" trên LÕI sáng ≥ 240 khi cả hai khung có lõi, và ở phim được nhận là
phụ đề trắng thì khung có chữ mà không có lõi (nền sáng) chỉ OCR đầu đoạn và mỗi 1 s. Phim phụ đề xám (không lõi) giữ
nguyên luật cũ. Đo tập 1–12 phim mẫu: khung đi OCR 56,8 % → 26,1 % (tập 1–5) và 61 % → 30 % (tập 6–12), phụ đề như cũ.

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
#: [v3.23.697] Lõi chữ trắng: điểm ảnh sáng ≥ mức này. Chữ phụ đề trắng luôn có lõi; nền sáng (trời, áo trắng, đèn) thường
#: chỉ 200–239. Đo tập 1–5 phim mẫu (7927 khung mẫu): 58 % khung đi OCR ở chế độ cũ đọc ra ĐÚNG chữ lần trước — nền sáng
#: chuyển động làm mặt nạ ngưỡng 200 đổi dù chữ đứng yên.
STRONG_TEXT_LUMA = 240
CORE_MIN_PIXELS = 8
#: Phim được coi là phụ đề TRẮNG khi, sau chừng này khung có chữ, ít nhất chừng ấy phần khung có lõi.
WHITE_WARMUP_FRAMES = 30
WHITE_CORE_RATIO = 0.5
#: Ở phim phụ đề trắng, khung có chữ mà KHÔNG có lõi (thường là nền sáng, hoặc một câu xám lẻ) chỉ OCR ở đầu đoạn và tối
#: đa mỗi chừng này giây; giữa chừng chép kết quả lần OCR đó — câu xám kéo dài vẫn được đọc.
NOCORE_REOCR_SEC = 1.0

#: [v3.23.710] Sau khi chữ ĐỔI, OCR lại một khung sau chừng này giây (chữ đã hiện rõ). Từ v3.23.697 mỗi câu gần như chỉ
#: được OCR ở khung ĐẦU TIÊN lúc chữ vừa hiện, mọi khung sau chép kết quả đó: khung đầu đọc hỏng (chữ đang hiện dần) thì
#: cả câu mất hoặc sai, và việc bỏ phiếu qua nhiều khung mất tác dụng. Lượt chạy thật phim 86 tập: bản 3.23.690 sót 2 câu
#: một chữ, bản 3.23.708 sót 7 ("是", "你", "哎", "好"… — các khung trong câu đều rỗng); tái hiện trên CPU ở tập 74 ("好").
CONFIRM_DELAY_SEC = 0.08
#: [v3.23.710] Khi chữ còn hiện, OCR lại định kỳ để có vài lần đọc độc lập cho mỗi câu (bỏ phiếu, khôi phục chữ bị rơi).
TEXT_REOCR_SEC = 0.40

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
        # [v3.23.697] lõi chữ trắng (chỉ khi ngưỡng người dùng còn thấp hơn mức lõi)
        self._use_core = self._luma < STRONG_TEXT_LUMA - 1
        self._last_core: np.ndarray | None = None
        self._last_core_px = 0
        self._pending_core: np.ndarray | None = None
        self._pending_core_px = 0
        self._text_frames = 0
        self._core_frames = 0
        self._nocore_ocr_ts: float | None = None
        self._nocore_pending = False
        # [v3.23.710] khung xác nhận sau khi chữ đổi
        self._confirm_due: float | None = None
        self._pending_reason = ""
        # [v3.23.711] khung vừa phán có cùng điểm ảnh chữ với khung OCR trước
        self._continues = False

    # ── API ────────────────────────────────────────────────────────────────
    def verdict(self, image_rgb: np.ndarray, timestamp_sec: float) -> FrameVerdict:
        """Khung này cần OCR, là bản sao của khung trước, hay trống?"""
        self._continues = False
        if self._gate:
            result = self._gate_verdict(image_rgb, timestamp_sec)
        else:
            result = self._classic_verdict(image_rgb)
        if result is FrameVerdict.DUPLICATE:
            self._continues = True
        elif result is FrameVerdict.BLANK:
            self._continues = False
        return result

    @property
    def continues_text(self) -> bool:
        """[v3.23.711] Khung vừa phán có CÙNG điểm ảnh chữ với khung đã OCR gần nhất (khung trùng, hoặc khung được đưa
        đi OCR lại theo lịch xác nhận / định kỳ mà chữ không đổi). Đọc ngay sau :meth:`verdict`."""
        return self._continues

    def accept(self, image_rgb: np.ndarray, timestamp_sec: float) -> None:
        """Ghi nhớ khung vừa được gửi đi OCR làm mốc so sánh cho các khung sau."""
        if self._gate:
            if self._nocore_pending:             # [v3.23.697] khung không lõi: không làm mốc so sánh
                self._nocore_pending = False
                self._last_ocr_ts = timestamp_sec
                return
            if self._pending_mask is not None:
                self._last_mask = self._pending_mask
                self._last_text_px = self._pending_text_px
                self._last_core, self._last_core_px = self._pending_core, self._pending_core_px
            else:                                # chưa gọi verdict (an toàn): tự tính
                self._last_mask, self._last_text_px = self._text_mask(image_rgb)
                self._last_core, self._last_core_px = None, 0
            self._last_ocr_ts = timestamp_sec
            self._pending_mask = None
            self._pending_core = None
            # [v3.23.710] Chữ vừa đổi → hẹn một khung xác nhận; khung xác nhận / định kỳ thì không hẹn tiếp.
            self._confirm_due = timestamp_sec + CONFIRM_DELAY_SEC if self._pending_reason == "change" else None
            self._pending_reason = ""
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
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
        _, mask = cv2.threshold(gray, self._luma, 255, cv2.THRESH_BINARY)
        text_px = int(cv2.countNonZero(mask))
        area = int(mask.size)
        if text_px < max(BLANK_MIN_PIXELS, int(BLANK_FRACTION * area)):
            self._last_mask = None            # lần có chữ kế tiếp luôn được OCR
            self._pending_mask = None
            self._nocore_ocr_ts = None
            self._confirm_due = None
            return FrameVerdict.BLANK
        core, core_px = None, 0
        if self._use_core:
            _, core = cv2.threshold(gray, STRONG_TEXT_LUMA - 1, 255, cv2.THRESH_BINARY)
            core_px = int(cv2.countNonZero(core))
            self._text_frames += 1
            self._core_frames += core_px >= CORE_MIN_PIXELS
            if core_px < CORE_MIN_PIXELS and self._white_subtitles():
                # [v3.23.697] Phim phụ đề trắng, khung không có lõi: OCR ở đầu đoạn và tối đa mỗi NOCORE_REOCR_SEC.
                if self._nocore_ocr_ts is not None and timestamp_sec - self._nocore_ocr_ts < NOCORE_REOCR_SEC:
                    return FrameVerdict.DUPLICATE
                self._nocore_ocr_ts = timestamp_sec
                self._last_mask = None        # khung có lõi kế tiếp luôn được OCR
                self._pending_mask = None
                self._nocore_pending = True
                return FrameVerdict.OCR
        self._nocore_ocr_ts = None
        self._nocore_pending = False
        self._pending_mask, self._pending_text_px = mask, text_px
        self._pending_core, self._pending_core_px = core, core_px
        last = self._last_mask
        if last is None or last.shape != mask.shape:
            self._pending_reason = "change"
            return FrameVerdict.OCR
        # [v3.23.711] Xét "chữ có đổi không" TRƯỚC lịch đọc lại: khung xác nhận / định kỳ mà điểm ảnh chữ không đổi là
        # một lần đọc nữa của CÙNG câu (``continues_text``) — khâu dựng gộp các lần đọc đó thay vì coi là câu khác.
        if self._text_changed(mask, text_px, core, core_px, area):
            self._pending_reason = "change"
            return FrameVerdict.OCR
        self._continues = True
        if timestamp_sec - self._last_ocr_ts >= MAX_GAP_SEC:
            self._pending_reason = "periodic"
            return FrameVerdict.OCR
        # [v3.23.710] Khung xác nhận sau khi chữ đổi, và đọc lại định kỳ khi chữ (có lõi sáng) còn hiện.
        if self._confirm_due is not None and timestamp_sec >= self._confirm_due - 1e-6:
            self._pending_reason = "confirm"
            return FrameVerdict.OCR
        if core_px >= CORE_MIN_PIXELS and timestamp_sec - self._last_ocr_ts >= TEXT_REOCR_SEC:
            self._pending_reason = "periodic"
            return FrameVerdict.OCR
        return FrameVerdict.DUPLICATE

    def _text_changed(
        self, mask: np.ndarray, text_px: int, core: np.ndarray | None, core_px: int, area: int,
    ) -> bool:
        """Điểm ảnh chữ của khung này khác khung đã OCR gần nhất (đã nở 3×3 chống rung 1 px)."""
        last = self._last_mask
        assert last is not None
        # [v3.23.697] Cả hai khung có lõi chữ trắng → so trên lõi (nền sáng 200–239 chuyển động không còn làm "đổi chữ").
        if (
            core is not None and self._last_core is not None and core_px >= CORE_MIN_PIXELS
            and self._last_core_px >= CORE_MIN_PIXELS and self._last_core.shape == core.shape
        ):
            mask, last, text_px, last_px = core, self._last_core, core_px, self._last_core_px
        else:
            last_px = self._last_text_px
        grown_last = cv2.dilate(last, _KERNEL)
        grown_now = cv2.dilate(mask, _KERNEL)
        appeared = int(cv2.countNonZero(cv2.bitwise_and(mask, cv2.bitwise_not(grown_last))))
        vanished = int(cv2.countNonZero(cv2.bitwise_and(last, cv2.bitwise_not(grown_now))))
        threshold = max(
            CHANGE_FRACTION_OF_AREA * area,
            CHANGE_FRACTION_OF_TEXT * max(text_px, last_px),
        )
        return appeared + vanished >= threshold

    def _white_subtitles(self) -> bool:
        """Phim này phụ đề trắng: đã thấy đủ khung có chữ và đa số có lõi sáng ≥ ``STRONG_TEXT_LUMA``."""
        return self._text_frames >= WHITE_WARMUP_FRAMES and self._core_frames >= WHITE_CORE_RATIO * self._text_frames
