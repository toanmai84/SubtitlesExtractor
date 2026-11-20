"""[v3.23.727] Lọc chữ sáng ở mức HỘP CHỮ: hộp OCR không chứa điểm ảnh sáng nào thì không phải phụ đề.

Bộ lọc "chữ sáng" (``white_text_gate``) trước đây chỉ xét CẢ KHUNG (khử trùng, nhận khung trống). Khung có vật sáng ở
chỗ khác (sương, ánh đèn) vẫn được OCR, và OCR có thể đọc nhầm một vật màu thành chữ. Lượt chạy thật 3.23.726 (phim nối
86 tập, model medium): bông hoa lục bình hồng tím ở tập 72 giây 114,6 bị đọc thành "福" (tin cậy 0,73) và lên SRT.

Đo trên 10 tập (24, 28, 37, 70–76), tỉ lệ điểm ảnh có độ chói ≥ 200 bên trong hộp chữ, trên đúng ảnh OCR thấy (đã phóng,
làm nét, tăng tương phản):

* 483 hộp thuộc phụ đề thật: thấp nhất 0,134, 1 % thấp nhất 0,238, trung vị 0,548 (chữ trắng viền đen);
* hộp "福" (bông hoa): 0,0005.

Phụ đề hardsub là chữ sáng (trắng / vàng nhạt) nên hộp chữ thật luôn có điểm sáng; ngưỡng :data:`MIN_BRIGHT_RATIO` thấp
hơn hộp thật tối nhất 4,5 lần. Chỉ bỏ hộp khi CẢ tối CẢ tin cậy dưới :data:`MAX_CONFIDENCE_TO_DROP`: khung đầu của câu
hiện dần (chữ còn mờ) mà OCR đọc chắc tay thì vẫn giữ.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult

__all__ = ["MAX_CONFIDENCE_TO_DROP", "MIN_BRIGHT_RATIO", "NO_BRIGHT_RATIO", "bright_ratio", "drop_dark_boxes"]

#: Tỉ lệ điểm sáng tối thiểu của một hộp phụ đề thật (đo thật: hộp thật thấp nhất 0,134).
MIN_BRIGHT_RATIO: float = 0.03
#: Hộp tối mà OCR đọc chắc tay từ mức này thì vẫn giữ (khung đầu của câu hiện dần).
MAX_CONFIDENCE_TO_DROP: float = 0.90
#: [v3.23.730] Dưới mức này (gần như không có điểm sáng) thì bỏ hộp BẤT KỂ độ tin cậy. Phim "大乾悍刀行" (84 tập, 3.23.728):
#: chữ thư pháp mực đen trên cuộn giấy / thánh chỉ (94:03–94:57, 20:22) đọc rõ, tin cậy 0,90–1,00, lên 26 câu rác
#: ("皇上王爷", "火海", "可不"…) — tỉ lệ sáng trên ảnh đã tiền xử lý đúng 0,0000. Câu thật tối nhất (chữ "一" nét mảnh)
#: vẫn 0,033; mô phỏng trên cả phim (1322 hộp bị bỏ) chỉ bỏ đúng 26 câu rác, không mất câu thật nào.
NO_BRIGHT_RATIO: float = 0.005


def _luma_mask(image_rgb: np.ndarray, luma: int) -> np.ndarray:
    pixels = image_rgb.astype(np.float32)
    if pixels.ndim == 2:
        return np.asarray(pixels >= luma, dtype=bool)
    return np.asarray((0.299 * pixels[..., 0] + 0.587 * pixels[..., 1] + 0.114 * pixels[..., 2]) >= luma, dtype=bool)


def bright_ratio(
    bright: np.ndarray, polygon: Sequence[Sequence[float]], border_px: int = 0,
) -> float | None:
    """Tỉ lệ điểm sáng trong hình chữ nhật bao ``polygon`` (toạ độ của ảnh ``bright``), bỏ viền ``border_px`` quanh
    ảnh (viền trắng do tiền xử lý thêm vào). ``None`` nếu hộp rỗng / nằm ngoài ảnh."""
    if not polygon:
        return None
    height, width = bright.shape[:2]
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    x0 = max(border_px, int(np.floor(min(xs))))
    x1 = min(width - border_px, int(np.ceil(max(xs))))
    y0 = max(border_px, int(np.floor(min(ys))))
    y1 = min(height - border_px, int(np.ceil(max(ys))))
    if x1 <= x0 or y1 <= y0:
        return None
    return float(bright[y0:y1, x0:x1].mean())


def drop_dark_boxes(
    result: OcrFrameResult, image_rgb: np.ndarray | None, luma: int, border_px: int = 0,
) -> tuple[OcrFrameResult, int]:
    """Bỏ các hộp chữ TỐI (tỉ lệ sáng < :data:`MIN_BRIGHT_RATIO`) VÀ tin cậy thấp (< :data:`MAX_CONFIDENCE_TO_DROP`), và
    [v3.23.730] mọi hộp gần như không có điểm sáng (< :data:`NO_BRIGHT_RATIO`) bất kể tin cậy.

    ``image_rgb`` là đúng ảnh OCR đã đọc, toạ độ hộp cùng hệ với ảnh đó. Trả ``(kết quả, số hộp bị bỏ)``.
    """
    if image_rgb is None or getattr(image_rgb, "size", 0) == 0 or not result.text_boxes:
        return result, 0
    candidates = [box for box in result.text_boxes if box.polygon]
    if not candidates:
        return result, 0
    bright = _luma_mask(image_rgb, int(luma))
    dropped = set()
    for box in candidates:
        ratio = bright_ratio(bright, box.polygon, border_px)
        if ratio is None:
            continue
        # [v3.23.730] Hộp KHÔNG có điểm sáng nào là chữ tối trong cảnh dù OCR đọc chắc tay (chữ thư pháp trên cuộn giấy).
        if ratio < NO_BRIGHT_RATIO or (ratio < MIN_BRIGHT_RATIO and float(box.confidence) < MAX_CONFIDENCE_TO_DROP):
            dropped.add(id(box))
    if not dropped:
        return result, 0
    kept = [box for box in result.text_boxes if id(box) not in dropped]
    return dataclasses.replace(result, text_boxes=kept), len(dropped)
