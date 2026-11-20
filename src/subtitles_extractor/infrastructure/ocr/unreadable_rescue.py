"""[v3.23.732] Cứu chữ mà bộ nhận dạng chính KHÔNG CÓ trong bảng chữ (hàm thuần, không cần mô hình).

Chữ cảm thán "诶" (ēi) rất hay gặp trong phim ngắn nhưng không có trong bảng chữ của mọi mô hình nhận dạng PaddleOCR đang
dùng (PP-OCRv6 small / medium, PP-OCRv5 mobile / server — đã kiểm tệp ``inference.yml``). Đo trên khung thật phim
"[全集]" (3257,1 s; 4490,1 s; 3245,4 s): bộ DÒ chữ v6 vẫn thấy hộp chữ, nhưng nhận dạng v6 small trả chuỗi RỖNG (bị lọc
ngưỡng, hộp biến mất khỏi kết quả) hoặc "误" 0,37–0,51. Nhận dạng PP-OCRv5 mobile đọc chính hộp đó thành "埃" ổn định
(0,77–0,99 ở cả 3 chỗ, 8/8 khung) — chữ gần hình nhất có trong bảng chữ.

Cách cứu (chỉ cho chữ Hán):

1. ứng viên = hộp bộ dò thấy mà kết quả nhận dạng không có (đọc rỗng / dưới ngưỡng), và hộp MỘT chữ đọc dưới
   :data:`WEAK_SINGLE_CHAR_CONFIDENCE`;
2. đọc lại ứng viên bằng bộ nhận dạng phụ (PP-OCRv5 mobile, nạp khi cần lần đầu, cùng thiết bị với OCR chính), trên
   CẢ ảnh đã tiền xử lý lẫn ảnh gốc (:func:`best_reading`);
3. chỉ nhận khi bản đọc lại nằm trong bảng :data:`RESCUE_MAP` ("埃" → "诶") với tin cậy ≥ :data:`RESCUE_MIN_CONFIDENCE`.
   "埃" đứng MỘT MÌNH thành cả hộp chữ không phải lời thoại (chỉ có trong 埃及, 尘埃…), nên không đổi nhầm câu thật.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence

__all__ = [
    "RESCUE_MAPPED_MIN_CONFIDENCE",
    "RESCUE_MAP",
    "RESCUE_MIN_CONFIDENCE",
    "RESCUE_RECOGNITION_MODEL",
    "WEAK_SINGLE_CHAR_CONFIDENCE",
    "RescueCandidate",
    "apply_rescues",
    "best_reading",
    "crop_rect",
    "detected_polygons",
    "find_rescue_candidates",
]

#: Bộ nhận dạng phụ dùng để đọc lại hộp không đọc được.
RESCUE_RECOGNITION_MODEL = "PP-OCRv5_mobile_rec"
#: Bản đọc của bộ nhận dạng phụ → chữ đúng (chữ không có trong bảng chữ của mọi mô hình).
RESCUE_MAP: dict[str, str] = {"埃": "诶"}
RESCUE_MIN_CONFIDENCE = 0.6
#: [v3.23.742] Ngưỡng NHẬN bản đọc có trong :data:`RESCUE_MAP`. "诶" thật ở phim nối 86 tập (5660,3 s) chỉ đọc được "埃" 0,334
#: → sót câu. Đo cả phim (8351 khung mẫu, 455 lần cứu chữ): "埃" ra đúng 1 lần, ở chỗ "诶" thật; không lần nào trên vật trong
#: cảnh (rác cảnh ra 福 / 中 / số / chữ Latin). Hộp được cứu mang tin cậy tối thiểu :data:`RESCUE_MIN_CONFIDENCE`.
RESCUE_MAPPED_MIN_CONFIDENCE = 0.3
#: Hộp MỘT chữ đọc dưới mức này cũng được đọc lại ("诶" → "误" 0,51 ở 3245,4 s).
WEAK_SINGLE_CHAR_CONFIDENCE = 0.6
#: Hộp nhận dạng trùng hộp dò khi phần giao / hộp nhỏ hơn ≥ mức này.
_SAME_BOX_OVERLAP = 0.5

Polygon = Sequence[Sequence[float]]


@dataclass(frozen=True)
class RescueCandidate:
    """Hộp cần đọc lại: ``polygon`` theo toạ độ ảnh đưa vào OCR; ``replaces`` = vị trí hộp cũ cần thay (``None`` = thêm)."""

    polygon: tuple[tuple[float, float], ...]
    replaces: int | None


def _rect(polygon: Polygon) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap(a: Polygon, b: Polygon) -> float:
    ax0, ay0, ax1, ay1 = _rect(a)
    bx0, by0, bx1, by1 = _rect(b)
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return inter / smaller if smaller > 0 else 0.0


def _is_han(ch: str) -> bool:
    return "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿"


def find_rescue_candidates(
    detected: Sequence[Polygon], recognized: Sequence[OcrTextBox],
) -> list[RescueCandidate]:
    """Hộp dò được mà không có kết quả nhận dạng, và hộp một chữ Hán đọc yếu."""
    candidates: list[RescueCandidate] = []
    for polygon in detected:
        if len(polygon) < 3:
            continue
        if not any(box.polygon and _overlap(polygon, box.polygon) >= _SAME_BOX_OVERLAP for box in recognized):
            candidates.append(RescueCandidate(tuple((float(p[0]), float(p[1])) for p in polygon), None))
    for index, box in enumerate(recognized):
        text = (box.text or "").strip()
        if (
            box.polygon and len(text) == 1 and _is_han(text)
            and float(box.confidence) < WEAK_SINGLE_CHAR_CONFIDENCE
        ):
            candidates.append(RescueCandidate(tuple((float(p[0]), float(p[1])) for p in box.polygon), index))
    return candidates


def apply_rescues(
    result: OcrFrameResult, candidates: Sequence[RescueCandidate], readings: Sequence[tuple[str, float]],
) -> tuple[OcrFrameResult, int]:
    """Gắn bản đọc lại hợp lệ (qua :data:`RESCUE_MAP`) vào kết quả khung; trả ``(kết quả, số hộp được cứu)``."""
    boxes = list(result.text_boxes)
    rescued = 0
    for candidate, (text, score) in zip(candidates, readings, strict=True):
        target = RESCUE_MAP.get((text or "").strip())
        if target is None or float(score) < RESCUE_MAPPED_MIN_CONFIDENCE:
            continue
        # Tin cậy thấp ở đây là mô hình phụ không chắc chữ GẦN GIỐNG (chữ thật nằm ngoài bảng chữ), không phải đọc kém.
        box = OcrTextBox(
            text=target, confidence=Confidence(max(RESCUE_MIN_CONFIDENCE, min(1.0, float(score)))),
            polygon=[(int(round(x)), int(round(y))) for x, y in candidate.polygon],
        )
        if candidate.replaces is None:
            boxes.append(box)
        else:
            boxes[candidate.replaces] = box
        rescued += 1
    if not rescued:
        return result, 0
    return dataclasses.replace(result, text_boxes=boxes), rescued


def best_reading(alternatives: Sequence[tuple[str, float]]) -> tuple[str, float]:
    """Chọn bản đọc tốt nhất của một hộp trong các lần đọc lại (ảnh đã tiền xử lý / ảnh gốc): ưu tiên bản có trong
    :data:`RESCUE_MAP` với tin cậy cao nhất. Đo 6 khung "诶": ảnh đã tiền xử lý cứu 4/6, ảnh gốc 4/6, hợp lại 5/6."""
    if not alternatives:
        return "", 0.0
    mapped = [item for item in alternatives if (item[0] or "").strip() in RESCUE_MAP]
    return max(mapped or alternatives, key=lambda item: float(item[1]))


def detected_polygons(raw: Any) -> list[list[tuple[float, float]]]:
    """Hộp bộ DÒ chữ thấy (``dt_polys``) trong kết quả thô của PaddleOCR — kể cả hộp nhận dạng đọc rỗng / dưới ngưỡng
    (``rec_polys`` chỉ giữ hộp đã qua ngưỡng nhận dạng)."""
    target = getattr(raw, "res", raw)
    if isinstance(target, dict) and isinstance(target.get("res"), dict):
        target = target["res"]
    try:
        polys = target.get("dt_polys") if isinstance(target, dict) else getattr(target, "dt_polys", None)
    except Exception:  # noqa: BLE001 — định dạng lạ: không cứu, không làm hỏng lượt OCR
        return []
    if polys is None:
        return []
    out: list[list[tuple[float, float]]] = []
    for polygon in polys:
        try:
            out.append([(float(point[0]), float(point[1])) for point in polygon])
        except (TypeError, ValueError, IndexError):
            continue
    return out


def crop_rect(polygon: Sequence[Sequence[float]], width: int, height: int, pad: int = 4) -> tuple[int, int, int, int] | None:
    """Hình chữ nhật cắt (x0, y0, x1, y1) bao hộp, nới ``pad`` px, kẹp trong ảnh; ``None`` nếu rỗng."""
    x0, y0, x1, y1 = _rect(polygon)
    left, top = max(0, int(x0) - pad), max(0, int(y0) - pad)
    right, bottom = min(width, int(x1 + 0.999) + pad), min(height, int(y1 + 0.999) + pad)
    if right - left < 4 or bottom - top < 4:
        return None
    return left, top, right, bottom
