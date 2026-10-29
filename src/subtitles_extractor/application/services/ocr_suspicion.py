"""[v3.23.654] Dòng phụ đề "nghi OCR sai" cho trang Chỉnh sửa (hàm thuần).

PP-OCRv6 trả độ tin cậy ~1,0 cho hầu hết mọi dòng, kể cả dòng đọc sai — bộ lọc "Lỗi OCR" theo độ tin cậy gần như không
bắt được gì. Tín hiệu dùng ở đây là HÌNH THÁI của kết quả:

* ``few_frames``: dòng chỉ thấy ở rất ít khung (≤ 3) — dễ là mảnh chuyển cảnh/chữ chạy;
* ``progressive``: câu là phần đầu của câu liền sau (hiện dần: ``我`` → ``我只是``);
* ``near_duplicate``: gần giống câu liền kề mà không trùng (OCR tách một câu thành hai biến thể);
* ``mixed_script``: chữ Latin/số lẫn trong câu chữ Hán (watermark dính: ``嗯不对OVONET``);
* ``low_confidence``: độ tin cậy dưới ngưỡng (khi engine có trả điểm thật);
* ``restored_char``: [v3.23.655] khâu dựng phụ đề đã khôi phục một ký tự OCR làm rơi (vd ``一`` đầu câu) — đúng phần lớn
  nhưng thỉnh thoảng thêm nhầm, nên đưa ra soát.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rapidfuzz import fuzz

SUSPECT_FEW_FRAMES = "few_frames"
SUSPECT_PROGRESSIVE = "progressive"
SUSPECT_NEAR_DUPLICATE = "near_duplicate"
SUSPECT_MIXED_SCRIPT = "mixed_script"
SUSPECT_LOW_CONFIDENCE = "low_confidence"
SUSPECT_RESTORED_CHAR = "restored_char"   # [v3.23.655] = OCR_NOTE_RESTORED của khâu dựng phụ đề

_FEW_FRAMES = 3
_NEIGHBOR_GAP_SEC = 0.5
_NEAR_DUPLICATE_RATIO = 80.0
_LOW_CONFIDENCE = 0.6
_CJK = re.compile(r"[㐀-鿿]")
_LATIN_RUN = re.compile(r"[A-Za-z0-9]{2,}")


def _key(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def ocr_suspicions(events: Sequence[object]) -> dict[int, list[str]]:
    """``{vị trí dòng: [lý do, …]}`` các dòng đáng soát lại chữ OCR."""
    found: dict[int, list[str]] = {}

    def add(position: int, reason: str) -> None:
        bucket = found.setdefault(position, [])
        if reason not in bucket:
            bucket.append(reason)

    keys = [_key(str(getattr(event, "text", "") or "")) for event in events]
    for position, event in enumerate(events):
        text = keys[position]
        if not text:
            continue
        frames = int(getattr(event, "frame_count", 0) or 0)
        if 0 < frames <= _FEW_FRAMES:
            add(position, SUSPECT_FEW_FRAMES)
        confidence = float(getattr(event, "confidence", 1.0) or 0.0)
        if 0.0 < confidence < _LOW_CONFIDENCE:
            add(position, SUSPECT_LOW_CONFIDENCE)
        if SUSPECT_RESTORED_CHAR in (getattr(event, "ocr_notes", ()) or ()):
            add(position, SUSPECT_RESTORED_CHAR)
        if _CJK.search(text) and _LATIN_RUN.search(text):
            add(position, SUSPECT_MIXED_SCRIPT)
        if position + 1 < len(events):
            following = keys[position + 1]
            gap = float(getattr(events[position + 1], "start_sec", 0.0)) - float(getattr(event, "end_sec", 0.0))
            if following and gap <= _NEIGHBOR_GAP_SEC:
                if len(following) > len(text) and following.startswith(text):
                    add(position, SUSPECT_PROGRESSIVE)
                elif following != text and fuzz.ratio(text, following) >= _NEAR_DUPLICATE_RATIO:
                    add(position, SUSPECT_NEAR_DUPLICATE)
                    add(position + 1, SUSPECT_NEAR_DUPLICATE)
    return dict(sorted(found.items()))


__all__ = [
    "SUSPECT_FEW_FRAMES",
    "SUSPECT_LOW_CONFIDENCE",
    "SUSPECT_MIXED_SCRIPT",
    "SUSPECT_NEAR_DUPLICATE",
    "SUSPECT_PROGRESSIVE",
    "SUSPECT_RESTORED_CHAR",
    "ocr_suspicions",
]
