"""[v3.23.742] Câu "诶" bị SÓT (phim nối 86 tập, 5660,32–5660,88 s) — tìm ra nhờ quét điểm ảnh video ROI (giai đoạn 0).

Bộ dò thấy hộp chữ; PP-OCRv6 không có "诶" trong bảng chữ → rỗng; bộ cứu chữ PP-OCRv5 đọc "埃" (→ "诶") tin cậy 0,334 <
ngưỡng 0,6 → bỏ. Đo cả phim (8351 khung mẫu, 455 lần cứu chữ): "埃" xuất hiện 1 lần, đúng chỗ "诶" thật; KHÔNG lần nào
trên vật trong cảnh → ngưỡng nhận riêng cho bản đổi "埃" hạ về 0,3; hộp được cứu mang tin cậy tối thiểu 0,6 để bộ dựng
không bỏ câu một chữ (tin cậy thấp do mô hình phụ không chắc chữ gần giống, không phải đọc kém).
"""

from __future__ import annotations

from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult
from subtitles_extractor.infrastructure.ocr.unreadable_rescue import (
    RESCUE_MIN_CONFIDENCE,
    RescueCandidate,
    apply_rescues,
)

POLY = [(330, 70), (390, 70), (390, 128), (330, 128)]


def test_weak_mapped_reading_is_rescued_with_floor_confidence() -> None:
    frame = OcrFrameResult(0, 5660.6, [])
    out, rescued = apply_rescues(frame, [RescueCandidate(polygon=POLY, replaces=None)], [("埃", 0.334)])
    assert rescued == 1 and out.text_boxes[0].text == "诶"
    assert float(out.text_boxes[0].confidence) >= RESCUE_MIN_CONFIDENCE


def test_too_weak_or_unmapped_reading_is_not_rescued() -> None:
    frame = OcrFrameResult(0, 5660.6, [])
    assert apply_rescues(frame, [RescueCandidate(polygon=POLY, replaces=None)], [("埃", 0.2)])[1] == 0
    assert apply_rescues(frame, [RescueCandidate(polygon=POLY, replaces=None)], [("福", 0.9)])[1] == 0


def test_builder_keeps_rescued_single_char_line() -> None:
    from subtitles_extractor.application.services.subtitle_pipeline.orchestrator import (
        SubtitleBuilder,
    )
    from subtitles_extractor.composition.builder_config import app_default_builder_config
    from subtitles_extractor.domain.value_objects.roi import Roi

    step = 0.04
    frames = []
    for i in range(60):
        t = round(5659.6 + i * step, 3)
        boxes = []
        if 5660.32 <= t <= 5660.88:
            frame = OcrFrameResult(i, t, [])
            boxes = list(apply_rescues(frame, [RescueCandidate(polygon=POLY, replaces=None)], [("埃", 0.334)])[0].text_boxes)
        frames.append(OcrFrameResult(i, t, boxes, continues_previous=bool(boxes) and 5660.32 < t))
    events = SubtitleBuilder(app_default_builder_config(step)).build(frames, roi=Roi(0, 774, 720, 136))
    assert [e.text for e in events] == ["诶"]
