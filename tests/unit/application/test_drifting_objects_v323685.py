"""[v3.23.685] Câu có hộp chữ TRÔI ngang đều qua khung là vật trong cảnh, không phải phụ đề.

Số liệu chép từ OCR thô thật của phim (``fixed_tron_bo.seraw.json.gz`` và tập 1, bước 0,12 s): "00" đọc từ vòng đai lưng
ở giây 45 trôi 32–40 px trong 0,7 s; chữ phụ đề đứng yên (câu dài nhất cũng chỉ rung hộp 0,08 lần bề rộng).
"""

from __future__ import annotations

import json
from pathlib import Path

from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
from subtitles_extractor.application.services.subtitle_pipeline.box_geometry import (
    MAX_DRIFT_WIDTH_RATIO,
    drift_ratio,
    drop_drifting_events,
)
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi, TextOrientation
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

STEP = 0.12
ROI = Roi(x=57, y=819, width=603, height=119)

#: Tập 1, giây 45: (thời điểm, chữ, tin cậy, x0, y0, x1, y1) — kể cả khung đọc thành "-00".
BELT_RING = [
    (45.0, "00", 0.95, 293, 13, 312, 85),
    (45.36, "00", 0.97, 303, 24, 334, 96),
    (45.48, "-00", 0.83, 302, 0, 343, 89),
    (45.6, "00", 0.96, 315, 0, 361, 75),
    (45.72, "00", 0.82, 322, 0, 362, 66),
]


def _box(text: str, conf: float, x0: int, y0: int, x1: int, y1: int) -> OcrTextBox:
    return OcrTextBox(text=text, confidence=Confidence(conf), polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _line(text: str, shift: int = 0) -> OcrTextBox:
    width = 68 * len(text)
    x0 = (603 - width) // 2 + shift
    return _box(text, 1.0, x0, 20, x0 + width, 88)


def _frames(spec: list[tuple[float, list[OcrTextBox]]]) -> list[OcrFrameResult]:
    return [OcrFrameResult(frame_index=i, timestamp_sec=t, text_boxes=b) for i, (t, b) in enumerate(spec)]


def _event(text: str, start: float, end: float) -> SubtitleEvent:
    return SubtitleEvent(index=1, text=text, interval=TimeInterval(start, end), confidence=Confidence(1.0), frame_count=5)


def test_belt_ring_drifts_and_real_lines_do_not() -> None:
    ring = _frames([(t, [_box(x, c, *xy)]) for t, x, c, *xy in BELT_RING])
    kept, dropped = drop_drifting_events([_event("00", 45.0, 45.57)], ring, STEP)
    assert dropped == 1 and kept == []
    # Câu dài, hộp rung qua lại ±20 px (OCR cắt hộp mỗi khung một khác) — không đều nên không phải trôi.
    jitter = [0, 20, -18, 15, -20, 12, -15, 20, -10, 18]
    long_line = _frames([(10 + i * STEP, [_line("终于被我逮到了吧", s)]) for i, s in enumerate(jitter)])
    assert drop_drifting_events([_event("终于被我逮到了吧", 10.0, 11.1)], long_line, STEP)[1] == 0
    # Trôi đều nhưng ít (0,15 lần bề rộng — câu thật trôi nhiều nhất cả phim) → giữ.
    slight = _frames([(20 + i * STEP, [_line("好", -3 * i)]) for i in range(5)])
    assert drift_ratio([(20 + i * STEP, 302 - 3 * i, 68) for i in range(5)]) < MAX_DRIFT_WIDTH_RATIO
    assert drop_drifting_events([_event("好", 20.0, 20.5)], slight, STEP)[1] == 0


def test_partial_reading_frame_is_not_position_evidence() -> None:
    """Phim thật 8129 s: khung ngay trước "叶然小心" chỉ đọc được "心" với hộp phình sang phải (x 393–603)."""
    spec = [(8129.04, [_box("心", 0.66, 393, 0, 603, 119)])]
    spec += [(8129.16 + i * STEP, [_box("叶然小心", 1.0, 181, 20, 424, 88)]) for i in range(5)]
    assert drop_drifting_events([_event("叶然小心", 8129.16, 8129.64)], _frames(spec), STEP)[1] == 0


def test_too_few_frames_give_no_verdict() -> None:
    assert drift_ratio([(1.0, 100.0, 40.0), (1.12, 140.0, 40.0)]) == 0.0
    assert drift_ratio([]) == 0.0


def _real_episode_frames() -> list[OcrFrameResult]:
    data = json.loads((Path(__file__).parent / "data" / "ep01_first60s_ocr_v323685.json").read_text(encoding="utf-8"))
    return [
        OcrFrameResult(
            frame_index=i, timestamp_sec=t,
            text_boxes=[OcrTextBox(text=x, confidence=Confidence(c), polygon=[tuple(p) for p in poly]) for x, c, poly in boxes],
        )
        for i, (t, boxes) in enumerate(data["frames"])
    ]


def test_builder_drops_belt_ring_on_real_episode() -> None:
    """OCR thô thật tập 1 (0–60 s): bản 3.23.684 còn giữ "00" ở giây 45; các câu thoại xung quanh phải còn nguyên."""
    texts = [e.text for e in SubtitleBuilder(SubtitleBuilderConfig(sample_step_sec=STEP)).build(_real_episode_frames(), roi=ROI)]
    assert "00" not in texts
    assert {"恭喜宿主", "突破大乘境大圆满", "天下无敌"} <= set(texts)


def test_vertical_roi_is_left_alone() -> None:
    ring = _frames([(t, [_box(x, c, *xy)]) for t, x, c, *xy in BELT_RING])
    vertical = Roi(x=0, y=0, width=119, height=603, orientation=TextOrientation.VERTICAL)
    SubtitleBuilder(SubtitleBuilderConfig(sample_step_sec=STEP)).build(ring, roi=vertical)     # không ném lỗi
