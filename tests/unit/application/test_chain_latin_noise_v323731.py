"""[v3.23.731] Lượt "xuất dữ liệu thô OCR" phim "豪门Z外" (1080×1920, 93 phút, model small, bản 3.23.729) — dữ liệu ở
``test data/hmzw_v3_23_729`` (seraw + 3 đoạn video vùng ROI quanh các điểm đã kiểm bằng ảnh).

* 16:42: câu "给我揉一揉兴许我就好了" bị tách thành "给我揉揉兴许我就好了" (0,5 s, rơi chữ "一") + câu đúng. Gốc: watermark
  mờ "ENCIA…" ở mép trái làm 20 khung đọc "…好了ENC" — bản đọc thứ ba "khác kiểu" khiến bước gộp lần đọc cùng điểm ảnh
  bỏ cả chuỗi.
* 48:32: chữ in "Sweet dream" trên áo dính vào câu: "林娇娇这女人eet dream" (chữ thường — luật cắt watermark VIẾT HOA
  không bắt); hộp chữ lệch tâm 0,94 lần cỡ chữ.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import fields
from pathlib import Path

import pytest

from subtitles_extractor.application.services.subtitle_pipeline.box_geometry import (
    strip_offcenter_latin_tail,
)
from subtitles_extractor.application.services.subtitle_pipeline.read_fusion import (
    _frame_text,
    fuse_same_pixel_reads,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

DATA = Path(__file__).resolve().parents[3] / "test data" / "hmzw_v3_23_729"


def _box(text: str, x0: float, x1: float, conf: float = 1.0, y0: float = 18, y1: float = 101) -> OcrTextBox:
    return OcrTextBox(text=text, confidence=Confidence(conf), polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _frame(i: int, *boxes: OcrTextBox, same: bool = True) -> OcrFrameResult:
    return OcrFrameResult(frame_index=i, timestamp_sec=1002.231 + i * 0.042, text_boxes=list(boxes),
                          continues_previous=same and i > 0)


def _ev(text: str, start: float, end: float) -> SubtitleEvent:
    return SubtitleEvent(index=1, text=text, interval=TimeInterval(start, end), confidence=Confidence(1.0), frame_count=40)


# ── gộp lần đọc cùng điểm ảnh khi có chữ Latin dính chớp nhoáng ────────────────────────────────────────────────────
def test_chain_with_latin_noise_still_fuses_yi_variant() -> None:
    frames = (
        [_frame(i, _box("给我揉", 147, 365), _box("揉兴许我就好了", 421, 928)) for i in range(12)]
        + [_frame(i, _box("给我揉", 147, 366), _box("一揉兴许我就好了", 410, 928, 0.97)) for i in range(12, 22)]
        + [_frame(i, _box("给我揉一揉兴许我就好了", 144, 928)) for i in range(22, 37)]
        + [_frame(i, _box("给我揉一揉兴许我就好了", 144, 928), _box("ENC", 0, 80, 0.9)) for i in range(37, 57)]
    )
    out, fixed, _ = fuse_same_pixel_reads(frames, center_x=539.0, char_pitch=69.1, text_height=86.0)
    assert fixed > 0
    assert {_frame_text(f) for f in out} == {"给我揉一揉兴许我就好了"}     # bỏ cả "ENC" dính chớp nhoáng


def test_stable_latin_word_in_a_real_line_is_kept() -> None:
    """Chữ Latin thật đọc ổn định ("KTV") là bản đọc đa số → giữ nguyên."""
    frames = (
        [_frame(i, _box("我们去KTV", 380, 700)) for i in range(20)]
        + [_frame(i, _box("我们去", 380, 600)) for i in range(20, 24)]
    )
    out, _, _ = fuse_same_pixel_reads(frames, center_x=539.0, char_pitch=69.1, text_height=86.0)
    assert {_frame_text(f) for f in out} == {"我们去KTV"}


# ── đuôi Latin kéo lệch tâm ─────────────────────────────────────────────────────────────────────────────────────────
def test_offcenter_latin_tail_is_stripped_but_centered_latin_kept() -> None:
    frames = [OcrFrameResult(i, 10.0 + i * 0.04, [_box("林娇娇这女人eet dream", 313, 926)]) for i in range(30)]
    frames += [OcrFrameResult(100 + i, 20.0 + i * 0.04, [_box("我买了iPhone", 380, 700)]) for i in range(30)]
    events = [_ev("林娇娇这女人eet dream", 10.0, 11.2), _ev("我买了iPhone", 20.0, 21.2)]
    out, changed = strip_offcenter_latin_tail(events, frames, center_x=539.0, text_height=86.0)
    assert changed == 1 and [e.text for e in out] == ["林娇娇这女人", "我买了iPhone"]
    assert strip_offcenter_latin_tail(events, frames, None, 86.0) == (events, 0)


# ── phát lại cả lượt thật ────────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def replayed() -> list[SubtitleEvent]:
    from subtitles_extractor.application.dtos.extract_subtitles_dto import (
        SubtitleBuilderConfig,
    )
    from subtitles_extractor.application.services.subtitle_pipeline.orchestrator import (
        SubtitleBuilder,
    )
    from subtitles_extractor.domain.value_objects.roi import Roi
    from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import (
        load_raw_ocr,
    )

    raw = DATA / "hmzw.seraw.json.gz"
    run = json.load(gzip.open(raw))["meta"]["run"]
    names = {f.name for f in fields(SubtitleBuilderConfig)}
    config = SubtitleBuilderConfig(**{k: v for k, v in run["builder"].items() if k in names})
    frames, meta = load_raw_ocr(raw)
    return SubtitleBuilder(config).build(frames, roi=Roi(*meta.roi_xywh))


def _at(events: list[SubtitleEvent], t: float) -> list[str]:
    return [e.text for e in events if e.start_sec - 0.01 <= t <= e.end_sec + 0.01]


def test_real_run_fixes_split_and_latin_tail(replayed: list[SubtitleEvent]) -> None:
    assert _at(replayed, 1002.3) == ["给我揉一揉兴许我就好了"]
    assert _at(replayed, 1004.9) == ["给我揉一揉兴许我就好了"]
    assert _at(replayed, 2913.0) == ["林娇娇这女人"]
    assert _at(replayed, 2694.0) == ["强"]                      # "强" tin cậy 0,59 là phụ đề thật — giữ
    assert len(replayed) == 2131


def test_roi_clip_shows_the_watermark_that_broke_the_chain() -> None:
    import numpy as np

    from subtitles_extractor.infrastructure.video.roi_clip_reader import RoiClipReader

    assert len(list((DATA / "clips").glob("*.roi.mp4"))) == 3
    reader = RoiClipReader(DATA / "clips" / "yi_split_1002.roi.mp4", DATA / "clips" / "yi_split_1002.roi.json")
    try:
        image = reader.frame_at(1002.5)
        assert image is not None and image.shape[:2] == (140, 1080)
        core = int((np.asarray(image).max(axis=2)[:, 140:930] >= 240).sum())
        assert core > 1000                                         # phụ đề trắng đang hiện
    finally:
        reader.close()
