"""[v3.23.729] Lượt "xuất dữ liệu thô OCR" phim "[全集]" (98 phút, model small, bản 3.23.728) — dữ liệu ở
``test data/quanji_v3_23_728`` (seraw + 16 đoạn video vùng ROI quanh từng điểm đã kiểm bằng ảnh).

Đối chiếu bản người dùng sửa và từng khung: bản người dùng sửa KHÔNG phải đáp án tuyệt đối (gộp nhầm 3 câu khác nhau, cắt
ngắn 12 câu, bỏ sót 16 câu thật). Lỗi thật của ứng dụng đã khắc phục:

* 7 câu một chữ hiện 3–4 khung (啊, 啊, 嘿, 哼, 哎, 哎, 额; tin cậy 0,88–1,00) bị ``drop_flash_fragments`` xoá;
* logo tựa phim "一男王" (hộp bị mép trên ROI cắt, lệch dải phụ đề) lọt thành câu;
* soát câu "hai bản đọc" đổi SAI "龌龊" → "握龊", "注入" → "注人".
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import fields
from pathlib import Path

import pytest

from subtitles_extractor.application.services.subtitle_pipeline.box_geometry import (
    drop_clipped_offband_events,
)
from subtitles_extractor.application.services.subtitle_pipeline.event_refinement import (
    drop_flash_fragments,
)
from subtitles_extractor.application.services.subtitle_pipeline.geometry_check import (
    arbitrate_disputed_verdict,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

DATA = Path(__file__).resolve().parents[3] / "test data" / "quanji_v3_23_728"
STEP = 0.04


def _ev(text: str, start: float, end: float, frames: int, conf: float = 1.0) -> SubtitleEvent:
    return SubtitleEvent(
        index=1, text=text, interval=TimeInterval(start, end), confidence=Confidence(conf), frame_count=frames,
    )


# ── câu cảm thán chớp nhoáng ─────────────────────────────────────────────────────────────────────────────────────────
def test_short_single_char_read_over_three_frames_is_kept() -> None:
    real = [_ev("嘿", 4002.441, 4002.521, 3), _ev("哼", 4211.46, 4211.58, 4, 0.91), _ev("额", 5607.511, 5607.631, 4)]
    assert drop_flash_fragments(real, sample_step_sec=STEP) == real


@pytest.mark.parametrize(("event", "step"), [
    (_ev("我", 5.0, 5.04, 2), STEP),                 # 2 khung × 0,04 s — rác chớp như cũ
    (_ev("一", 10.0, 10.12, 3, 0.81), STEP),          # tin cậy thấp (vạt áo)
    (_ev("十", 48.28, 48.36, 3, 0.68), STEP),         # mảnh chữ đang hiện dần
    (_ev("嘿", 1.0, 1.12, 3), 0.0),                   # không biết bước lấy mẫu → như cũ
])
def test_flash_garbage_still_dropped(event: SubtitleEvent, step: float) -> None:
    assert drop_flash_fragments([event], sample_step_sec=step) == []


# ── logo bị mép ROI cắt ──────────────────────────────────────────────────────────────────────────────────────────────
def _frames(text: str, top: float, bottom: float, start: float, n: int) -> list[OcrFrameResult]:
    return [
        OcrFrameResult(frame_index=i, timestamp_sec=start + i * STEP, text_boxes=[OcrTextBox(
            text=text, confidence=Confidence(0.95), polygon=[(238, top), (385, top), (385, bottom), (238, bottom)])])
        for i in range(n)
    ]


def test_clipped_offband_logo_dropped_but_real_lines_kept() -> None:
    band, roi_height = (20.0, 74.0), 106.0
    frames = (
        _frames("一男王", -3, 40, 2.806, 20)                 # logo: mép trên cắt, chồng dải 0,37, cao 0,80 dải
        + _frames("我真的是冤枉的", -1, 51, 10.0, 20)         # sát mép, lệch dải nhưng đủ cỡ chữ (cao 0,96 dải)
        + _frames("说吧", 22, 106, 20.0, 20)                 # đáy bị ROI cắt nhưng nằm trên dải phụ đề
    )
    events = [_ev("一男王", 2.806, 3.566, 20), _ev("我真的是冤枉的", 10.0, 10.76, 20), _ev("说吧", 20.0, 20.76, 20)]
    kept, dropped = drop_clipped_offband_events(events, frames, band, roi_height)
    assert dropped == 1 and [e.text for e in kept] == ["我真的是冤枉的", "说吧"]
    assert drop_clipped_offband_events(events, frames, None, roi_height) == (events, 0)
    assert drop_clipped_offband_events(events, frames, (300.0, 380.0), roi_height) == (events, 0)  # toạ độ khung đầy đủ


# ── phân xử câu hai bản đọc ──────────────────────────────────────────────────────────────────────────────────────────
CORPUS = ["注入这祖龙之息", "同时注入才能解除", "注入灵力", "你和他的那些腌攒事", "以人炼丹的龌龊之辈"]


@pytest.mark.parametrize(("current", "reread", "expected"), [
    ("以人炼丹的龌龊之辈", "以人炼丹的握龊之辈", "以人炼丹的龌龊之辈"),    # chữ dính liền
    ("注入灵力", "注人灵力", "注入灵力"),                                  # 人/入: ngữ liệu phim có "注入"
    ("你和他的那些腌攒事", "你和他的那些腌臜事", "你和他的那些腌臜事"),    # sửa đúng vẫn được nhận
    ("清清自白", "清清白白", "清清白白"),                                  # hoà ngữ liệu → theo phiếu đọc lại
])
def test_disputed_reread_arbitration(current: str, reread: str, expected: str) -> None:
    assert arbitrate_disputed_verdict(current, reread, CORPUS) == expected


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

    raw = DATA / "quanji.seraw.json.gz"
    run = json.load(gzip.open(raw))["meta"]["run"]
    names = {f.name for f in fields(SubtitleBuilderConfig)}
    config = SubtitleBuilderConfig(**{k: v for k, v in run["builder"].items() if k in names})
    frames, meta = load_raw_ocr(raw)
    return SubtitleBuilder(config).build(frames, roi=Roi(*meta.roi_xywh))


def _at(events: list[SubtitleEvent], t: float) -> list[str]:
    return [e.text for e in events if e.start_sec - 0.05 <= t <= e.end_sec + 0.05]


def test_real_run_keeps_flash_lines_and_drops_logo(replayed: list[SubtitleEvent]) -> None:
    for t, text in [(3839.78, "啊"), (3951.84, "啊"), (4002.48, "嘿"), (4211.5, "哼"), (4235.4, "哎"),
                    (4260.65, "哎"), (5607.55, "额")]:
        assert _at(replayed, t) == [text], t
    assert not any("男王" in e.text for e in replayed)
    # Bản người dùng gộp nhầm — trên ảnh là hai câu khác nhau, ứng dụng phải giữ tách.
    for t, text in [(2957.5, "入了我青云宗的门"), (2959.0, "便是我青云宗的人"), (3997.9, "哎哟"), (3998.3, "哎呀"),
                    (5386.0, "即刻"), (5386.7, "立刻")]:
        assert _at(replayed, t) == [text], t
    app = re.findall(r"-->", (DATA / "quanji.app_v3_23_728.srt").read_text(encoding="utf-8"))
    assert len(replayed) == len(app) + 7 - 1


def test_roi_clips_line_up_with_seraw_times() -> None:
    import numpy as np

    from subtitles_extractor.infrastructure.video.roi_clip_reader import RoiClipReader

    clips = sorted((DATA / "clips").glob("*.roi.mp4"))
    assert len(clips) == 16
    # "诶" (3257,0–3257,45 s): chữ trắng có lõi sáng ≥ 240 rõ ràng, nhưng mọi mô hình đọc rỗng — chữ không có trong
    # bảng chữ của PP-OCRv6 small / PP-OCRv5 mobile.
    reader = RoiClipReader(DATA / "clips" / "ei_missed_3257.roi.mp4", DATA / "clips" / "ei_missed_3257.roi.json")
    try:
        with_text, without = reader.frame_at(3257.2), reader.frame_at(3256.85)
        assert with_text is not None and without is not None
        core = lambda img: int((np.asarray(img).max(axis=2)[:, 300:420] >= 240).sum())  # noqa: E731
        assert core(with_text) > 300 and core(without) < 50
    finally:
        reader.close()
