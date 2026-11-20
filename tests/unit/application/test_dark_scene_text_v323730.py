"""[v3.23.730] Lượt "xuất dữ liệu thô OCR" phim "大乾悍刀行" (84 tập nối, 2 giờ 32 phút, model small, bản 3.23.728) —
dữ liệu ở ``test data/dqhdx_v3_23_728`` (seraw + 8 đoạn video vùng ROI quanh các điểm đã kiểm bằng ảnh).

Lỗi tìm thấy khi đối chiếu từng khung với video ROI:

* 26 câu rác từ chữ thư pháp MỰC ĐEN trên cuộn giấy (94:03–94:57) và thánh chỉ (20:22) — OCR đọc chắc tay (0,90–1,00)
  nên bộ lọc chữ tối 3.23.727 (chỉ bỏ hộp tin cậy < 0,90) để lọt;
* hoa văn thêu trên áo đỏ đọc thành "心福" (97:20);
* phụ đề thật bị đọc lẫn kana: "三ニー" (三 二一), "ハ八八" (八八八), "ハ刀", "防弾衣".
"""

from __future__ import annotations

import gzip
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from subtitles_extractor.application.services.bright_text_check import drop_dark_boxes
from subtitles_extractor.application.services.subtitle_pipeline.box_geometry import (
    drop_oversized_glyph_events,
)
from subtitles_extractor.application.services.subtitle_pipeline.geometry_check import (
    arbitrate_disputed_verdict,
)
from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
    kana_to_han,
    unify_minority_kana,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

DATA = Path(__file__).resolve().parents[3] / "test data" / "dqhdx_v3_23_728"


def _box(text: str, conf: float, x0: float, x1: float, y0: float, y1: float) -> OcrTextBox:
    return OcrTextBox(text=text, confidence=Confidence(conf), polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _ev(text: str, start: float, end: float, conf: float = 1.0, frames: int = 20) -> SubtitleEvent:
    return SubtitleEvent(
        index=1, text=text, interval=TimeInterval(start, end), confidence=Confidence(conf), frame_count=frames,
    )


# ── kana đọc nhầm từ chữ Hán ─────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("text", "expected"), [
    ("三ニー", "三二一"), ("ハ八八", "八八八"), ("ハ刀", "八刀"), ("防弾衣", "防弹衣"),
    ("カード", "カード"),     # không có chữ Hán → không đủ căn cứ
    ("ー", "ー"),
    ("你好", "你好"),
])
def test_kana_lookalikes_become_han(text: str, expected: str) -> None:
    assert kana_to_han(text) == expected


def test_kana_unified_only_in_chinese_films() -> None:
    chinese = ["三ニー"] + ["王爷你好"] * 40
    assert unify_minority_kana(chinese) == {0: "三二一"}
    japanese = ["三ニー"] + ["ありがとう王子"] * 40      # phim tiếng Nhật: hiragana khắp nơi → giữ nguyên
    assert unify_minority_kana(japanese) == {}
    assert unify_minority_kana(["三ニー"] * 5) == {}    # quá ít dòng


def test_reread_kana_does_not_replace_han() -> None:
    """Lượt thật: soát câu "hai bản đọc" đổi "八刀" thành "ハ刀" (đọc lại trên ảnh gốc)."""
    assert arbitrate_disputed_verdict("八刀", "ハ刀", []) == "八刀"


# ── chữ tối trong cảnh, đọc chắc tay ────────────────────────────────────────────────────────────────────────────────
def test_confident_box_without_any_bright_pixel_is_dropped() -> None:
    image = np.full((118, 720, 3), 185, np.uint8)               # giấy vàng nhạt (độ chói < 200)
    image[20:100, 230:340] = (40, 35, 30)                       # nét mực đen
    scroll = _box("王", 0.99, 221, 353, 0, 117)
    result, dropped = drop_dark_boxes(OcrFrameResult(0, 5654.3, [scroll]), image, 200)
    assert dropped == 1 and result.text_boxes == []


def test_real_scroll_frames_from_roi_clip() -> None:
    """Khung thật cắt từ video ROI: hộp chữ thư pháp (tin cậy 0,98–1,00) không có điểm sáng nào → bỏ; hộp phụ đề "王爷"
    (5659,4 s — ngay sau cuộn giấy) có chữ trắng → giữ."""
    from subtitles_extractor.infrastructure.video.roi_clip_reader import RoiClipReader

    reader = RoiClipReader(DATA / "clips" / "scroll_text_5655.roi.mp4", DATA / "clips" / "scroll_text_5655.roi.json")
    try:
        scroll = reader.frame_at(5655.7)
        assert scroll is not None
        boxes = [_box("安", 1.00, 94, 206, 0, 117), _box("海", 1.00, 226, 340, 0, 117)]
        assert drop_dark_boxes(OcrFrameResult(0, 5655.7, boxes), scroll, 200)[1] == 2
    finally:
        reader.close()
    reader = RoiClipReader(DATA / "clips" / "kana_88_2287.roi.mp4", DATA / "clips" / "kana_88_2287.roi.json")
    try:
        subtitle = reader.frame_at(2287.6)
        assert subtitle is not None
        real = _box("八八八", 0.69, 271, 449, 22, 90)
        assert drop_dark_boxes(OcrFrameResult(0, 2287.6, [real]), subtitle, 200)[1] == 0
    finally:
        reader.close()


# ── hoa văn chữ to cả hai chiều ──────────────────────────────────────────────────────────────────────────────────────
def _frames(text: str, x0: float, x1: float, y0: float, y1: float, start: float, n: int = 20) -> list[OcrFrameResult]:
    return [OcrFrameResult(i, start + i * 0.04, [_box(text, 0.74, x0, x1, y0, y1)]) for i in range(n)]


def test_oversized_low_confidence_pattern_dropped_but_real_lines_kept() -> None:
    height = 66.0
    frames = (
        _frames("心福", 111, 381, -4, 121, 10.0)        # cao 1,89, rộng 2,05 / chữ, tin cậy 0,74 → hoa văn
        + _frames("但", 326, 417, 13, 109, 20.0)        # chỉ cao (hộp phình dọc) → giữ
        + _frames("哎", 283, 435, 20, 87, 30.0)         # chỉ rộng (hộp phình ngang) → giữ
        + _frames("王", 221, 353, -4, 122, 40.0)        # to cả hai chiều nhưng đọc chắc (1,0) → không thuộc luật này
    )
    events = [_ev("心福", 10.0, 10.8, 0.74), _ev("但", 20.0, 20.8, 0.74), _ev("哎", 30.0, 30.8, 0.74),
              _ev("王", 40.0, 40.8, 1.0)]
    kept, dropped = drop_oversized_glyph_events(events, frames, height)
    assert dropped == 1 and [e.text for e in kept] == ["但", "哎", "王"]
    assert drop_oversized_glyph_events(events, frames, None) == (events, 0)


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

    raw = DATA / "dqhdx.seraw.json.gz"
    run = json.load(gzip.open(raw))["meta"]["run"]
    names = {f.name for f in fields(SubtitleBuilderConfig)}
    config = SubtitleBuilderConfig(**{k: v for k, v in run["builder"].items() if k in names})
    frames, meta = load_raw_ocr(raw)
    return SubtitleBuilder(config).build(frames, roi=Roi(*meta.roi_xywh))


def test_real_run_has_no_kana_and_no_embroidery(replayed: list[SubtitleEvent]) -> None:
    texts = [e.text for e in replayed]
    assert not [t for t in texts if any("゠" <= ch <= "ヿ" or ch == "弾" for ch in t)]
    assert "三二一" in texts and "八八八" in texts
    assert "心福" not in texts
    # Cuộn giấy (chữ tối): bỏ ở bước OCR; [v3.23.734] nay bỏ được cả ở bước dựng (14 câu).
    assert len(replayed) == 3622  # [v3.23.734] −14 câu cuộn sách mực tối (drop_scene_text_block_events)


def test_roi_clips_line_up_with_seraw_times() -> None:
    from subtitles_extractor.infrastructure.video.roi_clip_reader import RoiClipReader

    assert len(list((DATA / "clips").glob("*.roi.mp4"))) == 8
    reader = RoiClipReader(DATA / "clips" / "kana_88_2287.roi.mp4", DATA / "clips" / "kana_88_2287.roi.json")
    try:
        before, during = reader.frame_at(2287.05), reader.frame_at(2287.6)
        assert before is not None and during is not None
        core = lambda img: int((np.asarray(img).max(axis=2)[:, 270:450] >= 240).sum())  # noqa: E731
        assert core(during) > 300 and core(before) < 50       # chữ "八八八" bắt đầu ở 2287,214 s trong seraw
    finally:
        reader.close()
