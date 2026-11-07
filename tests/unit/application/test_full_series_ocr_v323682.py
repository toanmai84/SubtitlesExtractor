"""[v3.23.682] Lỗi OCR tìm được khi so cả phim nối 86 tập (2 giờ 19 phút) với đáp án người dùng, đã đối chiếu ảnh.

Dữ liệu dưới đây chép từ OCR thô thật của phim (``fixed_tron_bo.seraw.json.gz``, bước 0,12 s): chữ, độ tin cậy, khung hộp.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
from subtitles_extractor.application.services.subtitle_pipeline.box_geometry import (
    drop_implausible_boxes,
    is_plausible_subtitle_box,
    typical_glyph_height,
)
from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
    _should_split_boundary,
    is_stable_distinct_pair,
)
from subtitles_extractor.application.services.subtitle_pipeline.event_refinement import drop_flash_fragments
from subtitles_extractor.application.services.subtitle_pipeline.text_correction import unify_minority_traditional
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

STEP = 0.12
ROI = Roi(x=57, y=819, width=603, height=119)


def _box(text: str, conf: float, x0: int, y0: int, x1: int, y1: int) -> OcrTextBox:
    return OcrTextBox(text=text, confidence=Confidence(conf), polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _line(text: str, conf: float = 1.0) -> OcrTextBox:
    """Câu thoại cỡ chữ thật của phim: cao 68 px, mỗi chữ Hán rộng ~68 px, căn giữa ROI rộng 603 px."""
    width = 68 * len(text)
    x0 = (603 - width) // 2
    return _box(text, conf, x0, 20, x0 + width, 88)


def _frames(spec: list[tuple[float, list[OcrTextBox]]]) -> list[OcrFrameResult]:
    return [OcrFrameResult(frame_index=i, timestamp_sec=t, text_boxes=b) for i, (t, b) in enumerate(spec)]


def _context(start: float, count: int = 40) -> list[tuple[float, list[OcrTextBox]]]:
    """Đủ câu thoại thường để bộ lọc hình học biết cỡ chữ (≥ 30 hộp ≥ 3 chữ Hán)."""
    return [(start + i * STEP, [_line("师叔今天要是有个")]) for i in range(count)]


def _build(spec, step: float = STEP) -> list[str]:
    events = SubtitleBuilder(SubtitleBuilderConfig(sample_step_sec=step)).build(_frames(spec), roi=ROI)
    return [e.text for e in events]


# ── Hình học hộp chữ ──
def test_scene_objects_read_as_text_are_dropped() -> None:
    frames = _frames(_context(0.0))
    height = typical_glyph_height(frames)
    assert height == 68
    # vòng đai lưng / hoa văn / bàn tay / chữ khắc dọc — số liệu thật
    assert not is_plausible_subtitle_box(_box("88", 0.96, 250, 30, 266, 49), height)          # chữ cao 19 px
    assert not is_plausible_subtitle_box(_box("入", 0.94, 195, 0, 468, 119), height)          # 1 chữ rộng 273 px
    assert not is_plausible_subtitle_box(_box("气气宗", 0.98, 534, 0, 583, 119), height)      # hộp đứng 49×119
    # chữ thật
    assert is_plausible_subtitle_box(_box("切", 1.0, 264, 20, 341, 93), height)
    assert is_plausible_subtitle_box(_box("我", 1.0, 256, 0, 345, 119), height)
    assert is_plausible_subtitle_box(_box("那师兄大可以试试", 1.0, 70, 24, 533, 85), height)
    # chữ thật đứng trên nền sáng: hộp phình rộng 209 px nhưng tin cậy 1,0 → giữ ("哦" trên thắt lưng trắng)
    assert is_plausible_subtitle_box(_box("哦", 1.0, 146, 7, 355, 117), height)


def test_geometry_filter_needs_enough_samples_and_turns_itself_off() -> None:
    few = _frames([(0.0, [_box("入", 0.8, 0, 0, 300, 119)])])
    assert typical_glyph_height(few) is None
    assert drop_implausible_boxes(few)[1] == 0
    # Toạ độ hộp lệch đơn vị (mỗi chữ hẹp bất thường ở MỌI câu): không xoá phụ đề hàng loạt.
    squeezed = _frames([(i * 0.1, [_box("十多个人肾虚", 0.95, 300, 20, 400, 80)]) for i in range(40)])
    kept, dropped = drop_implausible_boxes(squeezed)
    assert dropped == 0 and len(kept) == 40


def test_builder_drops_hand_read_as_char() -> None:
    spec = _context(0.0) + [(10.0 + i * STEP, [_box("入", 0.81, 195, 0, 468, 119)]) for i in range(13)]
    assert "入" not in _build(spec)


# ── Câu một chữ ngắn ──
def test_two_frame_single_char_kept_at_default_step() -> None:
    """"我" chỉ 2 khung (0,24 s), tin cậy 1,0, ngay trước "我不管了" — trước đây bị coi là chớp nhiễu."""
    spec = _context(0.0) + [
        (10.0, [_line("你")]), (10.12, [_line("你")]), (10.24, [_line("你")]),
        (10.84, [_line("我")]), (10.96, [_line("我")]),
        *[(12.16 + i * STEP, [_line("我不管了")]) for i in range(5)],
    ]
    texts = _build(spec)
    assert "我" in texts and "我不管了" in texts


def test_two_frame_single_char_still_flicker_at_fine_step_or_low_conf() -> None:
    fine = [(i * 0.04, [_line("师叔今天要是有个")]) for i in range(40)] + [
        (5.0, [_line("我")]), (5.04, [_line("我")]),
    ]
    assert "我" not in _build(fine, step=0.04)                # 2 khung × 0,04 s = 0,08 s
    low = _context(0.0) + [(10.0, [_line("一", 0.81)]), (10.12, [_line("一", 0.81)])]
    assert "一" not in _build(low)


def test_flash_duration_counts_frames_for_cjk_only() -> None:
    def ev(text: str, start: float, end: float, frames: int) -> SubtitleEvent:
        return SubtitleEvent(
            index=1, text=text, interval=TimeInterval(start, end), confidence=Confidence(1.0), frame_count=frames,
        )

    kept = drop_flash_fragments([ev("切", 1.0, 1.15, 2), ev("V", 5.0, 5.12, 2)], sample_step_sec=STEP)
    assert [e.text for e in kept] == ["切"]
    assert drop_flash_fragments([ev("切", 1.0, 1.15, 2)]) == []        # không biết bước → như cũ


def test_long_stable_single_char_with_variant_readings_kept() -> None:
    """"别" đứng 1,4 s nhưng bộ nhận dạng dao động 别/別 nên tin cậy trung bình 0,67."""
    readings = [("别", 0.53), ("别", 0.82), ("別", 0.53), ("别", 0.54), ("別", 0.67), ("別", 0.62),
                ("別", 0.69), ("別", 0.75), ("别", 0.69), ("别", 0.82), ("别", 0.58), ("别", 0.86)]
    spec = _context(0.0) + [(10.0 + i * STEP, [_box(t, c, 264, 17, 339, 93)]) for i, (t, c) in enumerate(readings)]
    texts = _build(spec)
    assert any(t in ("别", "別") for t in texts)


# ── Hai câu liền nhau chỉ khác vài chữ ──
@pytest.mark.parametrize(("a", "b", "span_a", "span_b", "split"), [
    ("我去哪", "你去哪", 0.48, 0.48, True),
    ("我不闭关了", "好不闭关了", 1.56, 0.72, True),
    ("那师兄大可以试试", "那师兄今天就试试", 1.32, 2.0, True),
    ("你不要抛弃我", "不要抛弃我", 0.84, 0.84, True),
    ("就可以实现", "不就可以实现", 0.5, 1.0, False),          # OCR làm rơi "不" 0,5 s → một câu
    ("我去哪", "你去哪", 0.12, 0.48, False),                  # khối 1–2 khung là lỗi đọc
    ("想自毀灵台", "想自毁灵台", 0.52, 1.2, False),           # khác cách viết phồn/giản → một câu
    ("一码是一码", "码是一码", 1.0, 1.0, False),              # mất nét mảnh "一" ở đầu
    ("师兄BOR来了", "师兄来了", 1.0, 1.0, False),             # Latin chen vào = watermark dính
])
def test_split_boundary_rules(a: str, b: str, span_a: float, span_b: float, split: bool) -> None:
    assert _should_split_boundary(a, b, span_a, span_b) is split
    assert is_stable_distinct_pair(a, b, span_a, span_b, 5, 5) is (split and min(span_a, span_b) >= 0.3)


def test_builder_keeps_both_one_char_different_lines() -> None:
    spec = _context(0.0) + [
        *[(10.12 + i * STEP, [_line("我去哪")]) for i in range(5)],
        *[(10.72 + i * STEP, [_line("你去哪")]) for i in range(5)],
    ]
    texts = _build(spec)
    assert "我去哪" in texts and "你去哪" in texts


# ── Chữ phồn lẻ tẻ trong phim giản thể ──
def test_unify_minority_traditional() -> None:
    simplified = ["这个强者", "我决定了", "说得对"] * 20          # phim thật: 3 câu phồn / 1951 câu giản
    texts = [*simplified, "最強的五人"]
    assert unify_minority_traditional(texts) == {60: "最强的五人"}
    assert unify_minority_traditional(["最強的五人", "這個說法"] * 10) == {}       # phim phồn thể: giữ nguyên
    assert unify_minority_traditional(["最強的五人", "这个强者"]) == {}             # quá ít căn cứ
    spec = _context(0.0, 30)
    # 60 câu giản thể rời nhau (mỗi câu 5 khung, cách nhau 1 s, chữ khác nhau để không gộp)
    lines = ["说得对强者", "我决定了强", "这个强者来"]
    for i in range(60):
        spec += [(10.0 + i * 1.0 + k * STEP, [_line(lines[i % 3])]) for k in range(5)]
    spec += [(80.0 + i * STEP, [_line("最強的五人")]) for i in range(8)]
    assert "最强的五人" in _build(spec)
