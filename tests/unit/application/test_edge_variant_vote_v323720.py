"""[v3.23.720] Một dòng phụ đề bị viterbi chia thành các nhóm xen kẽ theo bản đọc OCR (hơn / kém một chữ ở mép).

Dữ liệu: đúng các câu và chữ từng khung của OCR thật ``test3`` (chủ dự án gửi 13/11/2026, phụ đề chuẩn ``test3.srt``).
Mã 3.23.719:
- ``absorb_flickers`` giữ chữ câu "chủ": 3304,12 s "我来了" (16 khung) xen "我来" (4 khung) → "我来" + "我来了" tách đôi.
- ``post_merge_duplicates`` nhánh nhảy qua câu ngắn VỨT các lần đọc "瑶儿" (2 và 6 khung) để nối các mảnh "瑶", rồi so
  trọng số dồn (khung bản kia cộng vào) → "瑶" + "瑶儿" tách đôi, chữ đầu thiếu "儿".
"""

from __future__ import annotations

from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
from subtitles_extractor.application.services.flicker_absorber import absorb_flickers
from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
    frame_text_scope,
    line_reads,
    post_merge_duplicates,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

_CONFIG = SubtitleBuilderConfig(
    sample_step_sec=0.04, merge_gap_sec=0.2, min_duration_sec=0.15, use_viterbi=True, min_text_chars=0,
    temporal_padding_sec=0.0,
)


def _event(text: str, start: float, end: float, frames: int, conf: float = 0.98) -> SubtitleEvent:
    return SubtitleEvent(index=0, interval=TimeInterval(start, end), text=text, confidence=Confidence(conf),
                         frame_count=frames)


def _frames(start: float, texts: list[str], conf: float = 0.98) -> list[tuple[float, str, float]]:
    return [(round(start + 0.04 * index, 3), text, conf) for index, text in enumerate(texts)]


def test_absorbed_flickers_take_the_text_with_the_most_frames() -> None:
    """test3 3304,12 s: nhóm viterbi "我来了" 4 / "我来" 3 / "我来了" 3 / "我来" 1 / "我来了" 9 khung."""
    events = [
        _event("我来了", 3304.12, 3304.24, 4), _event("我来", 3304.28, 3304.36, 3),
        _event("我来了", 3304.40, 3304.48, 3), _event("我来", 3304.52, 3304.521, 1),
        _event("我来了", 3304.56, 3304.88, 9),
    ]
    out = absorb_flickers(events, min_duration_sec=0.15, similarity_threshold=0.75, merge_gap_sec=0.2)
    assert all(event.text == "我来了" for event in out)          # mã cũ: câu đầu "我来"


def test_skipped_variant_reads_are_kept_and_the_line_votes_by_frames() -> None:
    """test3 642,59 s: 瑶 3 / 瑶儿 2 / 瑶 3 / 瑶儿 6 / 瑶 1 / 瑶儿 5 khung (thứ tự thật)."""
    events = [
        _event("瑶", 642.585, 642.704, 3), _event("瑶儿", 642.705, 642.784, 2), _event("瑶", 642.785, 642.904, 3),
        _event("瑶儿", 642.905, 643.105, 6), _event("瑶", 643.145, 643.184, 1), _event("瑶儿", 643.185, 643.345, 5),
    ]
    reads = ["瑶"] * 3 + ["瑶儿"] * 2 + ["瑶"] * 3 + ["瑶儿"] * 6 + ["瑶"] + ["瑶儿"] * 5
    textful = [entry[0] for entry in _frames(642.585, reads)]
    with frame_text_scope(_frames(642.585, reads)):
        out = post_merge_duplicates(events, _CONFIG, textful)
    assert [event.text for event in out] == ["瑶儿"]              # mã cũ: ["瑶", "瑶儿"]
    assert abs(out[0].start_sec - 642.585) < 1e-6


def test_low_confidence_longer_read_does_not_win() -> None:
    """test3 1926,17 s: "你" 8 khung tin 1,0 / "你人" 3 khung tin 0,51 → giữ "你"."""
    events = [_event("你", 1926.165, 1926.315, 4, 1.0), _event("你人", 1926.325, 1926.404, 2, 0.51),
              _event("你", 1926.405, 1926.565, 5, 1.0)]
    frames = _frames(1926.165, ["你"] * 4, 1.0) + _frames(1926.325, ["你人"] * 2, 0.51) + _frames(1926.405, ["你"] * 4, 1.0)
    frames.append((1926.365, "你人", 0.51))
    with frame_text_scope(frames):
        out = post_merge_duplicates(events, _CONFIG, [entry[0] for entry in frames])
    assert [event.text for event in out] == ["你"]


def test_line_reads_extends_over_the_whole_line_and_stops_at_a_gap() -> None:
    frames = _frames(10.0, ["岁月", "岁", "岁", "岁", "岁月", "岁", "岁月", "岁月"], 0.95) + _frames(12.0, ["岁月"] * 5, 0.95)
    with frame_text_scope(frames):
        reads = line_reads(("岁月", "岁"), 10.0, 10.04)
    assert reads == {"岁月": (4, 0.95), "岁": (4, 0.95)}           # dòng sau (12 s, cách xa) không tính
    assert line_reads(("岁",), 10.0, 10.1) is None                  # ngoài phạm vi → không đoán


def test_line_built_by_absorption_still_reaches_the_minimum_duration() -> None:
    """Câu lẻ bị chia hai mảnh (mảnh sau 1 khung) gộp lại phải được kéo tới thời lượng tối thiểu như mọi câu ngắn.

    Mã 3.23.719 (đệm đuôi 0): câu gộp 5,0–5,201 s không bao giờ được xét lại → 0,2 s thay vì 0,5 s.
    """
    events = [_event("XYZUNIQUE", 5.0, 5.1, 2), _event("XYZUNIQUE", 5.2, 5.201, 1)]
    out = absorb_flickers(events, min_duration_sec=0.5, similarity_threshold=0.95, merge_gap_sec=0.1)
    assert len(out) == 1
    assert out[0].duration_sec >= 0.5 - 1e-9
    # Không lấn sang câu kế tiếp.
    events.append(_event("World", 5.3, 5.9, 15))
    out = absorb_flickers(events, min_duration_sec=0.5, similarity_threshold=0.95, merge_gap_sec=0.1)
    assert out[0].end_sec < 5.3
