"""[v3.23.719] Mảnh đã đệm tới thời lượng tối thiểu, gộp qua câu rác bị NHẢY QUA, không được kéo dài câu gộp.

Thử sức "tall" trên OCR thật 豪门Z外 (936,48 s): "啊" đọc xen "阿", khung cuối "啊" (1 khung, 937,148 s) bị
``absorb_flickers`` đệm tới 0,15 s (937,298 s) dù khung kế tiếp 937,19 s TRỐNG. ``post_merge_duplicates`` gộp nó vào
câu "啊" trước theo nhánh nhảy qua câu ngắn "阿" — nhánh này không bỏ phần đệm như nhánh gộp trực tiếp (v3.23.715) →
câu kết thúc 937,34 s thay vì 937,19 s (lấn 0,15 s sang khoảng đã thấy trống).
"""

from __future__ import annotations

from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
    post_merge_duplicates,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

STEP = 0.0416667
#: Mốc khung CÓ chữ (đọc thật 936,481 → 937,148 s, mỗi khung 1/24 s), khung 937,19 s trở đi trống.
TEXTFUL = [round(936.481 + i * STEP, 3) for i in range(17)]


def _event(index: int, text: str, start: float, end: float, conf: float, frames: int) -> SubtitleEvent:
    return SubtitleEvent(index=index, interval=TimeInterval(start, end), text=text,
                         confidence=Confidence(conf), frame_count=frames)


def _config() -> SubtitleBuilderConfig:
    return SubtitleBuilderConfig(sample_step_sec=0.04, merge_gap_sec=0.6, min_duration_sec=0.15)


def _real_sequence() -> list[SubtitleEvent]:
    """Đúng các câu ra khỏi ``filter_short_duration_events`` ở lượt thử sức (đã truy bằng mã 3.23.718)."""
    return [
        _event(1, "啊", 936.481, 936.730, 0.89, 6),
        _event(2, "啊", 936.773, 937.064, 0.79, 7),
        _event(3, "阿", 937.065, 937.147, 0.80, 2),
        _event(4, "啊", 937.148, 937.298, 1.00, 1),   # 1 khung, đã đệm tới 0,15 s
    ]


def test_skip_merge_drops_minimum_duration_padding_of_the_fragment() -> None:
    out = post_merge_duplicates(_real_sequence(), _config(), TEXTFUL)
    assert [event.text for event in out] == ["啊"]
    # Khung có chữ cuối cùng là 937,148 s; mã cũ giữ mốc đệm 937,298 s (cộng độ giữ khung ra 937,34 s).
    assert abs(out[0].end_sec - 937.148) < 1e-6
    assert abs(out[0].start_sec - 936.481) < 1e-6


def test_without_frame_evidence_the_skip_merge_keeps_the_old_end() -> None:
    """Không có danh sách khung có chữ (dữ liệu cũ / gọi trực tiếp) thì không đoán: giữ mốc như trước."""
    out = post_merge_duplicates(_real_sequence(), _config(), None)
    assert [event.text for event in out] == ["啊"]
    assert abs(out[0].end_sec - 937.298) < 1e-6


def test_skip_merge_of_a_long_line_is_untouched() -> None:
    """Câu dài (không phải mảnh) gộp qua câu rác: mốc cuối là của chính nó, không bị cắt về khung có chữ."""
    events = [
        _event(1, "你好啊", 10.0, 11.0, 0.95, 25),
        _event(2, "x", 11.01, 11.05, 0.5, 1),
        _event(3, "你好啊", 11.06, 12.0, 0.95, 24),
    ]
    textful = [round(10.0 + i * 0.04, 3) for i in range(26)]  # có chữ chỉ tới 11,0 s (giả định ngặt)
    out = post_merge_duplicates(events, _config(), textful)
    assert len(out) == 1
    assert abs(out[0].end_sec - 12.0) < 1e-6
