"""[v3.23.615] Mốc kết thúc câu phụ đề = lúc chữ BIẾN MẤT, không phải mốc khung cuối.

Đo bằng điểm ảnh vùng phụ đề trên video thật 14/10 (phim "天骄女友退婚" tập 1, 25fps): với
mọi khung được lấy mẫu, 84/94 câu kết thúc sớm đúng 40ms (một khung) và giữa hai câu liền
nhau luôn có khe 40ms không có thật (``00:00:02,800 --> ...`` rồi ``00:00:02,840``). Khung
ở mốc ``t`` còn đứng trên màn hình tới ``t + 1/fps``. Sau sửa: 83/94 câu đúng mốc.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
    frame_display_hold,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence

STEP = 0.04


def _frame(index: int, text: str) -> OcrFrameResult:
    boxes = [] if not text else [
        OcrTextBox(
            text=text,
            confidence=Confidence(0.99),
            polygon=[(228, 26), (349, 29), (348, 100), (227, 97)],
        )
    ]
    return OcrFrameResult(
        frame_index=index, timestamp_sec=round(index * STEP, 3), text_boxes=boxes
    )


def _real_opening() -> list[OcrFrameResult]:
    """Bốn câu mở đầu thật (1.56s tới 5.80s): ba câu nối liền, một câu sau khe trống."""
    script = (
        [""] * 39 + ["修炼一途"] * 32 + ["自炼气始"] * 33 + ["经筑基"] * 18
        + [""] * 4 + ["金丹"] * 19 + [""] * 3
    )
    return [_frame(index, text) for index, text in enumerate(script)]


@pytest.fixture(scope="module")
def events():
    config = SubtitleBuilderConfig(sample_step_sec=STEP, temporal_padding_sec=0.0)
    return SubtitleBuilder(config).build(_real_opening())


def test_back_to_back_lines_touch_instead_of_leaving_a_fake_gap(events) -> None:
    assert [event.text for event in events] == ["修炼一途", "自炼气始", "经筑基", "金丹"]
    assert events[0].end_sec == pytest.approx(events[1].start_sec - 0.001)
    assert events[1].end_sec == pytest.approx(events[2].start_sec - 0.001)


def test_line_followed_by_blank_ends_when_text_disappears(events) -> None:
    last_visible = (39 + 32 + 33 + 18 - 1) * STEP
    assert events[2].end_sec == pytest.approx(last_visible + STEP)


def test_hold_is_one_interval_for_frame_level_sampling() -> None:
    assert frame_display_hold([0.0, 0.04, 0.08, 0.16]) == pytest.approx(0.04)


def test_hold_is_half_interval_for_sparse_sampling() -> None:
    assert frame_display_hold([0.0, 0.25, 0.5, 0.75]) == pytest.approx(0.125)


def test_no_hold_without_enough_frames() -> None:
    assert frame_display_hold([1.0]) == 0.0


def test_reocr_event_does_not_spill_past_the_scanned_range() -> None:
    from subtitles_extractor.application.use_cases.reocr import _clip_events_to_range
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

    inside = SubtitleEvent(1, "金丹", TimeInterval(start_sec=5.04, end_sec=5.80))
    spilling = SubtitleEvent(2, "元婴", TimeInterval(start_sec=5.84, end_sec=6.04))
    clipped = _clip_events_to_range([inside, spilling], range_end_sec=6.0)
    assert [(e.start_sec, e.end_sec) for e in clipped] == [(5.04, 5.80), (5.84, 6.0)]
