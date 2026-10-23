"""[v3.23.616] Dựng câu OCR: chữ phồn/giản, câu 2 chữ ổn định, soft-drop lệch, rác số.

Dữ liệu là OCR thật của phim "天骄女友退婚" tập 1 (seraw 14/10, PP-OCRv6, ROI 601x122):
chữ, điểm tin cậy và toạ độ từng khung chép nguyên văn. Cả hai lần trích (15:50 và 21:48)
người dùng phải tự thêm lại câu "瑶瑶" ở 1:34.6 và tự xoá câu "00" trong trình sửa; câu
"瑶瑶" thứ hai bắt đầu trễ 200ms so với lúc chữ hiện ra (đo bằng điểm ảnh).
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
    extend_group_timing_from_soft_drops,
)
from subtitles_extractor.application.services.subtitle_pipeline.event_refinement import (
    refine_final_events,
)
from subtitles_extractor.application.services.subtitle_pipeline.frame_group import (
    FrameGroup,
)
from subtitles_extractor.application.services.text_similarity import text_similarity
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

STEP = 0.04
ROI = Roi(x=54, y=807, width=601, height=122)
NAME_BOX = (227, 30, 347, 96)

#: (mốc, chữ, conf, bbox) — nguyên văn seraw 3eb62b6f, 94.60-95.04 và 97.08-97.64.
REAL_FRAMES: list[tuple[float, str, float, tuple[int, int, int, int]]] = [
    (94.60, "瑶瑶", 0.632, (227, 30, 347, 96)),
    (94.64, "瑶瑶", 0.632, (227, 30, 347, 96)),
    (94.68, "瑶瑶", 0.734, (227, 31, 346, 96)),
    (94.72, "瑶瑶", 0.620, (227, 30, 347, 96)),
    (94.76, "瑶瑶", 0.633, (227, 30, 348, 97)),
    (94.80, "瑶瑶", 0.657, (227, 30, 348, 97)),
    (94.84, "瑶瑶", 0.657, (227, 30, 348, 97)),
    (94.88, "瑶瑶", 0.613, (227, 30, 348, 97)),
    (94.92, "瑶瑶", 0.592, (227, 30, 348, 96)),
    (94.96, "瑤瑤", 0.563, (227, 30, 348, 96)),
    (95.00, "瑤瑶", 0.578, (227, 30, 347, 96)),
    (95.04, "瑤瑶", 0.578, (227, 30, 347, 96)),
    (97.08, "瑤瑤", 0.667, (227, 26, 346, 98)),
    (97.12, "瑤瑶", 0.576, (226, 26, 347, 98)),
    (97.16, "瑤瑤", 0.722, (227, 30, 347, 97)),
    (97.20, "瑶瑤", 0.554, (227, 30, 346, 97)),
    (97.24, "瑶瑤", 0.554, (227, 30, 346, 97)),
    (97.28, "瑶瑶", 0.873, (227, 30, 345, 96)),
    (97.32, "瑶瑤", 0.535, (227, 31, 346, 96)),
    (97.36, "瑶瑶", 0.856, (227, 30, 346, 96)),
    (97.40, "瑶瑶", 0.640, (227, 30, 346, 96)),
    (97.44, "瑶瑶", 0.640, (227, 30, 346, 96)),
    (97.48, "瑶瑶", 0.531, (227, 30, 346, 96)),
    (97.52, "瑶瑶", 0.837, (227, 30, 346, 96)),
    (97.56, "瑶瑶", 0.811, (227, 30, 346, 96)),
    (97.60, "瑤瑤", 0.730, (227, 29, 346, 97)),
    # Khung kế tiếp: chữ đã tắt, OCR đọc một nét áo ở góc phải trên thành "山".
    (97.64, "山", 0.606, (547, 0, 566, 22)),
]

#: Câu thoại rõ nét quanh đó (conf ~1.0) — để phim có đủ câu thật như dữ liệu gốc.
CONTEXT_LINES: list[tuple[float, float, str, tuple[int, int, int, int]]] = [
    (87.92, 89.00, "师兄", (228, 27, 346, 97)),
    (98.00, 98.72, "你出关了", (175, 28, 400, 97)),
    (99.20, 99.88, "感觉怎么样", (148, 29, 455, 97)),
]


def _box(text: str, conf: float, bbox: tuple[int, int, int, int]) -> OcrTextBox:
    x1, y1, x2, y2 = bbox
    return OcrTextBox(
        text=text,
        confidence=Confidence(conf),
        polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
    )


def _frames() -> list[OcrFrameResult]:
    by_ts: dict[float, OcrTextBox] = {
        round(ts, 2): _box(text, conf, bbox) for ts, text, conf, bbox in REAL_FRAMES
    }
    for start, end, text, bbox in CONTEXT_LINES:
        ts = start
        while ts <= end + 1e-9:
            by_ts[round(ts, 2)] = _box(text, 0.999, bbox)
            ts += STEP
    frames: list[OcrFrameResult] = []
    for index in range(round(87.0 / STEP), round(100.5 / STEP)):
        ts = round(index * STEP, 2)
        box = by_ts.get(ts)
        frames.append(
            OcrFrameResult(
                frame_index=index, timestamp_sec=ts, text_boxes=[box] if box else []
            )
        )
    return frames


def _build() -> list[SubtitleEvent]:
    config = SubtitleBuilderConfig(sample_step_sec=STEP, temporal_padding_sec=0.0)
    return SubtitleBuilder(config=config).build(_frames(), roi=ROI)


def _name_calls(events: list[SubtitleEvent]) -> list[SubtitleEvent]:
    return [event for event in events if event.text in {"瑶瑶", "瑤瑤", "瑤瑶", "瑶瑤"}]


class TestVariantSimilarity:
    def test_traditional_and_simplified_reading_are_the_same_line(self) -> None:
        assert text_similarity("瑶瑶", "瑤瑤") >= 0.9
        assert text_similarity("瑶瑤", "瑶瑶") >= 0.9

    def test_different_names_stay_different(self) -> None:
        assert text_similarity("瑶瑶", "楚瑶") < 0.75


class TestRealNameCalls:
    def test_both_name_calls_survive(self) -> None:
        calls = _name_calls(_build())
        assert len(calls) == 2

    def test_first_call_is_kept_with_pixel_timing(self) -> None:
        first = _name_calls(_build())[0]
        assert first.start_sec == pytest.approx(94.60)
        assert first.end_sec == pytest.approx(95.08)  # khung cuối 95.04 + 40ms giữ khung

    def test_second_call_starts_when_the_text_appears(self) -> None:
        second = _name_calls(_build())[1]
        assert second.start_sec == pytest.approx(97.08)  # trước đây 97.28

    def test_far_away_garbage_does_not_stretch_the_end(self) -> None:
        second = _name_calls(_build())[1]
        assert second.end_sec == pytest.approx(97.64)  # không phải 97.68


class TestTwoCharThreshold:
    def _single(self, frame_count: int, conf: float) -> list[SubtitleEvent]:
        frames = [
            OcrFrameResult(
                frame_index=i,
                timestamp_sec=round(10.0 + i * STEP, 2),
                text_boxes=[_box("瑶瑶", conf, NAME_BOX)] if i < frame_count else [],
            )
            for i in range(frame_count + 5)
        ]
        config = SubtitleBuilderConfig(sample_step_sec=STEP, temporal_padding_sec=0.0)
        return SubtitleBuilder(config=config).build(frames, roi=ROI)

    def test_stable_low_confidence_pair_is_kept(self) -> None:
        assert [event.text for event in self._single(12, 0.62)] == ["瑶瑶"]

    def test_short_low_confidence_pair_is_still_dropped(self) -> None:
        assert self._single(3, 0.62) == []

    def test_stable_but_very_low_confidence_pair_is_dropped(self) -> None:
        assert self._single(12, 0.50) == []


class TestSoftDropPosition:
    def _group(self) -> FrameGroup:
        return FrameGroup(
            reconstructed_text="瑶瑶",
            start_timestamp_sec=97.08,
            end_timestamp_sec=97.60,
            accumulated_confidence=10.0,
            total_frames_count=14,
            aggregated_bounding_box=(225, 26, 348, 98),
        )

    def test_drop_elsewhere_on_screen_is_ignored(self) -> None:
        group = self._group()
        extend_group_timing_from_soft_drops(
            [group], [97.64], STEP, {97.64: [(547, 0, 566, 22)]}
        )
        assert group.end_timestamp_sec == pytest.approx(97.60)

    def test_fading_frame_of_the_same_line_still_extends(self) -> None:
        group = self._group()
        extend_group_timing_from_soft_drops(
            [group], [97.64], STEP, {97.64: [(230, 30, 340, 95)]}
        )
        assert group.end_timestamp_sec == pytest.approx(97.64)

    def test_unknown_position_keeps_old_behaviour(self) -> None:
        group = self._group()
        extend_group_timing_from_soft_drops([group], [97.64], STEP)
        assert group.end_timestamp_sec == pytest.approx(97.64)


def _event(index: int, start: float, text: str, conf: float) -> SubtitleEvent:
    return SubtitleEvent(
        index=index,
        interval=TimeInterval(start_sec=start, end_sec=start + 0.32),
        text=text,
        confidence=Confidence(conf),
        frame_count=8,
    )


CJK_LINES = (
    "修炼一途", "自炼气始", "经筑基", "金丹", "元婴", "化神", "合体", "终至大乘",
)


class TestDigitGarbage:

    def _track(self, extra: SubtitleEvent) -> list[SubtitleEvent]:
        events = [
            _event(i + 1, float(i), text, 0.99) for i, text in enumerate(CJK_LINES)
        ]
        return [*events, extra]

    def test_real_belt_ring_reading_is_dropped(self) -> None:
        refined = refine_final_events(self._track(_event(9, 45.0, "00", 0.703)))
        assert "00" not in [event.text for event in refined]

    def test_dotted_fragment_is_dropped(self) -> None:
        refined = refine_final_events(self._track(_event(9, 53.64, "0..", 0.83)))
        assert "0.." not in [event.text for event in refined]

    def test_clearly_read_number_is_kept(self) -> None:
        refined = refine_final_events(self._track(_event(9, 60.0, "2024", 0.97)))
        assert "2024" in [event.text for event in refined]

    def test_numbers_in_a_latin_track_are_untouched(self) -> None:
        events = [_event(i + 1, float(i), f"Line {i}", 0.99) for i in range(8)]
        events.append(_event(9, 45.0, "00", 0.70))
        assert "00" in [event.text for event in refine_final_events(events)]
