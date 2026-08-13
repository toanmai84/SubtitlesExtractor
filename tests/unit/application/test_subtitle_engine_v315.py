"""Test các cải tiến engine dựng phụ đề v3.15 (hiệu chuẩn với ground-truth thực).

Bộ fix rút ra từ phân tích từng lỗi trên `haomen_roi_subtitle.seraw.json` (114k box)
so với `haomen.srt` (2122 câu): F1 0.9965 → 0.9995, CER 0.0068 → 0.0016.
"""

from __future__ import annotations

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
    _should_split_boundary,
    post_merge_duplicates,
    split_stable_variant_groups,
)
from subtitles_extractor.application.services.subtitle_pipeline.event_refinement import (
    drop_flash_fragments,
    drop_pure_latin_watermark_events,
    strip_latin_suffix_from_cjk_events,
)
from subtitles_extractor.application.services.subtitle_pipeline.frame_group import (
    FrameGroup,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


def _event(start: float, end: float, text: str, fc: int = 10) -> SubtitleEvent:
    return SubtitleEvent(
        index=1, text=text,
        interval=TimeInterval(start_sec=start, end_sec=end),
        confidence=0.95, frame_count=fc,
    )


def _group(start: float, end: float, text: str, member_texts=None, fc: int = 10):
    return FrameGroup(
        reconstructed_text=text, start_timestamp_sec=start, end_timestamp_sec=end,
        accumulated_confidence=0.95 * fc, total_frames_count=fc,
        member_texts=member_texts or [],
    )


class TestShouldSplitBoundary:
    def test_progressive_suffix_added(self) -> None:
        # ``不过`` → ``不过什么``: nói thêm chữ CUỐI → 2 câu.
        assert _should_split_boundary("不过", "不过什么")

    def test_progressive_suffix_removed(self) -> None:
        # ``舒服吗`` → ``舒服``: câu hỏi rồi câu đáp → 2 câu.
        assert _should_split_boundary("舒服吗", "舒服")

    def test_latin_delta_is_watermark(self) -> None:
        # ``泡小姑娘`` vs ``泡小姑娘BOR``: đuôi Latin = watermark dính → 1 câu.
        assert not _should_split_boundary("还在公司里泡小姑娘", "还在公司里泡小姑娘BOR")

    def test_single_leading_cjk_is_ocr_drop(self) -> None:
        # ``一群废物`` vs ``群废物``: mất 1 ký tự ĐẦU (nét mảnh) = lỗi OCR → 1 câu.
        assert not _should_split_boundary("一群废物", "群废物")

    def test_space_variant_same(self) -> None:
        assert not _should_split_boundary("阿姨那能一样吗", "阿姨 那能一样吗")

    def test_unrelated_texts_split(self) -> None:
        assert _should_split_boundary("有一个问题", "今天天气很好")


class TestSplitStableVariantGroups:
    @staticmethod
    def _member_texts(spec: list[tuple[str, int, float]]):
        # spec: (text, frames, step) — sinh chuỗi (ts, text).
        out, ts = [], 100.0
        for text, frames, step in spec:
            for _ in range(frames):
                out.append((ts, text))
                ts += step
        return out

    def test_two_stable_utterances_split(self) -> None:
        mt = self._member_texts([("舒服吗", 20, 0.05), ("舒服", 20, 0.05)])
        group = _group(mt[0][0], mt[-1][0], "舒服吗", member_texts=mt, fc=40)
        result = split_stable_variant_groups([group])
        assert [g.reconstructed_text for g in result] == ["舒服吗", "舒服"]

    def test_ocr_noise_runs_absorbed(self) -> None:
        # Run nhiễu 1-2 frame giữa chừng không gây tách.
        mt = self._member_texts(
            [("今天很好", 15, 0.05), ("今天很奷", 2, 0.05), ("今天很好", 15, 0.05)]
        )
        group = _group(mt[0][0], mt[-1][0], "今天很好", member_texts=mt, fc=32)
        result = split_stable_variant_groups([group])
        assert len(result) == 1

    def test_oscillation_not_split(self) -> None:
        # A-B-A khối lớn (OCR dao động) → không tách.
        mt = self._member_texts(
            [("你好世界", 10, 0.05), ("你好世男", 10, 0.05), ("你好世界", 10, 0.05)]
        )
        group = _group(mt[0][0], mt[-1][0], "你好世界", member_texts=mt, fc=30)
        assert len(split_stable_variant_groups([group])) == 1


class TestRepeatUtteranceGuard:
    def test_identical_text_with_long_blank_kept_separate(self) -> None:
        # ``我好热`` lặp 2 lần, biến mất 0.42s giữa chừng → 2 event.
        events = [
            _event(10.00, 11.08, "我好热", fc=26),
            _event(11.50, 12.75, "我好热", fc=30),
        ]
        merged = post_merge_duplicates(events, SubtitleBuilderConfig())
        assert len(merged) == 2

    def test_blink_fragments_still_merged(self) -> None:
        # Mảnh nhấp nháy nhỏ (fc thấp) vẫn được gộp như cũ.
        events = [
            _event(10.00, 10.20, "你好", fc=4),
            _event(10.60, 10.80, "你好", fc=4),
        ]
        merged = post_merge_duplicates(events, SubtitleBuilderConfig())
        assert len(merged) == 1


class TestEventRefinement:
    def test_strip_latin_suffix(self) -> None:
        events = [_event(1, 2, f"这是中文台词第{i}句") for i in range(9)]
        events.append(_event(10, 11, "嗯不对OVONET"))
        refined = strip_latin_suffix_from_cjk_events(events)
        assert refined[-1].text == "嗯不对"

    def test_whitelist_suffix_kept(self) -> None:
        events = [_event(1, 2, f"中文中文 {i}") for i in range(9)]
        events.append(_event(10, 11, "我们去KTV"))
        refined = strip_latin_suffix_from_cjk_events(events)
        assert refined[-1].text == "我们去KTV"

    def test_drop_pure_latin_watermark(self) -> None:
        events = [_event(i, i + 1, f"中文台词{i}") for i in range(9)]
        events.append(_event(20, 21, "ALENCIACA"))
        refined = drop_pure_latin_watermark_events(events)
        assert all(e.text != "ALENCIACA" for e in refined)
        assert len(refined) == 9

    def test_flash_single_cjk_dropped(self) -> None:
        events = [_event(1.0, 1.15, "強", fc=1), _event(2.0, 3.0, "正常的台词")]
        refined = drop_flash_fragments(events)
        assert [e.text for e in refined] == ["正常的台词"]

    def test_flash_echo_dropped(self) -> None:
        events = [
            _event(10.0, 10.09, "那我先回去了", fc=1),
            _event(10.2, 11.1, "那我先回去了", fc=20),
        ]
        refined = drop_flash_fragments(events)
        assert len(refined) == 1 and refined[0].frame_count == 20

    def test_short_but_unique_kept(self) -> None:
        # Mảnh ngắn KHÔNG có bằng chứng rác → giữ (OCR có thể bắt thiếu frame).
        events = [_event(10.0, 10.15, "好的", fc=3), _event(20.0, 21.0, "中文台词")]
        refined = drop_flash_fragments(events)
        assert len(refined) == 2
