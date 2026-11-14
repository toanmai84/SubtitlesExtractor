"""[v3.23.699] Dựng phụ đề: chọn chữ theo tổng số khung khi gộp, chữ lặp do OCR đọc thừa, mảnh rác OCR.

Ba lỗi tìm ra khi dựng lại phụ đề từ OCR thô của lượt trích xuất thật (phim 86 tập, GPU, bước 0,03 s).
"""

from __future__ import annotations

from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
    drop_mixed_noise_fragments,
    merge_adjacent_duplicates,
    post_merge_duplicates,
)
from subtitles_extractor.application.services.subtitle_pipeline.frame_group import FrameGroup
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


def _group(text: str, start: float, frames: int, conf: float = 0.95, step: float = 0.03) -> FrameGroup:
    return FrameGroup(
        reconstructed_text=text, start_timestamp_sec=start, end_timestamp_sec=start + (frames - 1) * step,
        accumulated_confidence=conf * frames, total_frames_count=frames,
    )


def _event(index: int, text: str, start: float, end: float, conf: float = 0.95, frames: int = 10) -> SubtitleEvent:
    return SubtitleEvent(index=index, interval=TimeInterval(start, end), text=text,
                         confidence=Confidence(conf), frame_count=frames)


class TestMergeTextChoice:
    def test_majority_text_wins_across_three_merges(self) -> None:
        """Ca thật: "十不存一"(7 khung) / "十不存"(8) / "十不存一"(14) — bản cũ ra "十不存" dù "十不存一" có 21/29 khung."""
        groups = [_group("十不存一", 10.0, 7), _group("十不存", 10.21, 8), _group("十不存一", 10.45, 14)]
        merged = merge_adjacent_duplicates(groups, SubtitleBuilderConfig())
        assert len(merged) == 1
        assert merged[0].reconstructed_text == "十不存一"
        assert merged[0].total_frames_count == 29

    def test_two_groups_keep_the_heavier_text(self) -> None:
        merged = merge_adjacent_duplicates(
            [_group("十不存", 10.0, 3), _group("十不存一", 10.09, 12)], SubtitleBuilderConfig())
        assert [g.reconstructed_text for g in merged] == ["十不存一"]


class TestStutterSuperset:
    def test_doubled_char_from_a_short_read_does_not_win(self) -> None:
        """Ca thật: "别闹" đa số khung, 1–2 khung đọc thừa "别别闹" — bản cũ luôn giữ bản dài."""
        events = [_event(1, "别闹", 5.0, 6.0, frames=30), _event(2, "别别闹", 6.02, 6.1, frames=2)]
        out = post_merge_duplicates(events, SubtitleBuilderConfig())
        assert [e.text for e in out] == ["别闹"]

    def test_real_longer_read_still_wins(self) -> None:
        """Thêm chữ KHÔNG lặp (đọc sót ở bản ngắn) vẫn giữ bản dài như trước."""
        events = [_event(1, "师叔", 5.0, 6.0, frames=30), _event(2, "小师叔", 6.02, 6.1, frames=2)]
        out = post_merge_duplicates(events, SubtitleBuilderConfig())
        assert [e.text for e in out] == ["小师叔"]


class TestNoiseFragments:
    def _film(self, extra: list[SubtitleEvent]) -> list[SubtitleEvent]:
        base = [_event(i, f"第{i}句台词", i * 2.0, i * 2.0 + 1) for i in range(1, 20)]
        return base + extra

    def test_drops_low_confidence_mixed_fragment_in_cjk_film(self) -> None:
        noise = _event(50, "色6%", 100.0, 100.5, conf=0.6)
        kept, dropped = drop_mixed_noise_fragments(self._film([noise]))
        assert dropped == 1 and noise not in kept

    def test_keeps_confident_or_cjk_lines(self) -> None:
        extra = [_event(50, "色6%", 100.0, 100.5, conf=0.9),       # tin cậy cao
                 _event(51, "啊？", 102.0, 102.5, conf=0.4),       # chỉ chữ Hán + dấu câu
                 _event(52, "VIP会员", 104.0, 104.5, conf=0.5)]    # ≥ 2 chữ Hán
        kept, dropped = drop_mixed_noise_fragments(self._film(extra))
        assert dropped == 0 and len(kept) == 22

    def test_latin_film_is_untouched(self) -> None:
        events = [_event(i, "ok go", i * 2.0, i * 2.0 + 1, conf=0.5) for i in range(1, 20)]
        assert drop_mixed_noise_fragments(events) == (events, 0)
        assert drop_mixed_noise_fragments([]) == ([], 0)
