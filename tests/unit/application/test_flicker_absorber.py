"""Tests cho :mod:`flicker_absorber` — hấp thụ câu chớp nhoáng."""

from __future__ import annotations

from subtitles_extractor.application.services.flicker_absorber import (
    absorb_flickers,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


def _make_event(
    text: str, start: float, end: float, conf: float = 0.9, uid: str | None = None
) -> SubtitleEvent:
    """Tạo SubtitleEvent với UID có thể chỉ định."""
    kwargs: dict = {
        "index": 1,
        "text": text,
        "interval": TimeInterval(start, end),
        "confidence": Confidence(conf),
        "frame_count": 3,
    }
    if uid is not None:
        kwargs["uid"] = uid
    return SubtitleEvent(**kwargs)


class TestAbsorbFlickers:
    def test_long_events_unchanged(self) -> None:
        events = [
            _make_event("Hello", 0.0, 1.0, uid="a"),
            _make_event("World", 1.5, 2.5, uid="b"),
        ]
        result = absorb_flickers(
            events,
            min_duration_sec=0.5,
            similarity_threshold=0.7,
            merge_gap_sec=0.6,
        )
        assert len(result) == 2

    def test_flicker_absorbed_into_similar_prev(self) -> None:
        # 'Hello' (1.0s) → 'Helo' (0.1s flicker, same text) → 'World' (1.0s).
        events = [
            _make_event("Hello", 0.0, 1.0, uid="a"),
            _make_event("Helo", 1.0, 1.1, uid="b"),  # flicker, similar to prev
            _make_event("World", 1.6, 2.6, uid="c"),
        ]
        result = absorb_flickers(
            events,
            min_duration_sec=0.5,
            similarity_threshold=0.7,
            merge_gap_sec=0.6,
        )
        # Flicker phải bị gộp vào 'Hello' → còn 2 event.
        assert len(result) == 2
        assert result[0].text == "Hello"
        # End đẩy ra tới 1.1.
        assert result[0].end_sec == 1.1

    def test_flicker_absorbed_into_similar_next(self) -> None:
        # Khác case: flicker giống next, không giống prev.
        events = [
            _make_event("Hello", 0.0, 1.0, uid="a"),
            _make_event("Worl", 1.4, 1.5, uid="b"),  # flicker similar to next
            _make_event("World", 1.6, 2.6, uid="c"),
        ]
        result = absorb_flickers(
            events,
            min_duration_sec=0.3,
            similarity_threshold=0.7,
            merge_gap_sec=0.6,
        )
        # Flicker phải bị gộp với 'World' → còn 2 event.
        assert len(result) == 2
        assert result[0].text == "Hello"
        assert result[1].text == "World"
        # Start của World có thể bị đẩy về sớm hơn (để bao trọn flicker).
        assert result[1].start_sec <= 1.5

    def test_very_short_flicker_no_neighbor_dropped(self) -> None:
        """Câu flicker < min/5 conf THẤP và không absorb được → DROP.

        Lưu ý v2.x [CRITICAL FIX] về 'Flicker Forgiveness': flicker conf
        >= 0.70 được PROTECT (không drop) để tránh mất phụ đề thật do OCR
        nhanh. Test này dùng conf=0.50 (dưới ngưỡng protect) để verify
        drop logic vẫn hoạt động cho flicker conf thấp.
        """
        events = [
            _make_event("Hello", 0.0, 1.0, uid="a"),
            # dur=0.04s < min/5 (0.3/5=0.06) + conf=0.5 → DROP (no protect)
            _make_event("XYZ", 5.0, 5.04, conf=0.5, uid="b"),
            _make_event("World", 10.0, 11.0, uid="c"),
        ]
        result = absorb_flickers(
            events,
            min_duration_sec=0.3,
            similarity_threshold=0.7,
            merge_gap_sec=0.6,
        )
        assert len(result) == 2
        assert result[0].text == "Hello"
        assert result[1].text == "World"

    def test_very_short_flicker_high_conf_protected(self) -> None:
        """Câu flicker conf >= 0.70 được PROTECT (extend) thay vì drop.

        Document hành vi v2.x [CRITICAL FIX] — flicker high-conf không bị
        drop ngay cả khi không có neighbor để absorb. Lý do: bảo vệ phụ
        đề thật chỉ render thoáng qua (vd 1-2 frame ở chuyển cảnh nhanh).
        """
        events = [
            _make_event("Hello", 0.0, 1.0, uid="a"),
            # dur=0.04s nhưng conf=0.90 → PROTECTED (extend_duration)
            _make_event("XYZ", 5.0, 5.04, conf=0.90, uid="b"),
            _make_event("World", 10.0, 11.0, uid="c"),
        ]
        result = absorb_flickers(
            events,
            min_duration_sec=0.3,
            similarity_threshold=0.7,
            merge_gap_sec=0.6,
        )
        # 'XYZ' được giữ lại (extend hoặc keep) do high-conf protection.
        assert len(result) == 3
        assert {e.text for e in result} == {"Hello", "XYZ", "World"}

    def test_junk_zero_duration_low_conf_dropped(self) -> None:
        """Câu dur=0.0s + conf thấp là rác OCR tuyệt đối → DROP.

        Như note v2.x: dur=0 + conf < 0.70 (không protect) → drop ngay.
        Hành vi v2.x đã thay đổi để protect rác có conf cao (tránh mất
        phụ đề thật), nên test cần conf thấp để verify drop logic.
        """
        events = [
            _make_event("了", 1.0, 1.0, conf=0.5, uid="junk"),  # dur=0.0
            _make_event("Hello", 2.0, 3.0, uid="ok"),
        ]
        result = absorb_flickers(events, 0.3, 0.7, 0.6)
        assert len(result) == 1
        assert result[0].text == "Hello"

    def test_cjk_traditional_simplified_absorbed(self) -> None:
        """'王建強' (Traditional) absorb vào '王建强' (Simplified).

        fuzz.ratio ≈ 67% < default threshold 70%. Dùng CJK short threshold
        0.55 → absorb được.
        """
        events = [
            _make_event("王建强", 23.920, 24.278, uid="main"),   # Simplified (dur=0.358 ≥ min)
            _make_event("王建強", 24.278, 24.518, uid="flicker"),  # Traditional (dur=0.240 < min)
        ]
        result = absorb_flickers(events, 0.3, 0.7, 0.6)
        assert len(result) == 1, f"Traditional/Simplified phải absorb, nhận {[e.text for e in result]}"
        assert result[0].text == "王建强"

    def test_valid_cjk_sentence_kept_not_dropped(self) -> None:
        """'自然会受到一些' dur=0.600s phải được GIỮ LẠI.

        Regression v2.19: estimate_min_reading_duration cho câu 7 CJK
        = 2.0s → very_short = 0.667s > 0.600s → DROP sai.
        Fix: không dùng estimate, chỉ dùng min_duration_sec config.
        """
        events = [
            _make_event("行走在宗门内时", 136.0, 137.8, uid="a"),
            _make_event("自然会受到一些", 137.8, 138.4, uid="b"),  # dur=0.600s hợp lệ
            _make_event("异样目光",       138.4, 139.5, uid="c"),
        ]
        result = absorb_flickers(events, 0.3, 0.7, 0.6)
        assert any(e.text == "自然会受到一些" for e in result), (
            "Câu hợp lệ '自然会受到一些' (dur=0.6s) bị drop sai!"
        )

    def test_medium_flicker_no_neighbor_extended(self) -> None:
        """Câu ≥ min/5 và không absorb được → KÉO DÀI tới min_duration."""
        events = [
            _make_event("Hello", 0.0, 1.0, uid="a"),
            # dur=0.12s > min/5 (0.5/5=0.1) → extend
            _make_event("XYZ", 5.0, 5.12, uid="b"),
            _make_event("World", 10.0, 11.0, uid="c"),
        ]
        result = absorb_flickers(
            events,
            min_duration_sec=0.5,
            similarity_threshold=0.7,
            merge_gap_sec=0.6,
        )
        assert len(result) == 3
        flicker = result[1]
        assert flicker.text == "XYZ"
        assert flicker.duration_sec >= 0.5

    def test_extension_does_not_overlap_next(self) -> None:
        """Kéo dài flicker tới min_duration mà KHÔNG lấn next event.

        Setup: flicker 0.4s (> 1/3 * 1.0 = 0.33s nên không bị drop),
        cô đơn (xa cả 2 hàng xóm) → kéo dài, nhưng bị giới hạn bởi
        start của next event.
        """
        events = [
            _make_event("Hello", 0.0, 1.0, uid="a"),
            _make_event("XYZ", 5.0, 5.4, uid="b"),  # 0.4s, isolated
            _make_event("World", 5.7, 6.6, uid="c"),  # cách XYZ 0.3s
        ]
        result = absorb_flickers(
            events,
            min_duration_sec=1.0,
            similarity_threshold=0.95,  # cao → không gộp
            merge_gap_sec=0.1,           # thấp → không gộp
        )
        # XYZ phải có trong result (đã extend, không drop).
        assert any(e.text == "XYZ" for e in result), (
            f"XYZ phải được extend, nhận: {[e.text for e in result]}"
        )
        flicker = next(e for e in result if e.text == "XYZ")
        # End không lấn 'World' (start=5.7).
        assert flicker.end_sec < 5.7

    def test_chooses_better_neighbor(self) -> None:
        # Flicker giống cả 2 hàng xóm; chọn bên có similarity cao hơn.
        events = [
            _make_event("Hello world", 0.0, 1.0, uid="a"),
            _make_event("Hello world!", 1.05, 1.15, uid="b"),  # gần như identical với prev
            _make_event("Hello", 1.6, 2.6, uid="c"),  # khác hơn với flicker
        ]
        result = absorb_flickers(
            events,
            min_duration_sec=0.5,
            similarity_threshold=0.5,
            merge_gap_sec=0.6,
        )
        # Flicker phải gộp về prev (similarity cao hơn).
        assert len(result) == 2
        assert result[0].text == "Hello world"
        # End đẩy ra tới 1.15.
        assert result[0].end_sec >= 1.15
        # 'Hello' (cuối) giữ nguyên.
        assert result[1].text == "Hello"

    def test_empty_input(self) -> None:
        assert absorb_flickers([], 0.5, 0.7, 0.6) == []

    def test_min_duration_zero_disables(self) -> None:
        # min_duration=0 → không lọc gì (return as-is).
        events = [_make_event("a", 0.0, 0.01, uid="x")]
        result = absorb_flickers(events, 0.0, 0.7, 0.6)
        assert len(result) == 1
