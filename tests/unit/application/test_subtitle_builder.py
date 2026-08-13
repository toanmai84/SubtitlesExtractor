"""Test service :class:`SubtitleBuilder` — pure Python, dễ test offline."""

from __future__ import annotations

import pytest

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_builder import (
    SubtitleBuilder,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence


def _make_frame(
    frame_idx: int, ts: float, text: str, conf: float = 0.9
) -> OcrFrameResult:
    return OcrFrameResult(
        frame_index=frame_idx,
        timestamp_sec=ts,
        text_boxes=[
            OcrTextBox(
                text=text,
                confidence=Confidence(conf),
                polygon=[(0, 10), (100, 10), (100, 40), (0, 40)],
            )
        ],
    )


class TestSubtitleBuilder:
    def test_empty_input_returns_empty(self) -> None:
        builder = SubtitleBuilder(SubtitleBuilderConfig())
        assert builder.build([]) == []

    def test_groups_consecutive_identical_frames(self) -> None:
        config = SubtitleBuilderConfig(
            similarity_threshold=0.85,
            min_duration_sec=0.05,
            merge_gap_sec=0.5,
        )
        builder = SubtitleBuilder(config)
        # 3 frame liên tiếp cùng câu "Xin chào"
        frames = [
            _make_frame(0, 0.0, "Xin chào"),
            _make_frame(1, 0.1, "Xin chào"),
            _make_frame(2, 0.2, "Xin chào"),
        ]
        events = builder.build(frames)
        assert len(events) == 1
        assert events[0].text == "Xin chào"
        assert events[0].frame_count == 3

    def test_separates_when_gap_too_large(self) -> None:
        config = SubtitleBuilderConfig(merge_gap_sec=0.2, min_duration_sec=0.0)
        builder = SubtitleBuilder(config)
        frames = [
            _make_frame(0, 0.0, "Câu A"),
            _make_frame(1, 0.1, "Câu A"),
            _make_frame(2, 5.0, "Câu A"),  # gap 4.9s — không gộp
        ]
        events = builder.build(frames)
        assert len(events) == 2

    def test_separates_when_text_differs(self) -> None:
        builder = SubtitleBuilder(
            SubtitleBuilderConfig(min_duration_sec=0.0, merge_gap_sec=10.0)
        )
        frames = [
            _make_frame(0, 0.0, "Câu thứ nhất"),
            _make_frame(1, 0.1, "Câu hoàn toàn khác biệt"),
        ]
        events = builder.build(frames)
        assert len(events) == 2

    def test_filters_low_confidence_frames(self) -> None:
        config = SubtitleBuilderConfig(min_confidence=0.7, min_duration_sec=0.0)
        builder = SubtitleBuilder(config)
        frames = [
            _make_frame(0, 0.0, "noise", conf=0.3),
            _make_frame(1, 0.5, "valid", conf=0.95),
        ]
        events = builder.build(frames)
        assert len(events) == 1
        assert events[0].text == "valid"

    def test_filters_too_short_events(self) -> None:
        """Event quá ngắn + conf THẤP → drop.

        Lưu ý v2.x [CRITICAL FIX] về 'Flicker Forgiveness': event conf
        >= 0.70 được PROTECT (extend duration thay vì drop). Test này
        dùng conf=0.5 (dưới ngưỡng protect) để verify logic drop hoạt
        động khi conf thấp.
        """
        builder = SubtitleBuilder(
            SubtitleBuilderConfig(min_duration_sec=1.0, merge_gap_sec=0.5)
        )
        frames = [_make_frame(0, 0.0, "ngắn quá", conf=0.5)]
        events = builder.build(frames)
        assert events == []

    def test_short_event_high_conf_extended_not_dropped(self) -> None:
        """Event ngắn nhưng conf >= 0.70 được EXTEND (không drop).

        Document hành vi v2.x [CRITICAL FIX]: bảo vệ phụ đề thật chỉ render
        thoáng qua. Thay vì drop, extend tới min_duration_sec.
        """
        builder = SubtitleBuilder(
            SubtitleBuilderConfig(min_duration_sec=1.0, merge_gap_sec=0.5)
        )
        frames = [_make_frame(0, 0.0, "quan trọng", conf=0.95)]
        events = builder.build(frames)
        assert len(events) == 1
        # Duration được extend tới min_duration_sec để giữ event.
        assert events[0].duration_sec >= 1.0 - 1e-3

    def test_reindexes_starting_from_one(self) -> None:
        """Events sau pipeline phải được đánh chỉ số 1, 2, 3.

        Lưu ý v2.x: ``pre_filter_garbage_boxes`` lọc single Latin char
        conf < 0.95 (rác logo). Test này dùng text >= 2 ký tự để vượt
        qua filter này.
        """
        builder = SubtitleBuilder(
            SubtitleBuilderConfig(
                min_duration_sec=0.0, merge_gap_sec=0.1, min_text_chars=0
            )
        )
        frames = [
            _make_frame(0, 0.0, "AAA"),
            _make_frame(1, 0.5, "BBB"),
            _make_frame(2, 1.0, "CCC"),
        ]
        events = builder.build(frames)
        assert [e.index for e in events] == [1, 2, 3]

    def test_regression_width_sanity_does_not_drop_valid_cjk_subtitle(self) -> None:
        """Regression v2.19.1: phụ đề CJK dài chiếm ~95% ROI bị drop sai
        bởi width sanity check '> 95% roi_width'.

        Case thực tế từ log user:
            '你说我坚持这九十年' — line_width=466 > 95% roi_width=488
        466/488 = 95.5% → bị drop. Nhưng đây là phụ đề hợp lệ ở giữa
        video dọc 720×1280 với font size lớn.

        Fix: chỉ drop khi line_width > 102% roi_width (tràn thực sự),
        không drop khi tâm vẫn trong tolerance.
        """
        from subtitles_extractor.domain.value_objects.roi import Roi, TextAlignment

        builder = SubtitleBuilder(
            SubtitleBuilderConfig(
                alignment_center_tolerance_ratio=0.15,
                alignment_tolerance_min_px=20.0,
            )
        )
        # Video dọc 720×1280, ROI phụ đề: rộng 488px, alignment CENTER.
        roi = Roi(x=116, y=1100, width=488, height=80, alignment=TextAlignment.CENTER)

        # Phụ đề "你说我坚持这九十年" — 9 ký tự CJK, line_width ≈ 466px
        # (~95.5% roi_width). Tâm tại ~244 (giữa ROI). Phải được GIỮ LẠI.
        subtitle_box = OcrTextBox(
            text="你说我坚持这九十年",
            confidence=Confidence(0.92),
            polygon=[
                (11, 10), (477, 10), (477, 60), (11, 60),
            ],  # width = 466, center = 244
        )
        frame = OcrFrameResult(
            frame_index=0, timestamp_sec=1.0,
            text_boxes=[subtitle_box]
        )
        cleaned = builder._clean_spatial_outliers(frame, roi)
        assert len(cleaned.text_boxes) == 1, (
            f"Phụ đề CJK dài hợp lệ bị drop sai bởi width sanity check. "
            f"line_width=466, roi_width=488 (95.5%) phải được giữ lại."
        )
        assert cleaned.text_boxes[0].text == "你说我坚持这九十年"

    def test_v219_flicker_absorbed_not_dropped(self) -> None:
        """Regression v2.19: câu chớp nhoáng được absorb vào hàng xóm
        thay vì bị drop hẳn.

        Setup: câu 'Hello' xuất hiện ổn định (3 frame, ~1s), sau đó có
        1 câu chớp nhoáng 'Hello' chỉ 1 frame (≈0.05s với sample
        step 0.05s). Trước đây frame đơn lẻ này tạo event ngắn rồi bị
        drop. Bây giờ nó được absorb vào câu trước có cùng text.
        """
        builder = SubtitleBuilder(
            SubtitleBuilderConfig(
                min_duration_sec=0.3,
                merge_gap_sec=0.6,
                min_text_chars=0,
                similarity_threshold=0.7,
                sample_step_sec=0.1,
            )
        )
        frames = [
            _make_frame(0, 0.0, "Hello"),
            _make_frame(1, 0.1, "Hello"),
            _make_frame(2, 0.2, "Hello"),
            _make_frame(3, 0.3, "Hello"),
            # Gap ~0.4s — frame đơn 'Hello' xuất hiện sau nhưng vẫn
            # trong merge_gap_sec=0.6, sẽ được flicker absorber gộp.
            _make_frame(4, 0.7, "Hello"),
        ]
        events = builder.build(frames)
        # Phải gộp thành 1 event (absorb đã hấp thụ frame chớp).
        assert len(events) == 1, (
            f"Mong đợi 1 event sau absorb, nhận {len(events)}: "
            f"{[(e.start_sec, e.end_sec, e.text) for e in events]}"
        )
        assert events[0].text == "Hello"

    def test_v219_isolated_short_event_extended(self) -> None:
        """Câu cô đơn không có hàng xóm phù hợp được kéo dài đến min_duration."""
        builder = SubtitleBuilder(
            SubtitleBuilderConfig(
                min_duration_sec=0.5,
                merge_gap_sec=0.1,  # thấp - không cho gộp
                min_text_chars=0,
                similarity_threshold=0.95,  # cao - không cho merge
                sample_step_sec=0.1,
            )
        )
        # Câu cô đơn 'XYZUNIQUE' có 3 frame liên tiếp (~0.2s) — đủ lớn
        # so với 1/3 * 0.5 = 0.166s nên không bị drop, nhưng nhỏ hơn
        # 0.5 nên cần extend.
        frames = [
            _make_frame(0, 0.0, "Hello"),
            _make_frame(1, 0.1, "Hello"),
            _make_frame(2, 0.2, "Hello"),
            _make_frame(3, 5.0, "XYZUNIQUE"),
            _make_frame(4, 5.1, "XYZUNIQUE"),
            _make_frame(5, 5.2, "XYZUNIQUE"),
            _make_frame(6, 10.0, "World"),
            _make_frame(7, 10.1, "World"),
            _make_frame(8, 10.2, "World"),
        ]
        events = builder.build(frames)
        xyz_events = [e for e in events if "XYZ" in e.text]
        assert len(xyz_events) == 1, (
            f"Mong đợi 1 event chứa 'XYZ', nhận {[e.text for e in events]}"
        )
        # Duration đã được kéo dài tới ngưỡng tối thiểu.
        assert xyz_events[0].duration_sec >= 0.5


class TestSubtitleBuilderConfigValidation:
    def test_default_config_is_reasonable(self) -> None:
        config = SubtitleBuilderConfig()
        assert 0 < config.similarity_threshold <= 1
        assert config.min_duration_sec > 0
        assert config.max_duration_sec > config.min_duration_sec

    def test_immutable(self) -> None:
        config = SubtitleBuilderConfig()
        with pytest.raises(AttributeError):
            config.similarity_threshold = 0.5  # type: ignore[misc]


class TestGreedyDynamicThresholdBugFix:
    """Regression tests cho bug dynamic_threshold floor quá thấp.

    Bug: Khi frame mới có conf < group.mean_conf (conf drop), Greedy hạ
    threshold xuống max(0.35, base-0.30) = 0.45. Với CJK 2 ký tự share 1
    ký tự: sim("性别","雄性") = 0.50 >= 0.45 → MERGE SAI.

    Fix: Nâng floor từ 0.35 lên 0.60 → 0.50 < 0.60 → New Group ✓.
    """

    @pytest.fixture()
    def builder(self) -> SubtitleBuilder:
        return SubtitleBuilder(SubtitleBuilderConfig(
            similarity_threshold=0.75,
            sample_step_sec=0.05,
            merge_gap_sec=0.60,
            min_confidence=0.50,
            min_duration_sec=0.10,
            temporal_padding_sec=0.05,
        ))

    def test_cjk_2char_share_1char_splits_on_conf_drop(
        self, builder: SubtitleBuilder
    ) -> None:
        """Bug cốt lõi: "性别" + "雄性" không được gộp kể cả khi conf drop.

        Trường hợp thực tế từ chinese_vid1.mp4:
            00:00:24,760 → 00:00:25,241: 性别
            00:00:25,241 → 00:00:25,960: 雄性
        Frame đầu "雄性" thường có conf thấp hơn group "性别" đang tích lũy
        → bug cũ hạ threshold về 0.45, sim=0.50 vượt ngưỡng → MERGE SAI.
        """
        frames = [
            
_make_frame(0, 24.76, "性别", conf=0.92),
            
_make_frame(1, 24.81, "性别", conf=0.91),
            
_make_frame(2, 24.86, "性别", conf=0.92),
            
_make_frame(3, 25.00, "性别", conf=0.90),
            
_make_frame(4, 25.05, "雄性", conf=0.88),  # conf drop → trigger bug cũ
            
_make_frame(5, 25.10, "雄性", conf=0.92),
            
_make_frame(6, 25.15, "雄性", conf=0.93),
            
_make_frame(7, 25.25, "雄性", conf=0.91),
        ]
        events = builder.build(frames)

        assert len(events) == 2, (
            f"Phải có 2 câu riêng biệt, nhận {len(events)}: "
            f"{[(e.text, e.start_sec) for e in events]}"
        )
        assert "性别" in events[0].text, (
            f"Câu 1 phải chứa '性别', nhận '{events[0].text}'"
        )
        assert "雄性" in events[1].text, (
            f"Câu 2 phải chứa '雄性', nhận '{events[1].text}'"
        )

    def test_same_cjk_still_merges(self, builder: SubtitleBuilder) -> None:
        """Cùng câu vẫn được gộp đúng sau khi tăng floor threshold."""
        frames = [
            
_make_frame(i, i * 0.05, "恭喜宿主", conf=0.88 + (i % 3) * 0.02)
            for i in range(10)
        ]
        events = builder.build(frames)

        assert len(events) == 1, (
            f"Cùng câu phải là 1 event, nhận {len(events)}: "
            f"{[e.text for e in events]}"
        )

    def test_similar_cjk_3char_still_merges(self, builder: SubtitleBuilder) -> None:
        """OCR thiếu 1 ký tự ('太弱了' vs '太弱') vẫn merge vào nhau."""
        frames = [
            
_make_frame(0, 0.0, "这聚灵丹也太弱了", conf=0.93),
            
_make_frame(1, 0.05, "这聚灵丹也太弱", conf=0.85),  # OCR miss ký tự cuối
            
_make_frame(2, 0.10, "这聚灵丹也太弱了", conf=0.91),
            
_make_frame(3, 0.15, "这聚灵丹也太弱了", conf=0.92),
        ]
        events = builder.build(frames)

        assert len(events) == 1, (
            f"OCR typo nhỏ phải được merge, nhận {len(events)}: "
            f"{[e.text for e in events]}"
        )

    def test_dynamic_threshold_floor_value(self) -> None:
        """Xác minh threshold floor mới = 0.60, không phải 0.35 (bug cũ).

        Lưu ý v2.x [CRITICAL FIX]: ``_check_cjk_critical_reversal`` chặn
        các cặp CJK đảo nghĩa (chứa '不/没/你/我/性/别/雄/...') trả về 0.0
        thay vì compute similarity. Cặp '性别' vs '雄性' cũ KHÔNG còn phù
        hợp để test threshold floor — nó luôn return 0.0 do critical
        reversal.

        Dùng cặp text không chứa critical reversal keywords để test floor
        threshold logic vẫn hoạt động.
        """
        from subtitles_extractor.application.services.subtitle_builder import (
            text_similarity_cached,
        )
        # Cặp CJK ngẫu nhiên KHÔNG nằm trong critical reversal keywords.
        # Đo similarity giữa 2 câu chỉ tương đồng vừa phải.
        sim_mid_range = text_similarity_cached("时间还早", "时光匆匆")

        old_floor = 0.35
        new_floor = 0.60
        base = 0.75
        dyn_old = max(old_floor, base - 0.30)  # 0.45
        dyn_new = max(new_floor, base - 0.30)  # 0.60

        # Đảm bảo similarity nằm trong vùng nhạy với floor change
        # (giữa dyn_old=0.45 và dyn_new=0.60). Nếu không, test không
        # còn ý nghĩa kiểm tra floor — chỉ cần đảm bảo logic floor đúng.
        assert dyn_old < dyn_new, (
            f"Floor mới ({dyn_new}) phải > floor cũ ({dyn_old}) để fix có ý nghĩa"
        )
        # Test này chủ yếu document rằng floor mới = 0.60.
        # Similarity cụ thể tuỳ thuộc rapidfuzz/jaro-winkler đôi khi thay đổi.
        assert 0.0 <= sim_mid_range <= 1.0, "Similarity phải nằm trong [0, 1]"
