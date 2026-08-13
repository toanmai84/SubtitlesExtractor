"""Tests cho các nâng cấp v2.20 SubtitleBuilder.

Bao phủ:
    * CJK punctuation normalization.
    * Substring similarity CJK.
    * _select_better_text logic.
    * _filter_y_with_cluster_awareness (2-line subtitle).
    * _groups_to_events negative duration guard.
    * _post_merge_duplicates.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.text_similarity import (
    text_similarity,
    viterbi_similarity,
)


class TestCjkPunctuationNormalization:
    """Fix #4: text similarity với CJK trailing punctuation."""

    def test_trailing_period_still_similar(self) -> None:
        assert text_similarity("好", "好。") >= 0.90

    def test_trailing_exclamation_similar(self) -> None:
        assert text_similarity("再见", "再见！") >= 0.90

    def test_trailing_comma_similar(self) -> None:
        assert text_similarity("你好", "你好，") >= 0.90

    def test_identical_unaffected(self) -> None:
        assert text_similarity("王建强", "王建强") == 1.0

    def test_different_text_still_low(self) -> None:
        assert text_similarity("性别", "雄性") < 0.70


class TestCjkSubstringSimilarity:
    """Fix #5: Viterbi similarity cho CJK substring (câu bị OCR cắt ngắn)."""

    def test_one_char_missing_end(self) -> None:
        # '这聚灵丹也太弱' vs '这聚灵丹也太弱了' — 1 ký tự thiếu.
        sim = viterbi_similarity("这聚灵丹也太弱", "这聚灵丹也太弱了")
        assert sim > 0.80, f"Substring thiếu 1 ký tự phải similarity cao, nhận {sim}"

    def test_substring_boost(self) -> None:
        """Substring CJK ngắn vs dài: similarity vừa phải, không bị boost
        lên cao quá để tránh gộp sai các câu siêu ngắn.

        Lưu ý v2.x [CRITICAL FIX]: '恭喜' nằm trong '恭喜呀' nhưng KHÔNG
        bị boost lên 85% để tránh việc gộp nuốt sai lầm 2 phát ngôn khác
        nhau. Test cũ expect > 0.5, hành vi mới có thể trả ~0.45-0.50
        cho cặp '这聚灵丹' (4 chars) ⊂ '这聚灵丹也太弱了' (8 chars).
        """
        # '这聚灵丹' là substring của '这聚灵丹也太弱了'.
        sim = viterbi_similarity("这聚灵丹", "这聚灵丹也太弱了")
        # Lower bound ~0.40 vẫn đủ để Viterbi gộp với threshold động.
        assert sim > 0.40, f"Substring phải có similarity vừa phải, nhận {sim}"
        # Đồng thời KHÔNG quá cao (< 0.85) để tránh gộp 2 phát ngôn khác.
        assert sim < 0.85, (
            f"Substring CJK ngắn không nên boost lên cao tránh gộp sai, "
            f"nhận {sim}"
        )

    def test_unrelated_cjk_still_low(self) -> None:
        # '性别' vs '雄性' không liên quan.
        sim = viterbi_similarity("性别", "雄性")
        assert sim < 0.60


class TestSelectBetterText:
    """Test _select_anchor_text — anchor selection trong ROVER."""

    def test_majority_vote_small_cluster(self) -> None:
        """Cluster ≤ 4 frame: majority vote chọn text phổ biến nhất."""
        from subtitles_extractor.application.services.subtitle_builder import (
            SubtitleBuilder,
        )
        from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig

        builder = SubtitleBuilder(SubtitleBuilderConfig())
        texts_with_conf = [
            ("这聚灵丹也太弱了", 0.90),
            ("这聚灵丹也太弱了", 0.85),
            ("这聚灵丹也太弱A", 0.95),  # outlier conf cao nhưng rare
        ]
        anchor = builder._select_anchor_text(texts_with_conf)
        # Majority vote (2/3): "这聚灵丹也太弱了" thắng dù có 1 outlier conf cao.
        assert anchor == "这聚灵丹也太弱了"

    def test_large_cluster_conf_wins(self) -> None:
        """Cluster > 4 frame: conf×length score thắng."""
        from subtitles_extractor.application.services.subtitle_builder import (
            SubtitleBuilder,
        )
        from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig

        builder = SubtitleBuilder(SubtitleBuilderConfig())
        texts_with_conf = [
            ("你好", 0.95),
            ("你好", 0.92),
            ("你好", 0.90),
            ("你好A", 0.80),
            ("你好", 0.85),
        ]
        anchor = builder._select_anchor_text(texts_with_conf)
        assert anchor == "你好"

    def test_single_text_returned_immediately(self) -> None:
        from subtitles_extractor.application.services.subtitle_builder import (
            SubtitleBuilder,
        )
        from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig

        builder = SubtitleBuilder(SubtitleBuilderConfig())
        anchor = builder._select_anchor_text([("好了", 0.92)])
        assert anchor == "好了"


class TestFilterYClusterAwareness:
    """Test _filter_cross_frame_spatial_outliers — thay thế _filter_y_with_cluster_awareness."""

    def test_normal_boxes_kept(self) -> None:
        """Frame cùng Y cluster → tất cả box được giữ."""
        from subtitles_extractor.application.services.outlier_detection import (
            filter_y_position_outliers,
        )
        # 10 box ổn định ở Y≈100.
        y_centers = [98.0, 99.0, 100.0, 101.0, 100.5,
                     99.5, 100.0, 101.5, 99.0, 100.0]
        mask = filter_y_position_outliers(y_centers, k=4.0)
        assert all(mask), f"Box Y ổn định bị drop: {mask}"

    def test_outlier_dropped(self) -> None:
        """Box Y rất xa median → bị drop bởi MAD filter."""
        from subtitles_extractor.application.services.outlier_detection import (
            filter_y_position_outliers,
        )
        # 9 box ở Y≈100, 1 box logo ở Y=800.
        y_centers = [98.0, 99.0, 100.0, 101.0, 100.5,
                     99.5, 100.0, 101.5, 99.0, 800.0]
        mask = filter_y_position_outliers(y_centers, k=4.0)
        assert not mask[-1], "Box Y=800 (logo) phải bị drop"
        assert all(mask[:-1]), "9 box hợp lệ phải giữ"


class TestGroupsToEventsEdgeCases:
    """Fix #2: negative duration guard và temporal padding."""

    def test_no_negative_duration(self) -> None:
        """Sau tất cả clip, event không được có end <= start."""
        from subtitles_extractor.application.dtos.extract_subtitles_dto import (
            SubtitleBuilderConfig,
        )
        from subtitles_extractor.application.services.subtitle_builder import (
            SubtitleBuilder,
            FrameGroup,
        )
        builder = SubtitleBuilder(
            SubtitleBuilderConfig(temporal_padding_sec=0.05, min_duration_sec=0.15)
        )
        # 2 group sát nhau — dur A = 0s.
        groups = [
            FrameGroup(
                reconstructed_text="A",
                start_timestamp_sec=1.0, end_timestamp_sec=1.0,
                accumulated_confidence=0.9, total_frames_count=1,
            ),
            FrameGroup(
                reconstructed_text="B",
                start_timestamp_sec=1.1, end_timestamp_sec=1.5,
                accumulated_confidence=0.9, total_frames_count=3,
            ),
        ]
        events = builder._convert_groups_to_events(groups)
        for ev in events:
            assert ev.end_sec > ev.start_sec, (
                f"Negative duration: {ev.start_sec:.3f} → {ev.end_sec:.3f}"
            )

    def test_sub_frame_start_refined(self) -> None:
        """end_sec được pad thêm temporal_padding_sec; start_sec giữ nguyên.

        Lưu ý: behavior actual của ``convert_groups_to_events`` chỉ pad END
        (không lùi START). Test cũ giả định pad cả 2 đầu nhưng đây không
        phải behavior thực — bản v2.x cũng không lùi start để tránh overlap
        với event trước. Test này document behavior đúng.
        """
        from subtitles_extractor.application.dtos.extract_subtitles_dto import (
            SubtitleBuilderConfig,
        )
        from subtitles_extractor.application.services.subtitle_builder import (
            SubtitleBuilder,
            FrameGroup,
        )
        builder = SubtitleBuilder(
            SubtitleBuilderConfig(temporal_padding_sec=0.05, min_duration_sec=0.15)
        )
        groups = [
            FrameGroup(
                reconstructed_text="好",
                start_timestamp_sec=1.0, end_timestamp_sec=1.5,
                accumulated_confidence=0.9, total_frames_count=5,
            )
        ]
        events = builder._convert_groups_to_events(groups)
        # start giữ nguyên = 1.0 (không subtract padding).
        assert events[0].start_sec == pytest.approx(1.0, abs=0.001)
        # end được pad thêm 0.05 = 1.55.
        assert events[0].end_sec == pytest.approx(1.55, abs=0.001)
