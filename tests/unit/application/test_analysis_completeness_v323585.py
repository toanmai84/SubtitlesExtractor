"""[v3.23.585] Câu thông báo khi phân tích ngữ cảnh còn thiếu đoạn video (hàm thuần)."""

from __future__ import annotations

from subtitles_extractor.application.services.analysis_completeness import (
    MAX_LISTED_SEGMENTS,
    describe_incomplete_analysis,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    ANALYSIS_STOP_DAILY_QUOTA,
    ANALYSIS_STOP_OVERLOAD,
    SubtitleContextAnalysis,
)


def test_complete_analysis_has_nothing_to_report() -> None:
    assert describe_incomplete_analysis(SubtitleContextAnalysis(overview="x")) == ""


def test_overload_stop_tells_how_to_continue() -> None:
    """Ca thật lần phân tích thứ hai: 16/17 đoạn, thiếu đoạn 14."""
    message = describe_incomplete_analysis(
        SubtitleContextAnalysis(
            missing_segments=(14,), total_segments=17, stop_reason=ANALYSIS_STOP_OVERLOAD
        )
    )
    assert message.startswith("Mới phân tích 16/17 đoạn video — còn thiếu đoạn 14.")
    assert "quá tải" in message
    assert "chỉ gửi đoạn còn thiếu" in message


def test_daily_quota_stop_names_its_own_remedy() -> None:
    """Ca thật lần phân tích đầu: dừng ở đoạn 8 vì mọi key hết hạn mức ngày."""
    message = describe_incomplete_analysis(
        SubtitleContextAnalysis(
            missing_segments=(1, 2, 3, 4, *range(6, 18)),
            total_segments=17,
            stop_reason=ANALYSIS_STOP_DAILY_QUOTA,
        )
    )
    assert "Mới phân tích 1/17 đoạn video" in message
    assert "1, 2, 3, 4, 6, 7, 8, 9…" in message
    assert "hết hạn mức ngày" in message


def test_unknown_reason_still_tells_what_is_missing() -> None:
    message = describe_incomplete_analysis(
        SubtitleContextAnalysis(missing_segments=(2,), total_segments=3)
    )
    assert "Mới phân tích 2/3 đoạn video — còn thiếu đoạn 2." in message
    assert "Bấm Phân tích lại" in message


def test_long_lists_are_cut() -> None:
    message = describe_incomplete_analysis(
        SubtitleContextAnalysis(missing_segments=tuple(range(1, 30)), total_segments=40)
    )
    assert f"{MAX_LISTED_SEGMENTS}…" in message
    assert f"{MAX_LISTED_SEGMENTS + 1}" not in message.split("…")[0]
