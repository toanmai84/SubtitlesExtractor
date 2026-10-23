"""[v3.23.585] Báo cho người dùng khi phân tích ngữ cảnh CHƯA đủ mọi đoạn video.

Khâu phân tích giữ các đoạn đã xong và dừng sớm khi hết hạn mức ngày (v3.23.564) hoặc
khi Gemini quá tải kéo dài (v3.23.585). Trước đây kết quả thiếu đoạn vẫn hiện thông báo
"Phân tích hoàn tất" như một lượt đủ — chỉ nhật ký mới ghi là thiếu — nên người dùng
dịch luôn với bảng nhân vật và Visual Cues của một phần phim mà không biết chỉ cần bấm
phân tích lại là chạy tiếp từ chỗ dở.

Module chỉ chứa HÀM THUẦN: không I/O.
"""

from __future__ import annotations

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    ANALYSIS_STOP_DAILY_QUOTA,
    ANALYSIS_STOP_OVERLOAD,
    SubtitleContextAnalysis,
)

#: Liệt kê tối đa bấy nhiêu số thứ tự đoạn trong thông báo.
MAX_LISTED_SEGMENTS = 8

_REASONS = {
    ANALYSIS_STOP_OVERLOAD: (
        "Gemini đang quá tải (503) nên đã dừng để giữ hạn mức ngày. Bấm Phân tích lại "
        "sau ít phút, hoặc đổi model phân tích"
    ),
    ANALYSIS_STOP_DAILY_QUOTA: (
        "Đã hết hạn mức ngày của model ở mọi API key. Bấm Phân tích lại khi hạn mức "
        "hồi (00:00 giờ Thái Bình Dương — 14:00 giờ Việt Nam, 15:00 từ đầu tháng 11 tới "
        "giữa tháng 3), hoặc thêm API key thuộc dự án khác"
    ),
}
_DEFAULT_REASON = "Một số đoạn gặp lỗi. Bấm Phân tích lại"


def _segment_list(segments: tuple[int, ...]) -> str:
    listed = ", ".join(str(number) for number in segments[:MAX_LISTED_SEGMENTS])
    return f"{listed}…" if len(segments) > MAX_LISTED_SEGMENTS else listed


def describe_incomplete_analysis(analysis: SubtitleContextAnalysis) -> str:
    """Thông báo cho người dùng khi kết quả phân tích còn thiếu đoạn video.

    Args:
        analysis: Kết quả phân tích.

    Returns:
        Câu thông báo; rỗng khi phân tích đã đủ.
    """
    missing = tuple(analysis.missing_segments)
    if not missing:
        return ""
    total = max(analysis.total_segments, len(missing))
    reason = _REASONS.get(analysis.stop_reason, _DEFAULT_REASON)
    return (
        f"Mới phân tích {total - len(missing)}/{total} đoạn video — còn thiếu đoạn "
        f"{_segment_list(missing)}. {reason} — các đoạn đã xong được giữ, chỉ gửi đoạn "
        "còn thiếu. Bảng nhân vật và Visual Cues hiện tại chỉ phủ phần phim đã phân tích."
    )


__all__ = ["MAX_LISTED_SEGMENTS", "describe_incomplete_analysis"]
