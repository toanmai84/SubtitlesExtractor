"""Kiểm tra trước khi trích xuất và chẩn đoán khi không ra kết quả (thuần, không Qt).

VÌ SAO cần
==========
Trích xuất OCR một tập mất rất nhiều thời gian. Hiện tại:

* **Không kiểm gì trước khi chạy** — ``start_extraction`` chỉ xét "đã nạp video chưa"
  và "có đang bận không". ROI vẽ lệch ra ngoài khung, ROI bé xíu, hay tệp video đã bị
  di chuyển đều không bị chặn; người dùng chờ xong mới biết hỏng.
* **Ra 0 câu vẫn báo THÀNH CÔNG** — dòng trạng thái hiện màu xanh
  ``"✓ Hoàn tất! 0 câu … → Đã lưu vào Database"``. Vừa sai (không lưu gì) vừa khiến
  người dùng tưởng phim không có phụ đề, trong khi nguyên nhân thường là ROI/ngôn ngữ.

Module này tách riêng phần *phán đoán* để kiểm thử được đầy đủ, và để cùng một logic
dùng chung cho cả chạy đơn lẻ lẫn chạy hàng loạt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

# ROI nhỏ hơn mức này gần như chắc chắn là vẽ nhầm (kéo trượt tay), không phải chủ ý.
_MIN_ROI_WIDTH_PX: Final[int] = 24
_MIN_ROI_HEIGHT_PX: Final[int] = 12

# ROI cao hơn tỉ lệ này của khung hình thì nhiều khả năng người dùng quên thu hẹp —
# OCR toàn khung rất chậm và dễ bắt nhầm chữ trong cảnh phim.
_TALL_ROI_RATIO: Final[float] = 0.6


class IssueLevel(StrEnum):
    """Mức độ của một phát hiện trước khi chạy."""

    BLOCKER = "blocker"
    """Chắc chắn hỏng — nên chặn, không cho chạy."""

    WARNING = "warning"
    """Có thể vẫn chạy được nhưng nhiều khả năng cho kết quả kém."""


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    """Một vấn đề phát hiện trước khi chạy trích xuất.

    Attributes:
        level: Mức độ nghiêm trọng.
        message: Mô tả vấn đề cho người dùng.
        hint: Việc nên làm để khắc phục.
    """

    level: IssueLevel
    message: str
    hint: str

    @property
    def is_blocker(self) -> bool:
        """``True`` nếu nên chặn không cho chạy."""
        return self.level is IssueLevel.BLOCKER


def check_before_extraction(
    *,
    video_path: Path | None,
    video_size: tuple[int, int] | None,
    roi: tuple[int, int, int, int] | None,
    duration_sec: float = 0.0,
) -> list[PreflightIssue]:
    """Kiểm các điều kiện dễ hỏng TRƯỚC khi tốn thời gian chạy.

    Args:
        video_path: Đường dẫn video (``None`` nếu chưa nạp).
        video_size: ``(rộng, cao)`` của video.
        roi: ``(x, y, rộng, cao)`` vùng OCR; ``None`` = để hệ thống tự dò.
        duration_sec: Thời lượng video (giây).

    Returns:
        Danh sách vấn đề, rỗng nghĩa là sẵn sàng chạy.
    """
    issues: list[PreflightIssue] = []

    if video_path is None:
        issues.append(
            PreflightIssue(
                IssueLevel.BLOCKER,
                "Chưa chọn video.",
                "Bấm “Chọn video” để bắt đầu.",
            )
        )
        return issues

    if not video_path.is_file():
        issues.append(
            PreflightIssue(
                IssueLevel.BLOCKER,
                f"Không tìm thấy tệp video: {video_path.name}",
                "Tệp có thể đã bị di chuyển hoặc xoá — hãy chọn lại.",
            )
        )
        return issues

    if not video_size or min(video_size) <= 0:
        issues.append(
            PreflightIssue(
                IssueLevel.BLOCKER,
                "Không đọc được kích thước khung hình.",
                "Tệp có thể hỏng hoặc không phải video — thử tệp khác.",
            )
        )
        return issues

    if duration_sec <= 0:
        issues.append(
            PreflightIssue(
                IssueLevel.WARNING,
                "Không xác định được thời lượng video.",
                "Vẫn chạy được, nhưng tiến độ sẽ không hiển thị chính xác.",
            )
        )

    if roi is not None:
        issues.extend(_check_roi(roi, video_size))

    return issues


def _check_roi(
    roi: tuple[int, int, int, int], video_size: tuple[int, int]
) -> list[PreflightIssue]:
    """Kiểm ROI có hợp lý so với khung hình không."""
    issues: list[PreflightIssue] = []
    x, y, width, height = roi
    video_width, video_height = video_size

    if width <= 0 or height <= 0:
        issues.append(
            PreflightIssue(
                IssueLevel.BLOCKER,
                "Vùng OCR có kích thước bằng 0.",
                "Vẽ lại vùng chứa phụ đề, hoặc bấm “Tự dò ROI”.",
            )
        )
        return issues

    if x < 0 or y < 0 or x + width > video_width or y + height > video_height:
        issues.append(
            PreflightIssue(
                IssueLevel.BLOCKER,
                f"Vùng OCR nằm ngoài khung hình ({video_width}×{video_height}).",
                "Có thể do đổi sang video khác độ phân giải — vẽ lại vùng.",
            )
        )
        return issues

    if width < _MIN_ROI_WIDTH_PX or height < _MIN_ROI_HEIGHT_PX:
        issues.append(
            PreflightIssue(
                IssueLevel.WARNING,
                f"Vùng OCR rất nhỏ ({width}×{height} điểm ảnh).",
                "Có thể do kéo trượt tay — kiểm lại vùng có bao trọn dòng chữ không.",
            )
        )

    if height > video_height * _TALL_ROI_RATIO:
        issues.append(
            PreflightIssue(
                IssueLevel.WARNING,
                "Vùng OCR phủ phần lớn khung hình.",
                "Thu hẹp về đúng dải phụ đề sẽ nhanh hơn nhiều và ít bắt nhầm chữ "
                "trong cảnh phim.",
            )
        )

    return issues


def diagnose_empty_result(
    *,
    frames_processed: int,
    roi: tuple[int, int, int, int] | None,
    ocr_language: str,
    probed: bool = False,
) -> str:
    """Giải thích vì sao trích xuất không ra câu nào và nên làm gì.

    Ra 0 câu KHÔNG phải thành công — nhưng cũng chưa chắc là lỗi phần mềm. Hàm này
    xếp nguyên nhân theo thứ tự khả năng, dựa trên dữ liệu thực tế của lần chạy.

    Args:
        frames_processed: Số khung hình đã xử lý.
        roi: Vùng OCR đã dùng (``None`` = tự dò).
        ocr_language: Mã ngôn ngữ OCR đã dùng.
        probed: ``True`` nếu đây là lần "Thử nhanh" (đoạn ngắn).

    Returns:
        Chuỗi nhiều dòng: kết luận + các nguyên nhân xếp theo khả năng.
    """
    if frames_processed <= 0:
        return (
            "Không đọc được khung hình nào từ video.\n"
            "• Tệp có thể hỏng hoặc dùng codec lạ — thử mở bằng trình phát khác.\n"
            "• Nếu video nằm trên ổ mạng, thử chép về máy rồi chạy lại."
        )

    causes: list[str] = []
    if roi is not None:
        causes.append(
            "Vùng OCR chưa trùng vị trí phụ đề — đây là nguyên nhân hay gặp nhất. "
            "Tua tới đoạn CÓ phụ đề rồi vẽ lại vùng, hoặc bấm “Tự dò ROI”."
        )
    else:
        causes.append(
            "Tự dò ROI không tìm được dải phụ đề — hãy tự vẽ vùng bao quanh dòng chữ."
        )

    causes.append(
        f"Sai ngôn ngữ OCR (đang dùng “{ocr_language or 'mặc định'}”) — "
        "đổi ở ô “Ngôn ngữ phụ đề” cho khớp chữ trong phim."
    )

    if probed:
        causes.append(
            "Đoạn thử rơi vào chỗ không có thoại — đổi vị trí thử sang đoạn khác."
        )

    causes.append(
        "Phim không có phụ đề cháy vào hình — nếu phụ đề bật/tắt được thì dùng "
        "“Quét track phụ đề nhúng” thay vì OCR."
    )

    header = (
        f"Đã xử lý {frames_processed:,} khung hình nhưng KHÔNG nhận được câu nào.\n"
        "Các nguyên nhân theo thứ tự khả năng:"
    )
    body = "\n".join(f"• {cause}" for cause in causes)
    return f"{header}\n{body}"


def summarise_issues(issues: list[PreflightIssue]) -> str:
    """Gộp danh sách vấn đề thành thông điệp hiển thị.

    Args:
        issues: Kết quả của :func:`check_before_extraction`.

    Returns:
        Chuỗi nhiều dòng; rỗng nếu không có vấn đề nào.
    """
    if not issues:
        return ""
    lines: list[str] = []
    for issue in issues:
        marker = "⛔" if issue.is_blocker else "⚠️"
        lines.append(f"{marker} {issue.message}\n   → {issue.hint}")
    return "\n".join(lines)


def has_blocker(issues: list[PreflightIssue]) -> bool:
    """``True`` nếu có vấn đề nghiêm trọng, không nên chạy."""
    return any(issue.is_blocker for issue in issues)


__all__ = [
    "IssueLevel",
    "PreflightIssue",
    "check_before_extraction",
    "diagnose_empty_result",
    "has_blocker",
    "summarise_issues",
]
