"""[v3.23.175] Phân loại kết cục khi tiến trình dịch bị gián đoạn (thuần, test được).

Tách logic quyết định "người dùng huỷ THẬT" vs "gián đoạn do timeout/lỗi mạng" ra khỏi
QThread để kiểm thử bằng pytest mà không cần khởi tạo Qt (tránh segfault trong CI).
``TranslationCancelledError`` có thể phát sinh gián tiếp khi chờ Gemini quá lâu (phân
tích ngữ cảnh 8 đoạn có thể mất 30+ phút kèm 503/504); khi đó cờ huỷ KHÔNG bật và ta
phải báo LỖI (để người dùng thử lại), không báo "đã huỷ".
"""

from __future__ import annotations

from enum import Enum


class CancellationOutcome(Enum):
    """Kết cục sau khi bắt ``TranslationCancelledError``.

    Attributes:
        USER_CANCELLED: Người dùng thực sự bấm huỷ (cờ huỷ bật).
        INTERRUPTED_BY_ERROR: Gián đoạn do timeout/lỗi mạng, KHÔNG phải người dùng huỷ.
    """

    USER_CANCELLED = "user_cancelled"
    INTERRUPTED_BY_ERROR = "interrupted_by_error"


def classify_cancellation_outcome(cancel_flag_set: bool) -> CancellationOutcome:
    """Phân loại kết cục dựa trên trạng thái cờ huỷ tại thời điểm bắt ngoại lệ.

    Args:
        cancel_flag_set: ``True`` nếu cờ huỷ hợp tác đã được người dùng bật.

    Returns:
        :class:`CancellationOutcome` tương ứng — dùng để quyết định phát signal
        ``cancelled`` (êm) hay ``failed`` (báo lỗi để thử lại).
    """
    if cancel_flag_set:
        return CancellationOutcome.USER_CANCELLED
    return CancellationOutcome.INTERRUPTED_BY_ERROR


__all__ = ["CancellationOutcome", "classify_cancellation_outcome"]
