"""Diễn giải lỗi từ Gemini API thành thông báo dễ hiểu kèm hướng dẫn khắc phục.

[v3.23.114] Trước đây khi gọi Gemini thất bại (dịch / TTS / phân tích ngữ cảnh), giao
diện hiển thị nguyên văn lỗi kỹ thuật (vd ``503 UNAVAILABLE``, ``RESOURCE_EXHAUSTED``)
khiến người dùng không biết phải làm gì. Module này là một HÀM THUẦN (dễ kiểm thử, không
phụ thuộc GUI) nhận chuỗi lỗi thô và trả về thông báo tiếng Việt có hướng dẫn cụ thể.

Nguyên tắc: chỉ NHẬN DẠNG các nhóm lỗi phổ biến để thêm hướng dẫn; vẫn giữ chi tiết gốc
để tiện gỡ lỗi. Nếu không khớp mẫu nào, trả về nguyên văn (không bịa thông tin).
"""

from __future__ import annotations

__all__ = ["humanize_gemini_error"]

# Mỗi mục: (các từ khoá nhận dạng (chữ thường), thông báo hướng dẫn). Xét theo thứ tự —
# mẫu cụ thể (key/quota) đặt trước mẫu chung (unavailable) để không bị nuốt nhầm.
_ERROR_HINTS: list[tuple[tuple[str, ...], str]] = [
    (
        ("api key not valid", "api_key_invalid", "invalid api key",
         "permission_denied", "api key expired", "unauthorized", " 401", " 403"),
        "API Key Gemini không hợp lệ hoặc đã hết hạn. Hãy kiểm tra lại Key "
        "(lấy Key mới tại https://aistudio.google.com/apikey) rồi dán vào ô 'API Key'.",
    ),
    (
        ("resource_exhausted", "quota", "rate limit", "ratelimit",
         "too many requests", " 429"),
        "Đã vượt hạn mức (quota) của Gemini. Hãy chờ vài phút rồi thử lại, giảm số "
        "luồng, hoặc dùng API Key của tài khoản có hạn mức cao hơn.",
    ),
    (
        ("safety", "blocked", "block_reason", "prohibited_content", "finish_reason"),
        "Nội dung bị bộ lọc an toàn của Gemini chặn. Thử đổi model, hoặc tách/sửa các "
        "câu nhạy cảm rồi dịch lại.",
    ),
    (
        ("503", "unavailable", "overloaded", "500", "internal error",
         "internal server"),
        "Máy chủ Gemini đang quá tải hoặc tạm gián đoạn (lỗi tạm thời). Thử lại sau ít "
        "phút; nếu đang dịch, tiến trình có thể tự tiếp tục từ checkpoint đã lưu.",
    ),
    (
        ("deadline", "timeout", "timed out", "deadline_exceeded"),
        "Hết thời gian chờ phản hồi từ Gemini. Kiểm tra kết nối mạng và thử lại; với "
        "phim dài, cân nhắc giảm kích thước mỗi lô (batch).",
    ),
    (
        ("getaddrinfo", "connection", "network", "ssl", "name resolution",
         "failed to establish", "max retries"),
        "Lỗi kết nối mạng tới máy chủ Gemini. Kiểm tra Internet/proxy/tường lửa rồi "
        "thử lại.",
    ),
]


def humanize_gemini_error(raw_message: str) -> str:
    """Trả về thông báo lỗi thân thiện kèm hướng dẫn dựa trên chuỗi lỗi thô.

    Args:
        raw_message: Chuỗi lỗi gốc (thường là ``str(exception)`` từ tầng gọi Gemini).

    Returns:
        Thông báo có hướng dẫn khắc phục nếu nhận dạng được nhóm lỗi (kèm chi tiết gốc).
        Nếu không khớp mẫu nào, trả về nguyên văn ``raw_message``.
    """
    if not raw_message:
        return raw_message
    lowered = raw_message.lower()
    for keywords, hint in _ERROR_HINTS:
        if any(kw in lowered for kw in keywords):
            return f"{hint}\n\n(Chi tiết kỹ thuật: {raw_message.strip()})"
    return raw_message
