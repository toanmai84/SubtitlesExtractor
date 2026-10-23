"""[v3.23.591] Gemini trả về KHÔNG có nội dung: bị CHẶN, hay chỉ là trục trặc nhất thời?

Từ v3.23.358, lời gọi có đính video mà trả rỗng thì BỎ video rồi thử lại ngay, với giả
định "video kích hoạt safety filter". Nhật ký thật v3.23.590 (phân tích 17 đoạn bằng
gemini-3.8-flash) cho thấy giả định đó không đúng cho mọi lần trả rỗng:

* Hai lần trả rỗng, cả hai ở đoạn 6 và cả hai giữa lúc máy chủ quá tải (liền trước/sau là
  các lượt 503). Sáng hôm sau, CÙNG đoạn video, CÙNG prompt chạy xong trong 32 giây — tức
  video không hề bị chặn.
* Bỏ video ở khâu PHÂN TÍCH là bỏ luôn căn cứ để gán người nói/người nghe cho ~75 dòng của
  đoạn đó; nếu lượt thử không video thành công thì kết quả vẫn được nhận, không dấu vết
  nào cho thấy đoạn ấy được phân tích "mù".
* Thông báo cũ chỉ in ``candidates=None`` — không ghi lý do máy chủ nêu, nên đọc nhật ký
  không phân biệt được hai trường hợp.

Quy tắc ở đây: máy chủ NÊU lý do chặn (``prompt_feedback.block_reason`` hoặc
``finish_reason`` thuộc nhóm chặn nội dung) thì bỏ video ngay như cũ — gửi lại đúng video
ấy chỉ tốn request. Không nêu lý do thì coi là nhất thời: thử lại MỘT lần vẫn kèm video;
rỗng lần nữa mới bỏ video.

Toàn bộ module là HÀM THUẦN: không I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Lý do máy chủ nêu khi CHẶN NỘI DUNG (đầu vào hoặc đầu ra) — gửi lại y nguyên vẫn bị
#: chặn.
BLOCKING_REASONS = frozenset({
    "SAFETY",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "MODEL_ARMOR",
    "JAILBREAK",
    "SPII",
    "RECITATION",
    "IMAGE_RECITATION",
})

#: [v3.23.622] Hết ngân sách đầu ra: token SUY NGHĨ tính vào ``max_output_tokens`` (thinking.md:
#: "returns truncated or empty output (while still billing for any thinking tokens)").
#: Đây là lý do RÕ RÀNG nhưng KHÔNG phải do video — bỏ video không cứu được, chỉ làm mất căn cứ
#: gán người nói. Cách chữa đúng là hạ ``thinking_level`` (tài liệu), không đổi video.
TRUNCATION_REASON = "MAX_TOKENS"

#: Số lần trả rỗng KHÔNG rõ lý do được thử lại kèm video trước khi bỏ video.
UNEXPLAINED_EMPTY_RETRIES_WITH_VIDEO = 1

_UNSPECIFIED_SUFFIX = "_UNSPECIFIED"


def _reason_name(value: object) -> str:
    """Tên lý do dạng chữ hoa (``"SAFETY"``); rỗng nếu không có hoặc "không xác định"."""
    if value is None:
        return ""
    name = str(getattr(value, "name", None) or value).strip().upper()
    name = name.rsplit(".", 1)[-1]
    return "" if not name or name.endswith(_UNSPECIFIED_SUFFIX) else name


@dataclass(frozen=True, slots=True)
class EmptyResponse:
    """Lý do máy chủ nêu cho một phản hồi không có nội dung.

    Attributes:
        block_reason: ``prompt_feedback.block_reason`` (đầu vào bị chặn); rỗng nếu
            không có.
        finish_reason: ``finish_reason`` của ứng viên đầu tiên, rỗng nếu không có.
    """

    block_reason: str = ""
    finish_reason: str = ""

    @property
    def truncated(self) -> bool:
        """Hết ngân sách token đầu ra (suy nghĩ + trả lời), không phải bị chặn."""
        return self.finish_reason == TRUNCATION_REASON

    @property
    def blocked(self) -> bool:
        """Máy chủ nêu rõ nội dung bị chặn."""
        return bool(self.block_reason) or self.finish_reason in BLOCKING_REASONS

    def describe(self) -> str:
        """Mô tả ngắn cho nhật ký: ``"block_reason=SAFETY"``, ``"không nêu lý do"``…"""
        parts = []
        if self.block_reason:
            parts.append(f"block_reason={self.block_reason}")
        if self.finish_reason:
            parts.append(f"finish_reason={self.finish_reason}")
        return ", ".join(parts) or "máy chủ không nêu lý do"


def diagnose_empty_response(response: object) -> EmptyResponse:
    """Đọc lý do của một phản hồi không có nội dung (hàm thuần).

    Args:
        response: Phản hồi ``generate_content`` của SDK Gemini (hoặc đối tượng cùng dạng).

    Returns:
        Lý do máy chủ nêu; các trường rỗng khi máy chủ không nêu gì.
    """
    feedback = getattr(response, "prompt_feedback", None)
    candidates = getattr(response, "candidates", None) or []
    first = candidates[0] if candidates else None
    return EmptyResponse(
        block_reason=_reason_name(getattr(feedback, "block_reason", None)),
        finish_reason=_reason_name(getattr(first, "finish_reason", None)),
    )


def should_drop_video(diagnosis: EmptyResponse, unexplained_empties: int) -> bool:
    """Có nên bỏ video ở lượt thử kế tiếp sau một phản hồi rỗng không (hàm thuần).

    Args:
        diagnosis: Lý do của phản hồi rỗng vừa nhận.
        unexplained_empties: Số phản hồi rỗng KHÔNG rõ lý do của lời gọi này, kể cả
            phản hồi vừa nhận.

    Returns:
        True nếu bị chặn rõ ràng, hoặc đã hết lượt thử lại kèm video.
    """
    if diagnosis.truncated:
        return False  # [v3.23.622] hết token do suy nghĩ — video vô can
    return diagnosis.blocked or unexplained_empties > UNEXPLAINED_EMPTY_RETRIES_WITH_VIDEO


__all__ = [
    "BLOCKING_REASONS",
    "TRUNCATION_REASON",
    "UNEXPLAINED_EMPTY_RETRIES_WITH_VIDEO",
    "EmptyResponse",
    "diagnose_empty_response",
    "should_drop_video",
]
