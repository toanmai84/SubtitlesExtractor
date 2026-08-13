"""Chính sách quyết định XOAY API key khi chọn key đầu phiên (thuần, không phụ thuộc).

Tách khỏi :class:`GeminiTranslationAdapter` để:
    * Kiểm thử được độc lập (không cần ``google-genai`` hay quota manager thật).
    * Tuân thủ SRP — quyết định chính sách tách khỏi cơ chế xoay/HTTP.

BỐI CẢNH (v3.23.294)
====================
Gemini File API cô lập file đã tải lên theo từng API key/project. Khi phase phân
tích ngữ cảnh tải video lên dưới key A, rồi phase dịch xoay sang key B để "dư
quota hơn", file thuộc key A không dùng được với key B → buộc **cắt + nén +
tải lên lại** (tốn vài phút + thêm 1 request). Log thực tế (tập 4, 5) cho thấy
đúng hiện tượng này: analyze dùng key #1, translate xoay sang #2/#3 → upload lại.

Nghịch lý: heuristic ``much_better`` (xoay khi key khác dư dả hơn hẳn) sinh ra để
*tránh* cạn quota giữa phiên, nhưng lại *gây* tải lại video ngay đầu phiên vì nó
không tính chi phí re-upload của video đã gắn với key hiện tại.

Chính sách ở đây phân biệt 2 loại xoay:
    * **Bắt buộc**: key hiện tại đã cạn (``<= 0``) hoặc KHÔNG đủ đi trọn phiên
      (``insufficient``) — vẫn phải xoay dù có mất công upload lại (không còn lựa
      chọn tốt hơn).
    * **Cơ hội** (``much_better``): chỉ để dư quota — BỎ khi ``avoid_reupload``
      bật (đang giữ video đã upload dưới key hiện tại), vì lúc đó xoay là LỖ ròng.
"""

from __future__ import annotations

# Ngưỡng "dư dả hơn hẳn" cho xoay cơ hội: key khác phải còn >= gấp đôi HOẶC nhiều
# hơn tối thiểu 5 request so với key hiện tại. Giữ nguyên hành vi lịch sử.
_MUCH_BETTER_MIN_EXTRA_REQUESTS: int = 5


def is_much_better(current_remaining: int, best_remaining: int) -> bool:
    """True nếu ``best`` dư dả hơn hẳn ``current`` (đủ để xoay cơ hội).

    Args:
        current_remaining: Số request/ngày còn lại của key hiện tại.
        best_remaining: Số request/ngày còn lại của key tốt nhất khác.

    Returns:
        True khi ``best_remaining >= max(current*2, current + 5)``.
    """
    threshold = max(
        current_remaining * 2, current_remaining + _MUCH_BETTER_MIN_EXTRA_REQUESTS
    )
    return best_remaining >= threshold


def is_insufficient(
    current_remaining: int, best_remaining: int, needed_requests: int
) -> bool:
    """True nếu key hiện tại KHÔNG đủ đi trọn phiên VÀ có key khác nhiều hơn.

    Args:
        current_remaining: Request/ngày còn lại của key hiện tại.
        best_remaining: Request/ngày còn lại của key tốt nhất khác.
        needed_requests: Số request dự trù cho trọn phiên (>= 1).

    Returns:
        True khi ``current_remaining < needed_requests`` và ``best`` nhiều hơn.
    """
    return (
        current_remaining < max(1, int(needed_requests))
        and best_remaining > current_remaining
    )


def should_switch_for_viability(
    *,
    current_remaining: int,
    best_remaining: int,
    needed_requests: int,
    avoid_reupload: bool,
) -> bool:
    """Quyết định có nên xoay sang key tốt nhất khác khi chọn key đầu phiên.

    Args:
        current_remaining: Request/ngày còn lại của key hiện tại.
        best_remaining: Request/ngày còn lại của key tốt nhất khác (đã biết
            khác key hiện tại).
        needed_requests: Số request dự trù cho trọn phiên (>= 1).
        avoid_reupload: True khi 1 video ĐÃ tải lên dưới key hiện tại và việc xoay
            sẽ buộc tải lại — khi đó KHÔNG xoay vì lý do cơ hội, chỉ xoay khi bắt
            buộc (cạn hoặc không đủ).

    Returns:
        True nếu nên xoay sang key tốt nhất khác.
    """
    # Bắt buộc: key hiện tại đã cạn quota ngày.
    if current_remaining <= 0:
        return True
    # Bắt buộc: không đủ đi trọn phiên (và có key khác khá hơn).
    if is_insufficient(current_remaining, best_remaining, needed_requests):
        return True
    # Cơ hội: chỉ xoay khi KHÔNG phải giữ video đã upload.
    if avoid_reupload:
        return False
    return is_much_better(current_remaining, best_remaining)


__all__ = [
    "is_insufficient",
    "is_much_better",
    "should_switch_for_viability",
]
