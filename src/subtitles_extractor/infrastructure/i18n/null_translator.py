"""[v3.23.382] Lưới an toàn dịch thuật cho các widget được tạo từ tầng ngoài.

Một số widget (vd ``MultiROIReviewDialog``) được khởi tạo gián tiếp qua tầng infrastructure
(``ocr_based_auto_roi_detector.review_clusters``). Đường gọi thực tế LUÔN truyền translator
thật xuống, nhưng để phòng thủ (test/khởi tạo trực tiếp thiếu translator) ta dùng
:func:`resolve_translator` — không bao giờ để widget ném ``AttributeError`` vì thiếu translator.
"""

from __future__ import annotations

from typing import Any


class NullTranslator:
    """Translator rỗng: trả về chính khoá, KHÔNG bao giờ ném lỗi.

    Chỉ dùng làm phương án cuối khi không có translator thật (không nên xảy ra ở đường
    gọi chuẩn). Trả khoá thay vì crash để UI vẫn hiện được (dù là khoá thô).
    """

    def translate(self, key: str, **kwargs: Any) -> str:  # noqa: ARG002 — API tương thích
        return key

    def set_locale(self, locale: str) -> None:  # noqa: ARG002 — API tương thích
        return None


def resolve_translator(translator: Any) -> Any:
    """Trả về ``translator`` nếu hợp lệ, ngược lại trả :class:`NullTranslator`.

    Args:
        translator: Đối tượng translator (kỳ vọng có phương thức ``translate``) hoặc ``None``.

    Returns:
        Translator dùng được — không bao giờ ``None``.
    """
    if translator is not None and hasattr(translator, "translate"):
        return translator
    return NullTranslator()


__all__ = ["NullTranslator", "resolve_translator"]
