"""[v3.23.622] Bí danh model Gemini đã ngừng nhưng vẫn được phục vụ (định tuyến sang bản mới).

Tài liệu Gemini API (deprecations.md, 8/10/2026): ``gemini-3.5-flash`` tự động chuyển sang
``gemini-3.6-flash`` và ``gemini-3.7-flash`` sang ``gemini-3.8-flash`` ("endpoint không
ngừng hẳn nhưng không có ngày đảm bảo"). Thăm dò thật: ``response.model_version`` của lời
gọi ``gemini-3.5-flash`` là ``gemini-3.6-flash``. Nghĩa là mọi so sánh theo TÊN (sức mạnh
model để dùng lại kết quả đã lưu, mức suy nghĩ hợp lệ, giá) phải coi bí danh = bản đích.
Hạn mức/ngày thì KHÔNG gộp: thăm dò cho thấy bí danh có thùng hạn mức riêng.
"""

from __future__ import annotations

#: Bí danh đã ngừng → model đích đang thực sự phục vụ.
DEPRECATED_ALIASES: dict[str, str] = {
    "gemini-3.5-flash": "gemini-3.6-flash",
    "gemini-3.7-flash": "gemini-3.8-flash",
}


def canonical_model_id(model_name: str) -> str:
    """Tên model thực sự phục vụ lời gọi (đã bỏ bí danh đã ngừng).

    Args:
        model_name: Tên người dùng chọn, vd ``"gemini-3.5-flash"``.

    Returns:
        Tên chữ thường của model đích; tên không phải bí danh được trả nguyên (chữ thường).
    """
    name = (model_name or "").strip().lower()
    return DEPRECATED_ALIASES.get(name, name)


def is_deprecated_alias(model_name: str) -> bool:
    """``True`` nếu ``model_name`` là bí danh đã ngừng (vẫn chạy nhưng định tuyến)."""
    return (model_name or "").strip().lower() in DEPRECATED_ALIASES


__all__ = ["DEPRECATED_ALIASES", "canonical_model_id", "is_deprecated_alias"]
