"""Hợp đồng dịch chuỗi giao diện (i18n).

Adapter chuẩn nạp từ file JSON, nhưng có thể thay bằng gettext nếu
trong tương lai cần thay đổi định dạng.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TranslatorPort(Protocol):
    """Tra cứu chuỗi theo khoá phân cấp."""

    @property
    def locale(self) -> str:
        """Mã ngôn ngữ hiện hành (``"vi"``, ``"en"``…)."""
        ...

    def set_locale(self, locale: str) -> None:
        """Đổi ngôn ngữ runtime."""
        ...

    def translate(self, key: str, **placeholders: object) -> str:
        """Trả về chuỗi đã dịch.

        Nếu ``key`` không tồn tại, adapter trả về chính ``key`` để tránh
        crash UI (fail-safe).
        """
        ...


__all__ = ["TranslatorPort"]
