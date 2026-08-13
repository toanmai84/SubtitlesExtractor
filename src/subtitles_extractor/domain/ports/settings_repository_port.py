"""Hợp đồng kho lưu trữ cấu hình bền vững.

Adapter mặc định dùng :class:`PyQt6.QtCore.QSettings` (registry trên
Windows, plist trên macOS, INI trên Linux). Có thể thay bằng adapter
JSON-file/YAML-file để test mà không cần PyQt.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SettingsRepositoryPort(Protocol):
    """Đọc/ghi cấu hình theo từng key.

    Kiểu của ``value`` được quản lý bởi tầng application (qua pydantic
    settings) — adapter chỉ đảm bảo round-trip: ``load(key) == saved value``.
    """

    def load(self, key: str, default: Any) -> Any:
        """Đọc giá trị key từ kho. Trả về ``default`` nếu key chưa tồn tại."""
        ...

    def save(self, key: str, value: Any) -> None:
        """Lưu một cặp (key, value). Có thể trì hoãn flush — gọi ``flush`` để chắc."""
        ...

    def flush(self) -> None:
        """Buộc ghi tất cả thay đổi xuống đĩa."""
        ...

    def reset(self) -> None:
        """Xoá toàn bộ cấu hình đã lưu trong kho."""
        ...


__all__ = ["SettingsRepositoryPort"]
