"""Lớp tương thích Qt — tập trung khác biệt giữa PySide6 và PyQt6 vào một chỗ.

[v3.23.267] Ứng dụng chuyển từ **PyQt6 (GPL v3)** sang **PySide6 (LGPL v3)** để dùng được
trong sản phẩm thương mại đóng mà không phải công khai nguồn. Xem LICENSE_ANALYSIS.md
và docs/PYSIDE6_MIGRATION_PLAN.md.

Module này cung cấp các tên thống nhất cho những chỗ hai binding khác nhau:

- ``Signal`` / ``Slot`` — PySide6 dùng ``Signal``/``Slot`` (PyQt6 dùng ``Signal``/
  ``Slot``). Code ứng dụng import từ đây để không phụ thuộc binding cụ thể.
- ``is_valid(obj)`` — thay cho ``not sip.isdeleted(obj)`` (PyQt6). PySide6 dùng
  ``shiboken6.isValid``. Trả về True nếu đối tượng C++ bên dưới CÒN sống.

Mọi khác biệt binding chỉ nằm ở đây → nếu sau này đổi binding lần nữa, chỉ sửa một file.
"""

from __future__ import annotations

from typing import Any

# PySide6 là binding chính (LGPL). Signal/Slot là tên chuẩn của PySide6.
from PySide6.QtCore import Signal, Slot

try:
    from shiboken6 import isValid as _shiboken_is_valid  # noqa: N813
except ImportError:  # pragma: no cover - PySide6 luôn kèm shiboken6
    _shiboken_is_valid = None


def is_valid(obj: Any) -> bool:
    """Trả về True nếu đối tượng Qt (C++ bên dưới) CÒN sống, chưa bị xoá.

    Thay cho thành ngữ PyQt6 ``not sip.isdeleted(obj)``. Dùng để bảo vệ trước khi truy cập
    widget có thể đã bị Qt huỷ (vd cửa sổ tách rời đã đóng, widget trong callback trễ).

    Args:
        obj: Đối tượng Qt cần kiểm tra.

    Returns:
        True nếu còn sống (an toàn để dùng); False nếu đã bị xoá hoặc None.
    """
    if obj is None:
        return False
    if _shiboken_is_valid is not None:
        return bool(_shiboken_is_valid(obj))
    # Fallback cực hiếm: không có shiboken6 -> coi như còn sống (không chặn nhầm).
    return True


def is_deleted(obj: Any) -> bool:
    """Nghịch đảo của :func:`is_valid` — thay trực tiếp cho ``sip.isdeleted(obj)``."""
    return not is_valid(obj)


__all__ = ["Signal", "Slot", "is_deleted", "is_valid"]
