"""Hợp đồng báo cáo tiến độ trong các tác vụ dài.

Tách biệt với UI để use case không phụ thuộc Qt — adapter UI có thể
đẩy tiến độ qua signal/slot, adapter CLI có thể in ra ``stdout``,
adapter test có thể ghi vào list để kiểm tra.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProgressReporterPort(Protocol):
    """Báo cáo tiến độ tới tầng presentation."""

    def report(self, current: int, total: int, message: str = "") -> None:
        """Cập nhật tiến độ.

        Args:
            current: Số đơn vị công việc đã hoàn thành.
            total:   Tổng số đơn vị; ``0`` nghĩa là không xác định.
            message: Thông điệp ngắn (tuỳ chọn).
        """
        ...

    def is_cancelled(self) -> bool:
        """Trả về ``True`` nếu người dùng yêu cầu huỷ tác vụ."""
        ...


class NullProgressReporter:
    """Triển khai mặc định không làm gì — dùng khi không cần theo dõi."""

    def report(self, current: int, total: int, message: str = "") -> None:
        """Không làm gì."""

    def is_cancelled(self) -> bool:
        """Luôn trả về ``False``."""
        return False


__all__ = ["NullProgressReporter", "ProgressReporterPort"]
