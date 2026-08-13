"""Enum :class:`DeviceKind` — thiết bị chạy suy luận OCR.

Thay magic string ``"gpu"`` / ``"cpu"`` bằng enum để type-checker (mypy)
bắt được lỗi truyền sai từ giai đoạn biên dịch.
"""

from __future__ import annotations

from enum import StrEnum


class DeviceKind(StrEnum):
    """Loại thiết bị chạy suy luận."""

    GPU = "gpu"
    CPU = "cpu"

    @classmethod
    def from_string(cls, raw: str) -> DeviceKind:
        """Parse linh hoạt từ chuỗi (không phân biệt hoa/thường)."""
        normalized = raw.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"Không nhận diện được thiết bị {raw!r}. "
            f"Giá trị hợp lệ: {[m.value for m in cls]}."
        )


class PrecisionMode(StrEnum):
    """Độ chính xác số thực khi bật TensorRT."""

    FP32 = "fp32"
    FP16 = "fp16"


class SubtitleFormat(StrEnum):
    """Định dạng tệp phụ đề xuất ra."""

    SRT = "srt"
    ASS = "ass"


__all__ = ["DeviceKind", "PrecisionMode", "SubtitleFormat"]
