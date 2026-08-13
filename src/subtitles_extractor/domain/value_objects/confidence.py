"""Value object :class:`Confidence` — điểm tin cậy nằm trong ``[0.0, 1.0]``.

Bao bọc ``float`` thô để bắt lỗi sớm nếu code gọi truyền nhầm thang đo
(0–100 chẳng hạn) hoặc số âm.
"""

from __future__ import annotations

from dataclasses import dataclass

from subtitles_extractor.domain.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True, order=True)
class Confidence:
    """Điểm tin cậy của một dự đoán OCR.

    Attributes:
        value: Số thực ``∈ [0.0, 1.0]``.

    Raises:
        ConfigurationError: Khi value nằm ngoài ``[0.0, 1.0]``.
    """

    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ConfigurationError(
                f"Điểm tin cậy phải nằm trong [0.0, 1.0], nhận được {self.value!r}."
            )

    @classmethod
    def zero(cls) -> Confidence:
        """Helper: trả về điểm tin cậy bằng 0."""
        return cls(0.0)

    @classmethod
    def from_percentage(cls, percent: int | float) -> Confidence:
        """Tạo :class:`Confidence` từ giá trị thang 0–100."""
        return cls(value=float(percent) / 100.0)

    def __float__(self) -> float:
        return self.value


__all__ = ["Confidence"]
