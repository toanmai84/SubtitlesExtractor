"""Value object :class:`TimeInterval` — khoảng thời gian phụ đề.

Tách biệt khái niệm "khoảng thời gian" khỏi nội dung phụ đề giúp:
    * Sử dụng lại cho ROI sampling, video seek, …
    * Đặt đầy đủ ràng buộc tại một nơi (không âm, end ≥ start).
"""

from __future__ import annotations

from dataclasses import dataclass

from subtitles_extractor.domain.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class TimeInterval:
    """Khoảng thời gian ``[start_sec, end_sec]`` đo bằng giây.

    Attributes:
        start_sec: Mốc bắt đầu, không âm.
        end_sec:   Mốc kết thúc, phải ≥ ``start_sec``.

    Raises:
        ConfigurationError: Khi giá trị vi phạm ràng buộc.
    """

    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        if self.start_sec < 0:
            raise ConfigurationError(
                f"Mốc bắt đầu phải không âm, nhận được {self.start_sec:.3f}s."
            )
        if self.end_sec < self.start_sec:
            raise ConfigurationError(
                f"Mốc kết thúc ({self.end_sec:.3f}s) phải ≥ mốc bắt đầu "
                f"({self.start_sec:.3f}s)."
            )

    @property
    def duration_sec(self) -> float:
        """Thời lượng khoảng thời gian (giây)."""
        return self.end_sec - self.start_sec

    def start_ms(self) -> int:
        """Mốc bắt đầu làm tròn sang ms."""
        return int(round(self.start_sec * 1000))

    def end_ms(self) -> int:
        """Mốc kết thúc làm tròn sang ms."""
        return int(round(self.end_sec * 1000))

    def overlaps_with(self, other: TimeInterval) -> bool:
        """Trả về True nếu hai khoảng giao nhau (kể cả tiếp xúc đầu mút)."""
        return self.start_sec <= other.end_sec and other.start_sec <= self.end_sec


__all__ = ["TimeInterval"]
