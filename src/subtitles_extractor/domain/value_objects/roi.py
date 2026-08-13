"""Value object :class:`Roi` — vùng quan tâm trên khung hình.

Bất biến (frozen) để có thể sử dụng làm khoá dict hoặc set.
[NEW]: Tích hợp TextAlignment và TextOrientation để làm bộ lọc dọn rác OCR.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from subtitles_extractor.domain.exceptions import ConfigurationError


class TextAlignment(Enum):
    CENTER = auto()
    LEFT = auto()
    RIGHT = auto()
    UNKNOWN = auto()


class TextOrientation(Enum):
    HORIZONTAL = auto()
    VERTICAL = auto()


@dataclass(frozen=True, slots=True)
class Roi:
    """Hình chữ nhật mô tả vùng quan tâm kèm siêu dữ liệu Căn lề."""
    x: int
    y: int
    width: int
    height: int
    alignment: TextAlignment = TextAlignment.UNKNOWN
    orientation: TextOrientation = TextOrientation.HORIZONTAL

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ConfigurationError(
                f"Toạ độ ROI phải không âm, nhận được ({self.x}, {self.y})."
            )
        if self.width <= 0 or self.height <= 0:
            raise ConfigurationError(
                f"Kích thước ROI phải dương, nhận được "
                f"({self.width}×{self.height})."
            )

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_xywh_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def clip_to(self, frame_width: int, frame_height: int) -> Roi:
        new_x = max(0, min(self.x, frame_width - 1))
        new_y = max(0, min(self.y, frame_height - 1))
        new_w = max(1, min(self.width, frame_width - new_x))
        new_h = max(1, min(self.height, frame_height - new_y))
        return Roi(
            x=new_x, y=new_y, width=new_w, height=new_h,
            alignment=self.alignment, orientation=self.orientation
        )


__all__ = ["Roi", "TextAlignment", "TextOrientation"]
