"""Entity mô tả trạng thái thao tác cuối cùng của người dùng trên một video."""

from __future__ import annotations

from dataclasses import dataclass

from subtitles_extractor.domain.value_objects.roi import Roi


@dataclass(slots=True)
class VideoState:
    """Trạng thái đã lưu của một video.

    Attributes:
        video_path: Đường dẫn tuyệt đối của video (Primary Key).
        roi: Vùng quan tâm (ROI) cuối cùng người dùng đã chọn.
    """
    video_path: str
    roi: Roi | None = None

__all__ = ["VideoState"]
