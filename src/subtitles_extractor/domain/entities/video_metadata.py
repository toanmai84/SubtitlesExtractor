"""Entity :class:`VideoMetadata` — thông tin tệp video.

Đặc điểm:
    * Đây là **entity** vì có định danh là ``path`` (đại diện một tệp cụ thể).
    * Bất biến (frozen) — bất kỳ chỉnh sửa nào (ví dụ gắn ROI mới) đều phải
      tạo instance mới qua :meth:`replace_roi`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from subtitles_extractor.domain.exceptions import ConfigurationError
from subtitles_extractor.domain.value_objects.roi import Roi


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Metadata của một tệp video.

    Attributes:
        path:          Đường dẫn tuyệt đối tới tệp.
        width:         Chiều rộng khung hình (px).
        height:        Chiều cao khung hình (px).
        fps:           Số khung hình trên giây.
        total_frames:  Tổng số khung hình.
        duration_sec:  Thời lượng (giây).
        codec:         Mã codec (ví dụ ``"h264"``, ``"hevc"``).
        roi:           Vùng quan tâm do người dùng chọn, hoặc ``None``.

    Raises:
        ConfigurationError: Khi giá trị vi phạm ràng buộc.
    """

    path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    codec: str = ""
    roi: Roi | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ConfigurationError(
                f"Kích thước video phải dương, nhận được "
                f"{self.width}×{self.height}."
            )
        if self.fps <= 0:
            raise ConfigurationError(
                f"FPS phải dương, nhận được {self.fps:.3f}."
            )
        if self.total_frames < 0:
            raise ConfigurationError(
                f"Số frame không được âm, nhận được {self.total_frames}."
            )
        if self.duration_sec < 0:
            raise ConfigurationError(
                f"Thời lượng không được âm, nhận được {self.duration_sec:.3f}s."
            )

    @property
    def filename(self) -> str:
        """Tên tệp (không kèm thư mục)."""
        return self.path.name

    @property
    def aspect_ratio(self) -> float:
        """Tỉ lệ khung hình (chiều rộng / chiều cao)."""
        return self.width / self.height

    @property
    def duration_str(self) -> str:
        """Thời lượng định dạng ``HH:MM:SS``."""
        total = int(self.duration_sec)
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def replace_roi(self, new_roi: Roi | None) -> VideoMetadata:
        """Trả về bản sao có ROI mới (giữ nguyên các trường khác)."""
        if new_roi is not None:
            new_roi = new_roi.clip_to(self.width, self.height)
        return replace(self, roi=new_roi)


__all__ = ["VideoMetadata"]
