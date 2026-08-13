"""Hợp đồng trích xuất phụ đề NHÚNG SẴN (embedded/soft subtitle) trong video.

Nhiều container (MKV, MP4, MOV…) chứa sẵn track phụ đề tách rời khỏi hình ảnh.
Trích các track này nhanh và chính xác hơn OCR hardsub:
  * Track TEXT-BASED (SubRip/ASS/mov_text): rút thẳng văn bản — chính xác 100%.
  * Track BITMAP (PGS/VOBSUB): phụ đề là ẢNH → cần OCR (PaddleOCR) ở tầng use-case.

Adapter chịu trách nhiệm liệt kê track (qua ffprobe) và trích từng track (qua
ffmpeg). Với track bitmap, adapter chỉ tách ẢNH + mốc thời gian; việc OCR do
use-case điều phối để tái dùng ``OcrEnginePort`` sẵn có.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent


@dataclass(frozen=True, slots=True)
class EmbeddedSubtitleTrack:
    """Mô tả một track phụ đề nhúng do ffprobe phát hiện.

    Attributes:
        track_index:  Chỉ số stream phụ đề trong container (0-based theo nhóm ``s``).
        codec:        Mã codec (``"subrip"``, ``"ass"``, ``"mov_text"``,
                      ``"hdmv_pgs_subtitle"``, ``"dvd_subtitle"``…).
        language:     Mã ngôn ngữ ISO 639 nếu có (``"eng"``, ``"vie"``, ``""``).
        title:        Nhãn track do người tạo file đặt (có thể rỗng).
        is_bitmap:    ``True`` nếu track là ảnh (cần OCR), ``False`` nếu text-based.
    """

    track_index: int
    codec: str
    language: str = ""
    title: str = ""
    is_bitmap: bool = False

    @property
    def display_label(self) -> str:
        """Nhãn thân thiện để hiển thị trên combo chọn track."""
        parts: list[str] = [f"#{self.track_index}"]
        if self.language:
            parts.append(self.language)
        if self.title:
            parts.append(self.title)
        kind = "ảnh/OCR" if self.is_bitmap else "văn bản"
        parts.append(f"({self.codec}, {kind})")
        return " · ".join(parts)


@dataclass(frozen=True, slots=True)
class BitmapSubtitleFrame:
    """Một ảnh phụ đề bitmap kèm khoảng thời gian hiển thị (chờ OCR)."""

    image_path: Path
    start_sec: float
    end_sec: float


@dataclass(frozen=True, slots=True)
class EmbeddedExtractionResult:
    """Kết quả trích một track phụ đề nhúng.

    Đúng MỘT trong hai trường có dữ liệu:
      * ``events`` — track text-based đã parse xong (dùng trực tiếp).
      * ``bitmap_frames`` — track bitmap, cần OCR ở tầng use-case.

    Attributes:
        events:         Danh sách câu phụ đề (text-based).
        bitmap_frames:  Danh sách ảnh + timestamp (bitmap, chờ OCR).
        is_bitmap:      Cho biết kết quả thuộc nhánh bitmap hay text.
    """

    events: list[SubtitleEvent] = field(default_factory=list)
    bitmap_frames: list[BitmapSubtitleFrame] = field(default_factory=list)
    is_bitmap: bool = False


@runtime_checkable
class EmbeddedSubtitlePort(Protocol):
    """Liệt kê & trích phụ đề nhúng từ container video."""

    def list_tracks(self, video_path: Path) -> list[EmbeddedSubtitleTrack]:
        """Liệt kê mọi track phụ đề nhúng trong video.

        Args:
            video_path: Đường dẫn tệp video.

        Returns:
            Danh sách track (rỗng nếu video không có phụ đề nhúng).

        Raises:
            FileNotFoundError: Khi không tìm thấy video.
            VideoDecodeError:  Khi ffprobe lỗi đọc container.
        """
        ...

    def extract_track(
        self, video_path: Path, track: EmbeddedSubtitleTrack
    ) -> EmbeddedExtractionResult:
        """Trích một track phụ đề nhúng.

        Track text-based → parse ngay thành ``events``. Track bitmap → tách ảnh +
        timestamp vào ``bitmap_frames`` để use-case OCR.

        Args:
            video_path: Đường dẫn tệp video.
            track:      Track cần trích (lấy từ :meth:`list_tracks`).

        Returns:
            Kết quả trích (text hoặc bitmap).

        Raises:
            VideoDecodeError: Khi ffmpeg lỗi trích track.
        """
        ...


__all__ = [
    "EmbeddedSubtitleTrack",
    "BitmapSubtitleFrame",
    "EmbeddedExtractionResult",
    "EmbeddedSubtitlePort",
]
