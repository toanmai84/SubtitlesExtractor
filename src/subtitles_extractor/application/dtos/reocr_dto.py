"""DTO cho use case Re-OCR.

Tách khỏi :class:`ExtractSubtitlesRequest` để loại bỏ các trường không
phù hợp (output_path, output_format, skip_export…) — Re-OCR luôn trả
trong-bộ-nhớ và **không** xuất file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    OcrEngineConfig,
    SubtitleBuilderConfig,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.frame_sampler_port import (
    FrameSamplingConfig,
)
from subtitles_extractor.domain.value_objects.roi import Roi


@dataclass(frozen=True, slots=True)
class TimeRange:
    """Khoảng thời gian đóng ``[start_sec, end_sec]`` (giây).

    Raises:
        ValueError: Nếu ``start_sec >= end_sec`` hoặc giá trị âm.
    """

    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        if self.start_sec < 0 or self.end_sec < 0:
            raise ValueError(
                f"TimeRange không nhận giá trị âm: "
                f"start={self.start_sec}, end={self.end_sec}."
            )
        if self.start_sec >= self.end_sec:
            raise ValueError(
                f"TimeRange không hợp lệ: start={self.start_sec} "
                f">= end={self.end_sec}."
            )

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def overlaps(self, other: TimeRange) -> bool:
        """``True`` nếu hai khoảng có giao nhau (không kể tiếp giáp)."""
        return self.start_sec < other.end_sec and other.start_sec < self.end_sec

    def merge(self, other: TimeRange) -> TimeRange:
        """Trộn 2 khoảng thành 1 khoảng tối thiểu chứa cả hai."""
        return TimeRange(
            start_sec=min(self.start_sec, other.start_sec),
            end_sec=max(self.end_sec, other.end_sec),
        )


@dataclass(frozen=True, slots=True)
class ReOcrRequest:
    """Yêu cầu chạy lại OCR cho một hoặc nhiều khoảng thời gian.

    Khác :class:`ExtractSubtitlesRequest`:
        * Không xuất file — kết quả chỉ trả trong-bộ-nhớ.
        * Hỗ trợ **danh sách time range** thay vì 1 khoảng duy nhất —
          xử lý chính xác trường hợp user chọn rows không liền kề.
        * ``replace_uids`` — thay thế *chính xác* các SubtitleEvent qua
          UID, không qua index (chống bug #1).

    Attributes:
        video_path:    Đường dẫn video gốc.
        time_ranges:   Danh sách khoảng thời gian cần OCR lại. Mỗi khoảng
                       sẽ được xử lý độc lập để tránh OCR lan vùng giữa
                       các rows không liền kề.
        replace_uids:  UID của các SubtitleEvent sẽ bị thay thế. Pipeline
                       chỉ xoá các event có UID này, không xoá theo index.
        roi:           ROI áp dụng cho mọi time range. ``None`` = full frame.
        sampling:      Cấu hình lấy mẫu frame.
        ocr:           Cấu hình OCR engine (model, device, preprocess).
        builder:       Cấu hình SubtitleBuilder.
        auto_tune_batch:    Có cho phép auto-tune batch_size không.
        save_debug_frames:  Có ghi frame ra ổ đĩa không.
        debug_frames_dir:   Thư mục ghi debug frame (None = mặc định).
        merge_window_sec:   Khoảng thời gian tối đa giữa 2 time range để
                            gộp chúng lại trước khi OCR (0 = không gộp).
                            Giúp tiết kiệm thời gian khi user chọn rows
                            gần nhau (vd: rows 5, 6, 7).
    """

    video_path: Path
    time_ranges: list[TimeRange]
    replace_uids: list[str]
    roi: Roi | None
    sampling: FrameSamplingConfig
    ocr: OcrEngineConfig
    builder: SubtitleBuilderConfig
    auto_tune_batch: bool = True
    save_debug_frames: bool = False
    debug_frames_dir: str | None = None
    merge_window_sec: float = 1.0

    def __post_init__(self) -> None:
        if not self.time_ranges:
            raise ValueError("time_ranges không được rỗng.")
        if not self.replace_uids:
            raise ValueError("replace_uids không được rỗng.")
        if self.merge_window_sec < 0:
            raise ValueError(
                f"merge_window_sec phải >= 0, nhận {self.merge_window_sec}."
            )

    @property
    def total_duration_sec(self) -> float:
        """Tổng thời lượng các khoảng (sau khi gộp)."""
        merged = _merge_overlapping_ranges(
            list(self.time_ranges), self.merge_window_sec
        )
        return sum(r.duration_sec for r in merged)


@dataclass(frozen=True, slots=True)
class ReOcrResponse:
    """Kết quả Re-OCR.

    Attributes:
        new_events:        Danh sách SubtitleEvent mới (chưa được merge
                           vào timeline gốc).
        replaced_uids:     UID của các event sẽ bị thay thế (copy từ
                           request — caller dùng để gọi
                           :meth:`replace_events_by_uid`).
        elapsed_seconds:   Thời gian xử lý.
        frames_processed:  Tổng số khung hình đã OCR.
        ranges_processed:  Số time range thực tế đã chạy (sau khi gộp).
        was_cancelled:     True nếu người dùng huỷ giữa chừng. Khi True, caller
                           KHÔNG được áp dụng thay thế (xóa ``replaced_uids``)
                           vì có thể còn range chưa quét → xóa nhầm phụ đề gốc
                           mà không có bản thay thế (mất dữ liệu).
    """

    new_events: list[SubtitleEvent]
    replaced_uids: list[str]
    elapsed_seconds: float
    frames_processed: int
    ranges_processed: int = 0
    was_cancelled: bool = False


def _merge_overlapping_ranges(
    ranges: list[TimeRange], merge_window_sec: float
) -> list[TimeRange]:
    """Gộp các khoảng giao nhau hoặc gần nhau (≤ ``merge_window_sec``).

    Args:
        ranges:           Danh sách khoảng (không cần sắp xếp).
        merge_window_sec: Khoảng cách tối đa giữa 2 range liền kề để
                          còn được gộp. ``0`` = chỉ gộp khi overlap thật.

    Returns:
        Danh sách khoảng đã gộp, sắp xếp theo ``start_sec``.
    """
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda r: r.start_sec)
    merged: list[TimeRange] = [sorted_ranges[0]]

    for current in sorted_ranges[1:]:
        last = merged[-1]
        # Có thể gộp nếu giao nhau HOẶC khoảng cách ≤ merge_window_sec.
        if current.start_sec <= last.end_sec + merge_window_sec:
            merged[-1] = TimeRange(
                start_sec=last.start_sec,
                end_sec=max(last.end_sec, current.end_sec),
            )
        else:
            merged.append(current)

    return merged


__all__ = [
    "ReOcrRequest",
    "ReOcrResponse",
    "TimeRange",
    "_merge_overlapping_ranges",
]
