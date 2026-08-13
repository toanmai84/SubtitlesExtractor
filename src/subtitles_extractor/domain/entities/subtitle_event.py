"""Entity :class:`SubtitleEvent` — một câu phụ đề hoàn chỉnh.

Khác biệt so với phiên bản cũ:
    * Không còn alias tiếng Việt (``start_giay``, ``so_frame`` …).
    * Không còn classmethod ``from_srt_file`` (loader tách sang module riêng,
      tránh phụ thuộc ``pysubs2`` ở tầng domain).
    * Sử dụng :class:`TimeInterval` cho khoảng thời gian thay vì 2 trường rời.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


@dataclass(slots=True)
class SubtitleEvent:
    """Một câu phụ đề hiển thị trong một khoảng thời gian.

    Attributes:
        index:        Số thứ tự (1-based) trong tệp xuất.
        text:         Nội dung; có thể chứa ``"\\n"`` cho ngắt dòng.
        interval:     Khoảng thời gian hiển thị.
        confidence:   Điểm tin cậy trung bình của OCR.
        frame_count:  Số khung hình đã gộp tạo nên event này.
        position:     Toạ độ ``(x, y)`` của box chính (cho overlay), hoặc None.
        uid:          UUID nội bộ — giữ ổn định cho undo/redo và khớp giữa
                      các bản chỉnh sửa.
    """

    index: int
    text: str
    interval: TimeInterval
    confidence: Confidence = field(default_factory=Confidence.zero)
    frame_count: int = 0
    position: tuple[int, int] | None = None
    bounding_box: tuple[int, int, int, int] | None = None
    """Bounding box ``(x_min, y_min, x_max, y_max)`` trong toạ độ video gốc.

    Được populate bởi :class:`SubtitleBuilder` từ tập hợp bbox của tất cả frame
    đóng góp vào event này. Dùng cho OCR overlay trên VideoCanvas.
    ``None`` nếu không có thông tin vị trí (ví dụ: import từ file SRT)."""
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def start_sec(self) -> float:
        """Tiện ích đọc — đại diện cho ``self.interval.start_sec``."""
        return self.interval.start_sec

    @property
    def end_sec(self) -> float:
        """Tiện ích đọc — đại diện cho ``self.interval.end_sec``."""
        return self.interval.end_sec

    @property
    def duration_sec(self) -> float:
        """Thời lượng (giây)."""
        return self.interval.duration_sec

    @property
    def cps(self) -> float:
        """Số ký tự / giây — chỉ số tốc độ đọc.

        Quy ước: < 15 dễ đọc, 15-20 vừa, > 20 quá nhanh người xem
        không kịp đọc.
        """
        if self.duration_sec <= 0:
            return 0.0
        # Bỏ ký tự whitespace + newline khỏi count cho chính xác.
        text_len = len(self.text.replace("\n", "").replace(" ", ""))
        return text_len / self.duration_sec

    @property
    def is_too_short(self) -> bool:
        """Câu phụ đề ngắn hơn 0.5 giây — khó đọc."""
        return 0 < self.duration_sec < 0.5

    @property
    def is_too_fast(self) -> bool:
        """CPS > 20 — quá nhanh để đọc kịp."""
        return self.cps > 20.0


__all__ = ["SubtitleEvent"]
