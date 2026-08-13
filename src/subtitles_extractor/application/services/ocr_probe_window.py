"""Tính cửa sổ thời gian cho "Thử nhanh" OCR (thuần, không phụ thuộc giao diện).

VÌ SAO cần
==========
Trích xuất OCR một tập phim dài 40 phút mất rất nhiều thời gian. Nếu vùng ROI vẽ sai,
chọn nhầm ngôn ngữ, hay tham số chưa hợp, người dùng chỉ phát hiện **sau khi đã chạy
xong** — mất toi hàng chục phút rồi phải làm lại.

"Thử nhanh" chạy đúng pipeline hiện có nhưng **bó vào một cửa sổ ngắn** (mặc định 60
giây), cho kết quả trong khoảng chục giây để kiểm ROI/ngôn ngữ/tham số trước khi chạy
cả tập.

Cách thực hiện
--------------
Tái dùng ``FrameSamplingConfig.skip_intro_sec`` / ``skip_outro_sec`` để giới hạn phạm
vi lấy mẫu — KHÔNG cần thêm đường dẫn code mới trong pipeline, nên kết quả thử đúng
bằng kết quả chạy thật trên đoạn đó.

Chọn đoạn nào?
--------------
Mặc định lấy **giữa phim**: đầu phim thường là intro/logo không có thoại, cuối phim là
credits — thử ở đó dễ ra kết quả rỗng gây hiểu nhầm là "OCR hỏng".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)

#: Độ dài cửa sổ thử mặc định (giây).
DEFAULT_PROBE_SECONDS: Final[float] = 60.0

#: Video ngắn hơn mức này thì thử toàn bộ, không cần cắt cửa sổ.
_MIN_USEFUL_DURATION: Final[float] = 5.0


@dataclass(frozen=True, slots=True)
class ProbeWindow:
    """Cửa sổ thời gian dùng cho lần thử nhanh.

    Attributes:
        start_sec: Mốc bắt đầu (giây) tính từ đầu video.
        end_sec: Mốc kết thúc (giây).
        skip_intro_sec: Giá trị gán cho ``FrameSamplingConfig.skip_intro_sec``.
        skip_outro_sec: Giá trị gán cho ``FrameSamplingConfig.skip_outro_sec``.
    """

    start_sec: float
    end_sec: float
    skip_intro_sec: float
    skip_outro_sec: float

    @property
    def length_sec(self) -> float:
        """Độ dài cửa sổ (giây)."""
        return max(0.0, self.end_sec - self.start_sec)

    @property
    def label_vi(self) -> str:
        """Mô tả ngắn để hiển thị, vd ``"từ phút 20:30 đến 21:30"``."""
        return f"từ {_format_timestamp(self.start_sec)} đến {_format_timestamp(self.end_sec)}"


def _format_timestamp(seconds: float) -> str:
    """Định dạng giây thành ``mm:ss`` (hoặc ``h:mm:ss`` nếu dài hơn một giờ)."""
    total = int(max(0.0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def compute_probe_window(
    duration_sec: float,
    *,
    probe_seconds: float = DEFAULT_PROBE_SECONDS,
    center_ratio: float = 0.5,
) -> ProbeWindow:
    """Tính cửa sổ thử nhanh nằm giữa video.

    Args:
        duration_sec: Tổng thời lượng video (giây).
        probe_seconds: Độ dài cửa sổ mong muốn.
        center_ratio: Vị trí tâm cửa sổ theo tỉ lệ ``0.0``–``1.0`` (mặc định giữa phim).

    Returns:
        :class:`ProbeWindow` đã kẹp vào khoảng hợp lệ. Nếu video ngắn hơn cửa sổ yêu
        cầu, trả về cửa sổ phủ TOÀN BỘ video (``skip_intro`` và ``skip_outro`` bằng 0).

    Notes:
        Luôn đảm bảo ``skip_intro_sec + skip_outro_sec < duration_sec`` — nếu không,
        bộ lấy mẫu sẽ không còn khung hình nào để xử lý và lần thử ra kết quả rỗng.
    """
    duration = max(0.0, float(duration_sec))
    if duration <= _MIN_USEFUL_DURATION:
        # Video quá ngắn: thử toàn bộ, cắt thêm chỉ tổ mất dữ liệu.
        return ProbeWindow(0.0, duration, 0.0, 0.0)

    window = max(1.0, min(float(probe_seconds), duration))
    if window >= duration:
        return ProbeWindow(0.0, duration, 0.0, 0.0)

    ratio = min(1.0, max(0.0, float(center_ratio)))
    center = duration * ratio
    start = center - window / 2.0

    # Kẹp cửa sổ vào trong video, giữ nguyên độ dài.
    start = max(0.0, min(start, duration - window))
    end = start + window

    return ProbeWindow(
        start_sec=start,
        end_sec=end,
        skip_intro_sec=start,
        skip_outro_sec=max(0.0, duration - end),
    )


def summarise_probe_result(
    event_count: int, window: ProbeWindow, sample_texts: list[str]
) -> str:
    """Tóm tắt kết quả thử nhanh thành thông điệp cho người dùng.

    Args:
        event_count: Số câu phụ đề dựng được trong cửa sổ.
        window: Cửa sổ đã thử.
        sample_texts: Vài câu đầu để người dùng đối chiếu với hình.

    Returns:
        Chuỗi nhiều dòng, nêu rõ cần làm gì tiếp nếu kết quả bất thường.
    """
    header = f"Thử nhanh {window.length_sec:.0f} giây ({window.label_vi})"
    if event_count == 0:
        return (
            f"{header}\n\n"
            "KHÔNG nhận được câu nào. Nguyên nhân thường gặp:\n"
            "• Vùng ROI chưa trùng vị trí phụ đề — vẽ lại hoặc bấm “Tự dò ROI”.\n"
            "• Sai ngôn ngữ OCR — kiểm trong Cài đặt.\n"
            "• Đoạn thử rơi vào chỗ không có thoại — thử đoạn khác.\n"
            "• Phim không có phụ đề cháy — dùng “Quét track phụ đề nhúng”."
        )

    preview = "\n".join(f"• {text}" for text in sample_texts[:5])
    tail = "" if len(sample_texts) <= 5 else f"\n… và {len(sample_texts) - 5} câu nữa."
    return (
        f"{header}\n\nNhận được {event_count} câu. Vài câu đầu:\n{preview}{tail}\n\n"
        "Nếu chữ đọc đúng, hãy chạy trích xuất đầy đủ."
    )


__all__ = [
    "DEFAULT_PROBE_SECONDS",
    "ProbeWindow",
    "compute_probe_window",
    "summarise_probe_result",
]
