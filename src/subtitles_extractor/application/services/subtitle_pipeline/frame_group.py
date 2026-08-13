"""Dataclass :class:`FrameGroup` và các utilities trích xuất vị trí không gian.

Một :class:`FrameGroup` đại diện cho **một cụm frame OCR liên tiếp có cùng
nội dung phụ đề**, được tạo ra từ thuật toán greedy/Viterbi grouping. Mỗi
group cuối cùng sẽ được chuyển thành một :class:`SubtitleEvent`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from subtitles_extractor.application.services.subtitle_pipeline.constants import (
    CJK_TRAILING_PUNCTUATIONS,
)
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult


@dataclasses.dataclass(slots=True)
class FrameGroup:
    """Cụm frame liên tiếp đã được gộp thành một câu phụ đề tiềm năng.

    Attributes:
        reconstructed_text:    Text canonical đã được ROVER vote từ các
                               frame thành phần.
        start_timestamp_sec:   Mốc thời gian bắt đầu (giây).
        end_timestamp_sec:     Mốc thời gian kết thúc (giây).
        accumulated_confidence: Tổng confidence trung bình của các frame
                               thành phần (chưa chia số frame).
        total_frames_count:    Số frame đã gộp vào group.
        primary_center_position: Toạ độ tâm ``(x, y)`` của box lớn nhất —
                               dùng cho overlay UI.
        aggregated_bounding_box: BBox ``(xmin, ymin, xmax, ymax)`` bao toàn
                               bộ các box trong group.
    """

    reconstructed_text: str
    start_timestamp_sec: float
    end_timestamp_sec: float
    accumulated_confidence: float = 0.0
    total_frames_count: int = 0
    primary_center_position: tuple[int, int] | None = None
    aggregated_bounding_box: tuple[int, int, int, int] | None = None
    #: [Stable Variant Split] Chuỗi (timestamp, joined_text) theo TỪNG frame thành
    #: viên — cho phép pass hậu kỳ phát hiện group chứa 2+ khối text ổn định
    #: (progressive utterance bị merge nhầm) và tách lại thành nhiều câu.
    member_texts: list[tuple[float, str]] = dataclasses.field(default_factory=list)

    def calculate_mean_confidence(self) -> float:
        """Trả về confidence trung bình của các frame trong group."""
        if self.total_frames_count <= 0:
            return 0.0
        return self.accumulated_confidence / self.total_frames_count

    @property
    def total_score(self) -> float:
        """Tổng score = tổng confidence — dùng so sánh chọn text tốt hơn khi merge."""
        return self.accumulated_confidence


def extract_primary_spatial_position(
    frame_result: OcrFrameResult,
) -> tuple[int, int] | None:
    """Trích xuất toạ độ tâm của box LỚN NHẤT trong frame.

    Args:
        frame_result: Frame OCR cần phân tích.

    Returns:
        Tuple ``(center_x, center_y)`` hoặc ``None`` nếu không có box hợp lệ.
    """
    if not frame_result.text_boxes:
        return None

    largest_text_box = max(
        (box for box in frame_result.text_boxes if box.bounding_box is not None),
        key=lambda current_box: (
            (current_box.bounding_box[2] - current_box.bounding_box[0])
            * (current_box.bounding_box[3] - current_box.bounding_box[1])
        ),
        default=None,
    )

    if largest_text_box is None or largest_text_box.bounding_box is None:
        return None

    x_minimum, y_minimum, x_maximum, y_maximum = largest_text_box.bounding_box
    return ((x_minimum + x_maximum) // 2, (y_minimum + y_maximum) // 2)


def extract_aggregated_bounding_box(
    frames_sequence: Sequence[OcrFrameResult],
) -> tuple[int, int, int, int] | None:
    """Tính bounding box bao toàn bộ box trong chuỗi frame.

    Args:
        frames_sequence: Chuỗi frame OCR.

    Returns:
        Tuple ``(xmin, ymin, xmax, ymax)`` hoặc ``None`` nếu không có
        box nào có polygon.
    """
    valid_bounding_boxes = [
        box.bounding_box
        for current_frame in frames_sequence
        for box in current_frame.text_boxes
        if box.bounding_box is not None
    ]
    if not valid_bounding_boxes:
        return None

    x_minimums, y_minimums, x_maximums, y_maximums = zip(
        *valid_bounding_boxes, strict=True
    )
    return min(x_minimums), min(y_minimums), max(x_maximums), max(y_maximums)


def normalize_cjk_punctuation(source_text: str) -> str:
    """Loại bỏ dấu câu CJK ở 2 đầu chuỗi để chuẩn hoá khi so sánh tương đồng.

    Args:
        source_text: Chuỗi đầu vào.

    Returns:
        Chuỗi đã trim cả dấu câu CJK biên + whitespace.
    """
    stripped_text = source_text.strip()
    while stripped_text and (
        stripped_text[-1] in CJK_TRAILING_PUNCTUATIONS or stripped_text[-1].isspace()
    ):
        stripped_text = stripped_text[:-1]
    while stripped_text and (
        stripped_text[0] in CJK_TRAILING_PUNCTUATIONS or stripped_text[0].isspace()
    ):
        stripped_text = stripped_text[1:]
    return stripped_text.strip()


__all__ = [
    "FrameGroup",
    "extract_aggregated_bounding_box",
    "extract_primary_spatial_position",
    "normalize_cjk_punctuation",
]
