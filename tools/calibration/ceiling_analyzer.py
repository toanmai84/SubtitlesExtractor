"""Phân tích TRẦN KHẢ THI (oracle ceiling) cho engine dựng phụ đề.

Trả lời câu hỏi cốt lõi: *"Engine còn cách 100% giống phụ đề chuẩn bao xa, và phần
chênh lệch là do ENGINE hay do GIỚI HẠN của OCR thô / phụ đề chuẩn không chuẩn?"*

Với mỗi câu ground-truth, ta phân loại:
    * ``achieved``      — engine đã dựng khớp 100% (sau chuẩn hoá).
    * ``recoverable``   — engine TRƯỢT nhưng văn bản đúng CÓ xuất hiện trong OCR thô
      ở cửa sổ thời gian đó → **engine còn sửa được**.
    * ``input_limited`` — không frame OCR nào chứa văn bản đúng → **giới hạn input**,
      không engine nào dựng được (trừ khi chạy lại OCR tốt hơn).

Tuỳ chọn ``traditional_as_simplified`` coi biến thể phồn/giản là tương đương (cùng
một từ), cho ra "điểm nội dung công bằng" khi phụ đề chuẩn dùng phồn-giản lẫn lộn.

Module thuần (không Qt/GPU), nhận hàm nạp OCR + builder qua DI để dễ test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

try:
    import zhconv  # type: ignore

    def _to_simplified(text: str) -> str:
        return zhconv.convert(text, "zh-cn")

except ImportError:  # pragma: no cover - zhconv là tuỳ chọn
    def _to_simplified(text: str) -> str:
        return text


_PUNCTUATION_PATTERN = re.compile(
    r"[\s\u3000,.，。！？、：；…—\-！?!~·「」『』\"'()（）]+"
)


def _normalize(text: str) -> str:
    """Bỏ khoảng trắng + dấu câu để so khớp nội dung."""
    return _PUNCTUATION_PATTERN.sub("", text)


@dataclass(frozen=True, slots=True)
class CeilingBreakdown:
    """Kết quả phân rã trần khả thi trên một corpus.

    Attributes:
        total_cues: Tổng số câu ground-truth (sau chuẩn hoá khác rỗng).
        achieved: Số câu engine dựng khớp 100%.
        recoverable: Số câu engine trượt nhưng OCR có chứa văn bản đúng.
        input_limited: Số câu không OCR frame nào chứa văn bản đúng.
    """

    total_cues: int
    achieved: int
    recoverable: int
    input_limited: int

    @property
    def achieved_rate(self) -> float:
        return self.achieved / self.total_cues if self.total_cues else 0.0

    @property
    def ceiling_rate(self) -> float:
        """Tỷ lệ exact-match TỐI ĐA có thể đạt với OCR thô hiện có."""
        if not self.total_cues:
            return 0.0
        return (self.achieved + self.recoverable) / self.total_cues

    @property
    def input_limited_rate(self) -> float:
        return self.input_limited / self.total_cues if self.total_cues else 0.0

    @property
    def engine_gap_rate(self) -> float:
        """Phần engine còn có thể cải thiện (recoverable / total)."""
        return self.recoverable / self.total_cues if self.total_cues else 0.0


def _collect_window_candidates(
    frames: Sequence[object],
    start_sec: float,
    end_sec: float,
    padding_sec: float,
    simplify: bool,
) -> set[str]:
    """Tập văn bản ứng viên (đã chuẩn hoá) từ các frame phủ cửa sổ thời gian."""
    candidates: set[str] = set()
    for frame in frames:
        timestamp = float(getattr(frame, "timestamp_sec", 0.0))
        if start_sec - padding_sec <= timestamp <= end_sec + padding_sec:
            text_boxes = getattr(frame, "text_boxes", [])
            for box in text_boxes:
                candidates.add(_norm_variant(box.text, simplify))
            joined = "".join(box.text for box in text_boxes)
            candidates.add(_norm_variant(joined, simplify))
            getter = getattr(frame, "get_joined_text", None)
            if callable(getter):
                candidates.add(_norm_variant(getter(), simplify))
    candidates.discard("")
    return candidates


def _norm_variant(text: str, simplify: bool) -> str:
    normalized = _normalize(text)
    return _to_simplified(normalized) if simplify else normalized


def analyze_ceiling(
    *,
    ground_truth: list[tuple[float, float, str]],
    built_events: list[tuple[float, float, str]],
    frames: Sequence[object],
    window_padding_sec: float = 0.30,
    min_overlap_sec: float = 0.05,
    traditional_as_simplified: bool = False,
) -> CeilingBreakdown:
    """Phân rã trần khả thi cho một dataset.

    Args:
        ground_truth: Danh sách ``(start_sec, end_sec, text)`` phụ đề chuẩn.
        built_events: Danh sách ``(start_sec, end_sec, text)`` engine dựng.
        frames: Chuỗi frame OCR thô (mỗi frame có ``timestamp_sec`` + ``text_boxes``).
        window_padding_sec: Nới cửa sổ thời gian khi gom ứng viên OCR.
        min_overlap_sec: Ngưỡng chồng lấn để ghép câu built với câu GT.
        traditional_as_simplified: Coi phồn/giản tương đương khi so khớp.

    Returns:
        :class:`CeilingBreakdown`.
    """
    simplify = traditional_as_simplified
    achieved = recoverable = input_limited = 0
    total = 0

    for gt_start, gt_end, gt_text in ground_truth:
        normalized_gt = _norm_variant(gt_text, simplify)
        if not normalized_gt:
            continue
        total += 1

        best_index = -1
        best_overlap = min_overlap_sec
        for index, (built_start, built_end, _text) in enumerate(built_events):
            overlap = min(gt_end, built_end) - max(gt_start, built_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index

        built_norm = (
            _norm_variant(built_events[best_index][2], simplify)
            if best_index >= 0
            else ""
        )
        if built_norm == normalized_gt:
            achieved += 1
            continue

        candidates = _collect_window_candidates(
            frames, gt_start, gt_end, window_padding_sec, simplify
        )
        if normalized_gt in candidates:
            recoverable += 1
        else:
            input_limited += 1

    return CeilingBreakdown(
        total_cues=total,
        achieved=achieved,
        recoverable=recoverable,
        input_limited=input_limited,
    )


def aggregate_breakdowns(breakdowns: Sequence[CeilingBreakdown]) -> CeilingBreakdown:
    """Cộng dồn nhiều :class:`CeilingBreakdown` thành một (toàn corpus)."""
    return CeilingBreakdown(
        total_cues=sum(b.total_cues for b in breakdowns),
        achieved=sum(b.achieved for b in breakdowns),
        recoverable=sum(b.recoverable for b in breakdowns),
        input_limited=sum(b.input_limited for b in breakdowns),
    )
