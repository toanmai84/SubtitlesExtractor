"""Các hàm đo lường thuần (pure functions) cho hiệu chuẩn phụ đề.

Mọi hàm ở đây là *pure*: kết quả chỉ phụ thuộc tham số đầu vào, không thay đổi
trạng thái bên ngoài, không I/O — nhờ vậy cực kỳ dễ unit-test và mock.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Khoảng ký tự cần loại khi chuẩn hoá so khớp CJK: khoảng trắng + dấu câu.
_PUNCTUATION_PATTERN = re.compile(r"[\s\u3000,.，。！？、：；…—\-！?!~·「」『』\"'()（）]+")


def normalize_for_match(text: str) -> str:
    """Chuẩn hoá chuỗi để so khớp: bỏ khoảng trắng và dấu câu.

    Args:
        text: Chuỗi gốc.

    Returns:
        Chuỗi chỉ còn ký tự nội dung (chữ Hán/chữ cái/số).
    """
    return _PUNCTUATION_PATTERN.sub("", text)


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    """Khoảng cách Levenshtein giữa hai chuỗi (quy hoạch động 1 hàng).

    Args:
        reference: Chuỗi tham chiếu.
        hypothesis: Chuỗi giả thuyết.

    Returns:
        Số phép chèn/xoá/thay tối thiểu để biến ``reference`` thành ``hypothesis``.
    """
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous_row = list(range(len(hypothesis) + 1))
    for ref_index, ref_char in enumerate(reference, start=1):
        current_diagonal = previous_row[0]
        previous_row[0] = ref_index
        for hyp_index, hyp_char in enumerate(hypothesis, start=1):
            insert_cost = previous_row[hyp_index] + 1
            delete_cost = previous_row[hyp_index - 1] + 1
            replace_cost = current_diagonal + (ref_char != hyp_char)
            current_diagonal = previous_row[hyp_index]
            previous_row[hyp_index] = min(insert_cost, delete_cost, replace_cost)
    return previous_row[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Tỷ lệ lỗi ký tự (CER) sau khi chuẩn hoá.

    Args:
        reference: Phụ đề chuẩn (ground-truth).
        hypothesis: Phụ đề do pipeline dựng.

    Returns:
        ``levenshtein / len(reference)`` trong [0, +∞); 0.0 là khớp hoàn hảo.
    """
    normalized_reference = normalize_for_match(reference)
    normalized_hypothesis = normalize_for_match(hypothesis)
    if not normalized_reference:
        return 0.0 if not normalized_hypothesis else 1.0
    distance = levenshtein_distance(normalized_reference, normalized_hypothesis)
    return distance / len(normalized_reference)


def temporal_overlap_sec(
    first_interval: tuple[float, float],
    second_interval: tuple[float, float],
) -> float:
    """Độ chồng lấn thời gian (giây) giữa hai khoảng ``(start, end)``.

    Args:
        first_interval: Khoảng thứ nhất.
        second_interval: Khoảng thứ hai.

    Returns:
        Số giây chồng lấn, tối thiểu 0.0.
    """
    overlap = min(first_interval[1], second_interval[1]) - max(
        first_interval[0], second_interval[0]
    )
    return max(0.0, overlap)


@dataclass(frozen=True, slots=True)
class SubtitleScore:
    """Bộ chỉ số chất lượng build phụ đề so với ground-truth (bất biến).

    Attributes:
        ground_truth_count: Số câu trong ground-truth.
        built_count: Số câu pipeline dựng được.
        matched_count: Số câu GT khớp được (theo thời gian) với một câu built.
        exact_count: Số câu khớp 100% nội dung sau chuẩn hoá.
        spurious_count: Số câu built không khớp GT nào (dương tính giả).
        missing_count: Số câu GT không khớp built nào (âm tính giả).
        average_cer: CER trung bình trên toàn bộ câu GT.
    """

    ground_truth_count: int
    built_count: int
    matched_count: int
    exact_count: int
    spurious_count: int
    missing_count: int
    average_cer: float

    @property
    def recall(self) -> float:
        """Tỷ lệ câu GT được khớp."""
        if self.ground_truth_count == 0:
            return 0.0
        return self.matched_count / self.ground_truth_count

    @property
    def exact_match_rate(self) -> float:
        """Tỷ lệ câu GT được dựng khớp 100% nội dung."""
        if self.ground_truth_count == 0:
            return 0.0
        return self.exact_count / self.ground_truth_count

    @property
    def spurious_rate(self) -> float:
        """Tỷ lệ câu built là dương tính giả."""
        if self.built_count == 0:
            return 0.0
        return self.spurious_count / self.built_count

    @property
    def quality(self) -> float:
        """Điểm vô hướng tổng hợp để tối ưu (càng cao càng tốt, ∈ ~[0, 1]).

        Trọng số ưu tiên *exact-match* (giống ground-truth tuyệt đối) và độ chính
        xác nội dung (1 − CER), có thưởng recall và phạt nhẹ dương-tính-giả.
        """
        raw = (
            0.50 * self.exact_match_rate
            + 0.25 * max(0.0, 1.0 - self.average_cer)
            + 0.20 * self.recall
            - 0.10 * self.spurious_rate
        )
        return max(0.0, min(1.0, raw))


def score_subtitles(
    ground_truth: list[tuple[float, float, str]],
    built: list[tuple[float, float, str]],
    min_overlap_sec: float = 0.05,
) -> SubtitleScore:
    """So khớp phụ đề built với ground-truth theo chồng lấn thời gian và chấm điểm.

    Mỗi câu GT được ghép với câu built chồng lấn thời gian nhiều nhất, sau đó so
    sánh nội dung bằng CER. Câu built không được ghép tính là dương-tính-giả.

    Args:
        ground_truth: Danh sách ``(start_sec, end_sec, text)`` của phụ đề chuẩn.
        built: Danh sách ``(start_sec, end_sec, text)`` do pipeline dựng.
        min_overlap_sec: Ngưỡng chồng lấn tối thiểu để coi là một cặp khớp.

    Returns:
        :class:`SubtitleScore` tổng hợp.
    """
    matched_built_indices: set[int] = set()
    cer_values: list[float] = []
    exact_count = 0
    missing_count = 0

    for gt_interval_start, gt_interval_end, gt_text in ground_truth:
        gt_interval = (gt_interval_start, gt_interval_end)
        best_index = -1
        best_overlap = min_overlap_sec
        for built_index, (built_start, built_end, _) in enumerate(built):
            overlap = temporal_overlap_sec(gt_interval, (built_start, built_end))
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = built_index

        if best_index >= 0:
            matched_built_indices.add(best_index)
            cer = character_error_rate(gt_text, built[best_index][2])
            cer_values.append(cer)
            if cer == 0.0:
                exact_count += 1
        else:
            missing_count += 1
            cer_values.append(1.0)

    average_cer = sum(cer_values) / len(cer_values) if cer_values else 1.0
    return SubtitleScore(
        ground_truth_count=len(ground_truth),
        built_count=len(built),
        matched_count=len(ground_truth) - missing_count,
        exact_count=exact_count,
        spurious_count=len(built) - len(matched_built_indices),
        missing_count=missing_count,
        average_cer=average_cer,
    )
