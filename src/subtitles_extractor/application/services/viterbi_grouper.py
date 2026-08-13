"""Viterbi-style grouper — Tối ưu hoá việc gộp frame thành câu phụ đề bằng Quy hoạch động.

CẢI TIẾN:
    1. Tối ưu hóa List Slicing O(1) cho Y-Spread Penalty.
    2. Boundary Penalty: Phạt việc cắt đôi một câu khi hai frame giống hệt nhau.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import rapidfuzz.fuzz as fuzz

from subtitles_extractor.application.services.cjk_utils import contains_cjk
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult

logger = logging.getLogger(__name__)

# Hằng số Phạt lệch dòng Y-Spread
_Y_SPREAD_TOLERANCE_PX: float = 50.0
_Y_SPREAD_PENALTY_PER_PX: float = 0.01


@dataclass(frozen=True, slots=True)
class ViterbiGrouperConfig:
    """Cấu hình tham số cho thuật toán Viterbi Grouper."""

    open_penalty: float = 0.35
    max_gap_sec: float = 1.0
    max_lookback: int = 30
    min_similarity_to_join: float = 0.40
    sample_step_sec: float = 0.05
    y_clustering_tolerance_ratio: float = 0.30
    y_clustering_tolerance_min_px: float = 5.0


def get_frame_y_center(frame: OcrFrameResult) -> float:
    if not frame.text_boxes:
        return 0.0
    largest_box = max(
        (box for box in frame.text_boxes if box.bounding_box is not None),
        key=lambda b: (b.bounding_box[2] - b.bounding_box[0]) * (b.bounding_box[3] - b.bounding_box[1]),
        default=None
    )
    if largest_box and largest_box.bounding_box:
        return (largest_box.bounding_box[1] + largest_box.bounding_box[3]) / 2.0
    return 0.0


@lru_cache(maxsize=8192)
def viterbi_similarity(text_a: str, text_b: str) -> float:
    """Tính độ tương đồng chuyên biệt cho Viterbi (Có bộ đệm Cache siêu tốc và RapidFuzz)."""
    if text_a == text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0

    if text_a > text_b:
        text_a, text_b = text_b, text_a

    len_a, len_b = len(text_a), len(text_b)
    max_len = max(len_a, len_b)

    if len_a > 2 * len_b or len_b > 2 * len_a:
        if max_len > 4:
            return 0.0

    base_ratio = fuzz.ratio(text_a, text_b, score_cutoff=30.0) / 100.0

    if base_ratio == 0.0:
        return 0.0

    length_diff = abs(len_a - len_b)
    if length_diff > 0:
        diff_ratio = length_diff / max_len

        if max_len <= 3:
            penalty = diff_ratio * 1.2
        elif max_len <= 5:
            penalty = diff_ratio * 0.8
        else:
            penalty = diff_ratio * 0.60

        base_ratio -= penalty
    else:
        is_cjk = contains_cjk(text_a) or contains_cjk(text_b)
        if max_len <= 3:
            if is_cjk:
                if base_ratio >= 0.66:
                    base_ratio = max(base_ratio, 0.85)
            else:
                if base_ratio >= 0.45:
                    base_ratio = max(base_ratio, 0.85)
        elif base_ratio >= 0.60:
            base_ratio += 0.10

    return max(0.0, min(1.0, base_ratio))


class ViterbiGrouper:
    """Dựng câu phụ đề sử dụng thuật toán Viterbi (Dynamic Programming)."""

    def __init__(self, config: ViterbiGrouperConfig) -> None:
        self._config = config

    def group(self, frames_sequence: Sequence[OcrFrameResult]) -> list[int]:
        total_frames = len(frames_sequence)
        if total_frames == 0:
            return []
        if total_frames == 1:
            return [0]

        viterbi_similarity.cache_clear()
        try:
            return self._execute_dynamic_programming(frames_sequence, total_frames)
        finally:
            viterbi_similarity.cache_clear()

    def _execute_dynamic_programming(
        self, frames_sequence: Sequence[OcrFrameResult], total_frames: int
    ) -> list[int]:
        extracted_texts = [
            frame.get_joined_text(
                self._config.y_clustering_tolerance_ratio,
                self._config.y_clustering_tolerance_min_px,
            )
            for frame in frames_sequence
        ]
        timestamps_seconds = [frame.timestamp_sec for frame in frames_sequence]
        confidence_scores = [float(frame.mean_confidence) for frame in frames_sequence]
        y_centers = [get_frame_y_center(f) for f in frames_sequence]

        dp_cost_array: list[float] = [float("inf")] * total_frames
        dp_parent_array: list[int] = [-1] * total_frames
        dp_cost_array[0] = self._config.open_penalty

        max_lookback = self._config.max_lookback
        maximum_gap_duration = self._config.max_gap_sec
        minimum_similarity = self._config.min_similarity_to_join
        penalty_for_new_group = self._config.open_penalty
        expected_step_duration = self._config.sample_step_sec

        for current_frame_idx in range(1, total_frames):
            optimal_cost = float("inf")
            optimal_parent_idx = current_frame_idx - 1

            lookback_boundary = max(-1, current_frame_idx - max_lookback - 1)

            for parent_candidate_idx in range(
                current_frame_idx - 1, lookback_boundary - 1, -1
            ):
                best_anchor_idx = current_frame_idx
                highest_observed_confidence = -1.0

                for probe_idx in range(parent_candidate_idx + 1, current_frame_idx + 1):
                    if (
                        confidence_scores[probe_idx] > highest_observed_confidence
                        and len(extracted_texts[probe_idx].strip()) > 0
                    ):
                        highest_observed_confidence = confidence_scores[probe_idx]
                        best_anchor_idx = probe_idx

                anchor_text = extracted_texts[best_anchor_idx]
                accumulated_error = 0.0
                is_valid_cluster_hypothesis = True

                for probe_idx in range(parent_candidate_idx + 1, current_frame_idx + 1):
                    if probe_idx > parent_candidate_idx + 1:
                        gap_duration = (
                            timestamps_seconds[probe_idx]
                            - timestamps_seconds[probe_idx - 1]
                        )
                        if gap_duration > maximum_gap_duration:
                            is_valid_cluster_hypothesis = False
                            break

                    similarity_score = viterbi_similarity(
                        extracted_texts[probe_idx], anchor_text
                    )
                    gap_to_anchor_frame = abs(
                        timestamps_seconds[probe_idx]
                        - timestamps_seconds[best_anchor_idx]
                    )
                    is_continuous_flow = gap_to_anchor_frame <= (
                        expected_step_duration * 1.5
                    )

                    if (
                        is_continuous_flow
                        and confidence_scores[probe_idx]
                        < confidence_scores[best_anchor_idx]
                    ):
                        error_weight_multiplier = 0.5
                    else:
                        error_weight_multiplier = 1.0

                    if (
                        similarity_score < minimum_similarity
                        and error_weight_multiplier == 1.0
                    ):
                        is_valid_cluster_hypothesis = False
                        break

                    accumulated_error += (1.0 - similarity_score) * error_weight_multiplier

                if not is_valid_cluster_hypothesis:
                    break

                # Tối ưu List Slicing O(1)
                cluster_y_centers = y_centers[parent_candidate_idx + 1 : current_frame_idx + 1]
                valid_ys = [y for y in cluster_y_centers if y > 0.0]

                position_penalty = 0.0
                if valid_ys:
                    y_spread = max(valid_ys) - min(valid_ys)
                    if y_spread > _Y_SPREAD_TOLERANCE_PX:
                        position_penalty = (y_spread - _Y_SPREAD_TOLERANCE_PX) * _Y_SPREAD_PENALTY_PER_PX

                # Logic Boundary Penalty:
                # Thuế cao nếu cắt đôi 2 frame giống hệt nhau.
                boundary_penalty = 0.0
                if parent_candidate_idx >= 0:
                    prev_frame_text = extracted_texts[parent_candidate_idx]
                    first_frame_text = extracted_texts[parent_candidate_idx + 1]
                    if prev_frame_text and first_frame_text:
                        boundary_sim = viterbi_similarity(prev_frame_text, first_frame_text)
                        boundary_penalty = boundary_sim * 0.5

                cluster_error_total = accumulated_error

                prefix_historical_cost = (
                    dp_cost_array[parent_candidate_idx]
                    if parent_candidate_idx >= 0
                    else 0.0
                )

                total_hypothesis_cost = (
                    prefix_historical_cost
                    + cluster_error_total
                    + penalty_for_new_group
                    + position_penalty
                    + boundary_penalty
                )

                if total_hypothesis_cost < optimal_cost:
                    optimal_cost = total_hypothesis_cost
                    optimal_parent_idx = parent_candidate_idx

            dp_cost_array[current_frame_idx] = optimal_cost
            dp_parent_array[current_frame_idx] = optimal_parent_idx

        return self._reconstruct_labels(dp_parent_array, total_frames)

    @staticmethod
    def _reconstruct_labels(
        dp_parent_array: list[int], total_frames: int
    ) -> list[int]:
        computed_intervals: list[tuple[int, int]] = []
        backtrack_idx = total_frames - 1

        while backtrack_idx >= 0:
            parent_link = dp_parent_array[backtrack_idx]
            computed_intervals.append((parent_link + 1, backtrack_idx))
            backtrack_idx = parent_link

        computed_intervals.reverse()

        final_labels = [0] * total_frames
        for cluster_id, (start_idx, end_idx) in enumerate(computed_intervals):
            for fill_idx in range(start_idx, end_idx + 1):
                final_labels[fill_idx] = cluster_id

        return final_labels


__all__ = ["ViterbiGrouper", "ViterbiGrouperConfig", "viterbi_similarity"]
