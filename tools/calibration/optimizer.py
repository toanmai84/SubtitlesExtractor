"""Bộ tối ưu tham số bằng coordinate-descent có tinh chỉnh đệ quy.

Ý tưởng cốt lõi (đệ quy): ở mỗi *cấp* (level), quét lần lượt từng tham số trên
một lưới rời rạc và giữ lại giá trị tốt nhất (coordinate descent). Khi một cấp hội
tụ, ta **thu hẹp biên độ lưới quanh điểm tốt nhất** rồi gọi đệ quy xuống cấp sâu
hơn để dò mịn hơn. Quá trình dừng khi hết độ sâu hoặc không còn cải thiện đáng kể.

Bộ tối ưu hoàn toàn *agnostic* với bài toán: nó chỉ cần một hàm mục tiêu
``objective(assignment) -> float`` (càng cao càng tốt). Nhờ vậy cùng một optimizer
phục vụ cả hiệu chuẩn build phụ đề lẫn hiệu chuẩn ROI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from loguru import logger

from tools.calibration.exceptions import ObjectiveEvaluationError
from tools.calibration.search_space import SearchSpace

ObjectiveFunction = Callable[[dict[str, float]], float]


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """Một lần đánh giá hàm mục tiêu (cho nhật ký/tái lập)."""

    assignment: dict[str, float | int | bool]
    score: float
    level: int


@dataclass(slots=True)
class OptimizationResult:
    """Kết quả tối ưu.

    Attributes:
        best_assignment: Bộ tham số tốt nhất (đã ép kiểu).
        best_score: Điểm mục tiêu tốt nhất.
        evaluations: Số lần gọi hàm mục tiêu (đã trừ cache-hit).
        history: Toàn bộ bản ghi đánh giá (gồm cả cache-hit) theo thời gian.
    """

    best_assignment: dict[str, float | int | bool]
    best_score: float
    evaluations: int = 0
    history: list[EvaluationRecord] = field(default_factory=list)


class RecursiveCoordinateDescentOptimizer:
    """Coordinate-descent với tinh chỉnh đệ quy quanh điểm tốt nhất.

    Args:
        search_space: Không gian tham số.
        objective: Hàm mục tiêu cần CỰC ĐẠI HOÁ.
        max_depth: Số cấp tinh chỉnh đệ quy (mỗi cấp thu hẹp biên lưới).
        span_shrink: Hệ số co biên mỗi cấp (∈ (0, 1)); vd 0.4.
        max_sweeps_per_level: Số vòng quét coordinate-descent tối đa mỗi cấp.
        improvement_epsilon: Mức cải thiện tối thiểu mới coi là "tốt hơn".
        max_evaluations: Trần số lần đánh giá (an toàn ngân sách); None = không giới hạn.
        time_budget_sec: Trần thời gian (giây); None = không giới hạn.
    """

    def __init__(
        self,
        *,
        search_space: SearchSpace,
        objective: ObjectiveFunction,
        max_depth: int = 2,
        span_shrink: float = 0.4,
        max_sweeps_per_level: int = 2,
        improvement_epsilon: float = 1e-4,
        max_evaluations: int | None = None,
        time_budget_sec: float | None = None,
    ) -> None:
        self._search_space = search_space
        self._objective = objective
        self._max_depth = max_depth
        self._span_shrink = span_shrink
        self._max_sweeps_per_level = max_sweeps_per_level
        self._improvement_epsilon = improvement_epsilon
        self._max_evaluations = max_evaluations
        self._time_budget_sec = time_budget_sec

        self._cache: dict[tuple[tuple[str, float | int | bool], ...], float] = {}
        self._history: list[EvaluationRecord] = []
        self._started_at: float = 0.0

    def optimize(self, initial_assignment: dict[str, float] | None = None) -> OptimizationResult:
        """Chạy tối ưu từ điểm khởi đầu (mặc định = trung điểm không gian).

        Returns:
            :class:`OptimizationResult` chứa bộ tham số và điểm tốt nhất.
        """
        self._started_at = time.monotonic()
        start_assignment = dict(
            initial_assignment or self._search_space.midpoint_assignment()
        )
        start_score = self._evaluate(start_assignment, level=0)
        best_assignment, best_score = self._descend(
            center=start_assignment, span_ratio=1.0, level=0,
            best_assignment=start_assignment, best_score=start_score,
        )
        return OptimizationResult(
            best_assignment=self._search_space.coerce_assignment(best_assignment),
            best_score=best_score,
            evaluations=len(self._cache),
            history=list(self._history),
        )

    def _descend(
        self,
        *,
        center: dict[str, float],
        span_ratio: float,
        level: int,
        best_assignment: dict[str, float],
        best_score: float,
    ) -> tuple[dict[str, float], float]:
        """Xử lý MỘT cấp coordinate-descent rồi đệ quy xuống cấp mịn hơn."""
        level_entry_score = best_score
        current = dict(best_assignment)

        for sweep_index in range(self._max_sweeps_per_level):
            improved_in_sweep = False
            for spec in self._search_space.specs:
                if self._budget_exhausted():
                    logger.warning("Hết ngân sách tối ưu — dừng sớm ở cấp {}.", level)
                    return current, best_score
                grid_values = spec.grid(center=current[spec.name], span_ratio=span_ratio)
                for grid_value in grid_values:
                    candidate = dict(current)
                    candidate[spec.name] = grid_value
                    candidate_score = self._evaluate(candidate, level=level)
                    if candidate_score > best_score + self._improvement_epsilon:
                        best_score = candidate_score
                        current = candidate
                        improved_in_sweep = True
            logger.debug(
                "Cấp {} | quét {} | best={:.5f}", level, sweep_index, best_score
            )
            if not improved_in_sweep:
                break

        gained = best_score - level_entry_score
        if level < self._max_depth and gained > self._improvement_epsilon:
            logger.info(
                "Tinh chỉnh đệ quy → cấp {} (thu hẹp biên ×{}), best hiện tại={:.5f}",
                level + 1, self._span_shrink, best_score,
            )
            return self._descend(
                center=current,
                span_ratio=span_ratio * self._span_shrink,
                level=level + 1,
                best_assignment=current,
                best_score=best_score,
            )
        return current, best_score

    def _evaluate(self, assignment: dict[str, float], level: int) -> float:
        """Đánh giá hàm mục tiêu với cache theo bộ tham số đã ép kiểu."""
        coerced = self._search_space.coerce_assignment(assignment)
        cache_key = tuple(sorted(coerced.items()))
        if cache_key in self._cache:
            score = self._cache[cache_key]
            self._history.append(EvaluationRecord(coerced, score, level))
            return score
        try:
            score = float(self._objective(dict(coerced)))
        except (ValueError, TypeError, KeyError) as evaluation_error:
            raise ObjectiveEvaluationError(
                f"Lỗi đánh giá mục tiêu cho {coerced}: {evaluation_error}"
            ) from evaluation_error
        self._cache[cache_key] = score
        self._history.append(EvaluationRecord(coerced, score, level))
        return score

    def _budget_exhausted(self) -> bool:
        if self._max_evaluations is not None and len(self._cache) >= self._max_evaluations:
            return True
        if self._time_budget_sec is not None:
            if time.monotonic() - self._started_at >= self._time_budget_sec:
                return True
        return False
