"""Điều phối hiệu chuẩn đệ quy có khả năng *tự cải tiến qua phiên* (warm-start).

Calibrator lưu bộ tham số tốt nhất ra JSON. Mỗi lần chạy lại, nó nạp trạng thái cũ
làm điểm khởi đầu nên kết quả chỉ tốt lên hoặc giữ nguyên — tích lũy cải tiến theo
thời gian. Đây là tầng "tự cải tiến" bao ngoài optimizer "đệ quy".
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger

from tools.calibration.exceptions import CalibrationStateError
from tools.calibration.optimizer import (
    ObjectiveFunction,
    OptimizationResult,
    RecursiveCoordinateDescentOptimizer,
)
from tools.calibration.search_space import SearchSpace


@dataclass(slots=True)
class CalibrationOutcome:
    """Kết quả một phiên hiệu chuẩn.

    Attributes:
        best_assignment: Bộ tham số tốt nhất hiện tại.
        best_score: Điểm tốt nhất.
        baseline_score: Điểm của bộ mặc định (để so cải thiện).
        previous_best_score: Điểm tốt nhất phiên trước (warm-start), nếu có.
        evaluations: Số lần đánh giá hàm mục tiêu trong phiên.
        improved: Phiên này có cải thiện so với trạng thái đã lưu không.
    """

    best_assignment: dict[str, float | int | bool]
    best_score: float
    baseline_score: float
    previous_best_score: float | None
    evaluations: int
    improved: bool
    history_size: int = 0
    extras: dict[str, float] = field(default_factory=dict)


class RecursiveCalibrator:
    """Bao optimizer đệ quy + nạp/lưu trạng thái tốt nhất.

    Args:
        search_space: Không gian tham số.
        objective: Hàm mục tiêu cần cực đại hoá.
        state_path: Nơi lưu/đọc bộ tham số tốt nhất (JSON). ``None`` = không bền hoá.
        optimizer_kwargs: Tham số chuyển tiếp cho optimizer (max_depth, span_shrink...).
    """

    def __init__(
        self,
        *,
        search_space: SearchSpace,
        objective: ObjectiveFunction,
        state_path: Path | None = None,
        optimizer_kwargs: dict[str, float | int] | None = None,
    ) -> None:
        self._search_space = search_space
        self._objective = objective
        self._state_path = state_path
        self._optimizer_kwargs = dict(optimizer_kwargs or {})

    def _load_state(self) -> dict[str, object] | None:
        if self._state_path is None or not self._state_path.is_file():
            return None
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as state_error:
            raise CalibrationStateError(
                f"Không đọc được trạng thái: {self._state_path}"
            ) from state_error

    def _save_state(self, outcome: CalibrationOutcome) -> None:
        if self._state_path is None:
            return
        payload = {
            "best_assignment": outcome.best_assignment,
            "best_score": outcome.best_score,
            "baseline_score": outcome.baseline_score,
            "evaluations": outcome.evaluations,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as state_error:
            raise CalibrationStateError(
                f"Không ghi được trạng thái: {self._state_path}"
            ) from state_error

    def run(self) -> CalibrationOutcome:
        """Chạy một phiên hiệu chuẩn (warm-start nếu có trạng thái cũ)."""
        baseline_assignment = self._search_space.midpoint_assignment()
        baseline_score = float(self._objective(
            dict(self._search_space.coerce_assignment(baseline_assignment))
        ))

        previous_state = self._load_state()
        previous_best_score: float | None = None
        initial_assignment: dict[str, float] = baseline_assignment
        if previous_state is not None:
            previous_best_score = float(previous_state.get("best_score", 0.0))
            stored = previous_state.get("best_assignment", {})
            initial_assignment = {
                name: float(stored[name])
                for name in self._search_space.names()
                if name in stored
            } or baseline_assignment
            logger.info(
                "Warm-start từ trạng thái cũ (best={:.5f}).", previous_best_score
            )

        optimizer = RecursiveCoordinateDescentOptimizer(
            search_space=self._search_space,
            objective=self._objective,
            **self._optimizer_kwargs,
        )
        result: OptimizationResult = optimizer.optimize(initial_assignment)

        improved = (
            previous_best_score is None or result.best_score > previous_best_score
        )
        best_assignment = result.best_assignment
        best_score = result.best_score
        if previous_best_score is not None and not improved:
            # Giữ trạng thái cũ tốt hơn (tự cải tiến: không bao giờ thụt lùi).
            best_assignment = {
                name: previous_state["best_assignment"][name]  # type: ignore[index]
                for name in self._search_space.names()
                if name in previous_state["best_assignment"]  # type: ignore[index]
            }
            best_score = previous_best_score

        outcome = CalibrationOutcome(
            best_assignment=best_assignment,
            best_score=best_score,
            baseline_score=baseline_score,
            previous_best_score=previous_best_score,
            evaluations=result.evaluations,
            improved=improved,
            history_size=len(result.history),
        )
        if improved:
            self._save_state(outcome)
            logger.info("Cải thiện! Đã lưu trạng thái mới (best={:.5f}).", best_score)
        else:
            logger.info("Không cải thiện so với phiên trước — giữ trạng thái cũ.")
        return outcome

    @staticmethod
    def outcome_as_dict(outcome: CalibrationOutcome) -> dict[str, object]:
        """Tiện ích chuyển outcome thành dict (cho báo cáo/serialize)."""
        return asdict(outcome)
