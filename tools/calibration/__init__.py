"""Khung tự hiệu chuẩn đệ quy cho ROI tự động và build phụ đề từ OCR.

Public API:
    * :class:`SearchSpace`, :class:`ParameterSpec` — định nghĩa không gian tham số.
    * :class:`SubtitleBuildEvaluator`, :class:`RoiCalibrationEvaluator` — hàm mục tiêu.
    * :class:`RecursiveCoordinateDescentOptimizer` — tối ưu đệ quy.
    * :class:`RecursiveCalibrator` — điều phối + warm-start (tự cải tiến qua phiên).
    * :func:`auto_pair_seraw_to_srt` — ghép cặp dữ liệu (điều kiện tiên quyết).
"""

from __future__ import annotations

from tools.calibration.datasets import CalibrationDataset
from tools.calibration.ground_truth import (
    GroundTruthCue,
    PairingResult,
    auto_pair_seraw_to_srt,
    parse_srt,
)
from tools.calibration.metrics import SubtitleScore, score_subtitles
from tools.calibration.optimizer import (
    OptimizationResult,
    RecursiveCoordinateDescentOptimizer,
)
from tools.calibration.recursive_calibrator import (
    CalibrationOutcome,
    RecursiveCalibrator,
)
from tools.calibration.roi_evaluator import RoiBand, RoiCalibrationEvaluator
from tools.calibration.search_space import ParameterSpec, SearchSpace
from tools.calibration.subtitle_evaluator import SubtitleBuildEvaluator

__all__ = [
    "CalibrationDataset",
    "GroundTruthCue",
    "PairingResult",
    "auto_pair_seraw_to_srt",
    "parse_srt",
    "SubtitleScore",
    "score_subtitles",
    "OptimizationResult",
    "RecursiveCoordinateDescentOptimizer",
    "CalibrationOutcome",
    "RecursiveCalibrator",
    "RoiBand",
    "RoiCalibrationEvaluator",
    "ParameterSpec",
    "SearchSpace",
    "SubtitleBuildEvaluator",
]
