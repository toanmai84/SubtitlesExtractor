"""Đánh giá chất lượng build phụ đề cho một bộ tham số ứng viên.

Lớp này biến một ``assignment`` (dict tên→giá trị tham số) thành điểm chất lượng
trung bình trên toàn corpus — chính là *hàm mục tiêu* mà optimizer cực đại hoá.

Thiết kế theo Dependency Injection: factory dựng builder và hàm nạp OCR được tiêm
vào, nên lớp này không phụ thuộc cứng vào tầng infrastructure và dễ mock khi test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from loguru import logger

from tools.calibration.datasets import CalibrationDataset
from tools.calibration.exceptions import EmptyDatasetError
from tools.calibration.ground_truth import parse_srt
from tools.calibration.metrics import SubtitleScore, score_subtitles


class SupportsInterval(Protocol):
    """Tối thiểu cần có của một sự kiện phụ đề để chấm điểm."""

    text: str

    @property
    def interval(self) -> "SupportsStartEnd": ...


class SupportsStartEnd(Protocol):
    start_sec: float
    end_sec: float


BuilderFactory = Callable[[dict[str, float | int | bool]], object]
OcrLoader = Callable[[object], tuple[Sequence[object], object]]


@dataclass(slots=True)
class _PreparedDataset:
    """Frames + ground-truth đã nạp sẵn (cache) cho một dataset."""

    label: str
    frames: Sequence[object]
    ground_truth: list[tuple[float, float, str]]


class SubtitleBuildEvaluator:
    """Hàm mục tiêu hiệu chuẩn build phụ đề.

    Args:
        corpus: Danh sách dataset (đã ghép cặp đúng).
        builder_factory: Hàm nhận ``assignment`` → trả builder có ``.build(frames, roi)``.
        ocr_loader: Hàm nạp OCR thô ``path -> (frames, meta)``.
        base_config_kwargs: Tham số cố định luôn áp dụng (vd ``sample_step_sec``).
    """

    def __init__(
        self,
        *,
        corpus: list[CalibrationDataset],
        builder_factory: BuilderFactory,
        ocr_loader: OcrLoader,
        base_config_kwargs: dict[str, float | int | bool] | None = None,
    ) -> None:
        if not corpus:
            raise EmptyDatasetError("Corpus hiệu chuẩn rỗng.")
        self._builder_factory = builder_factory
        self._base_config_kwargs = dict(base_config_kwargs or {})
        self._prepared: list[_PreparedDataset] = [
            self._prepare(dataset, ocr_loader) for dataset in corpus
        ]

    @staticmethod
    def _prepare(dataset: CalibrationDataset, ocr_loader: OcrLoader) -> _PreparedDataset:
        frames, _meta = ocr_loader(dataset.seraw_path)
        cues = parse_srt(dataset.srt_path)
        window = dataset.time_window
        if window is not None:
            low, high = window
            frames = [
                frame
                for frame in frames
                if low <= getattr(frame, "timestamp_sec", 0.0) <= high
            ]
            cues = [cue for cue in cues if low <= cue.start_sec <= high]
        ground_truth = [(cue.start_sec, cue.end_sec, cue.text) for cue in cues]
        logger.info(
            "Nạp dataset '{}': {} frames, {} câu GT{}",
            dataset.label, len(frames), len(ground_truth),
            "" if window is None else f" (cửa sổ {window[0]:.0f}-{window[1]:.0f}s)",
        )
        return _PreparedDataset(dataset.label, frames, ground_truth)

    def score_assignment(self, assignment: dict[str, float | int | bool]) -> SubtitleScore:
        """Chấm điểm tổng hợp một bộ tham số trên toàn corpus.

        Returns:
            :class:`SubtitleScore` gộp (tổng đếm cộng dồn, CER lấy trung bình
            có trọng số theo số câu GT).
        """
        config_kwargs = {**self._base_config_kwargs, **assignment}
        builder = self._builder_factory(config_kwargs)

        total_gt = total_built = total_matched = 0
        total_exact = total_spurious = total_missing = 0
        weighted_cer_sum = 0.0
        for prepared in self._prepared:
            events = builder.build(prepared.frames, roi=None)
            built = [
                (event.interval.start_sec, event.interval.end_sec, event.text)
                for event in events
            ]
            score = score_subtitles(prepared.ground_truth, built)
            total_gt += score.ground_truth_count
            total_built += score.built_count
            total_matched += score.matched_count
            total_exact += score.exact_count
            total_spurious += score.spurious_count
            total_missing += score.missing_count
            weighted_cer_sum += score.average_cer * max(1, score.ground_truth_count)

        average_cer = weighted_cer_sum / max(1, total_gt)
        return SubtitleScore(
            ground_truth_count=total_gt,
            built_count=total_built,
            matched_count=total_matched,
            exact_count=total_exact,
            spurious_count=total_spurious,
            missing_count=total_missing,
            average_cer=average_cer,
        )

    def objective(self, assignment: dict[str, float]) -> float:
        """Hàm mục tiêu vô hướng (quality) cho optimizer."""
        return self.score_assignment(assignment).quality
