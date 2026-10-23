"""[v3.23.645] Use case đo A/B OCR: chạy lần lượt các kịch bản trên một đoạn phim ngắn bằng ĐÚNG pipeline trích xuất."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

import numpy as np

from subtitles_extractor.application.dtos.extract_subtitles_dto import ExtractSubtitlesRequest
from subtitles_extractor.application.services.ocr_benchmark import (
    BenchmarkScenario,
    ScenarioResult,
    apply_scenario,
    is_redundant_scenario,
)
from subtitles_extractor.application.services.ocr_probe_window import ProbeWindow
from subtitles_extractor.application.use_cases.extract_subtitles import ExtractSubtitlesUseCase
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult
from subtitles_extractor.domain.exceptions import SubtitlesExtractorError
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig, OcrEnginePort
from subtitles_extractor.domain.ports.progress_reporter_port import (
    NullProgressReporter,
    ProgressReporterPort,
)

logger = logging.getLogger(__name__)

__all__ = ["BenchmarkOcrUseCase", "CountingOcrEngine"]

_WARMUP_FRAMES = 3


class CountingOcrEngine:
    """Bọc một engine OCR: đếm số khung / số lô / thời gian thật sự ở trong bộ OCR."""

    def __init__(self, inner: OcrEnginePort) -> None:
        self._inner = inner
        self.frames = 0
        self.batches = 0
        self.infer_seconds = 0.0

    @property
    def is_initialized(self) -> bool:
        return self._inner.is_initialized

    def initialize(self) -> None:
        self._inner.initialize()

    def release(self) -> None:
        self._inner.release()

    def infer(self, image_rgb: np.ndarray, frame_index: int, timestamp_sec: float) -> OcrFrameResult:
        return self.infer_batch([image_rgb], [frame_index], [timestamp_sec])[0]

    def infer_batch(
        self, images_rgb: list[np.ndarray], frame_indices: list[int], timestamps_sec: list[float],
    ) -> list[OcrFrameResult]:
        started = time.perf_counter()
        try:
            return self._inner.infer_batch(images_rgb, frame_indices, timestamps_sec)
        finally:
            self.infer_seconds += time.perf_counter() - started
            self.frames += len(images_rgb)
            self.batches += 1


class _ScenarioProgress:
    """Chuyển tiến độ của một lượt trích xuất thành tiến độ tổng của phép đo; huỷ theo bộ báo cáo cha."""

    def __init__(self, parent: ProgressReporterPort, index: int, total: int, label: str) -> None:
        self._parent, self._index, self._total, self._label = parent, index, total, label

    def report(self, current: int, total: int, message: str = "") -> None:
        fraction = (current / total) if total > 0 else 0.0
        self._parent.report(
            int((self._index + min(1.0, max(0.0, fraction))) * 100), self._total * 100,
            f"Kịch bản {self._index + 1}/{self._total}: {self._label}",
        )

    def is_cancelled(self) -> bool:
        return self._parent.is_cancelled()


class BenchmarkOcrUseCase:
    """Đo từng kịch bản rồi trả danh sách :class:`ScenarioResult` (chưa so sánh — dùng ``recommend``)."""

    def __init__(
        self,
        extract_factory: Callable[[OcrEnginePort], ExtractSubtitlesUseCase],
        engine: OcrEnginePort,
        configure_engine: Callable[[OcrEngineConfig], None],
    ) -> None:
        self._extract_factory = extract_factory
        self._engine = engine
        self._configure_engine = configure_engine

    def execute(
        self,
        base_request: ExtractSubtitlesRequest,
        scenarios: Sequence[BenchmarkScenario],
        window: ProbeWindow,
        progress: ProgressReporterPort | None = None,
    ) -> list[ScenarioResult]:
        reporter = progress or NullProgressReporter()
        # Mốc so sánh luôn chạy đầu; kịch bản trùng cấu hình hiện tại không cần đo lại.
        ordered = sorted(
            (s for s in scenarios if not is_redundant_scenario(base_request, s, window)),
            key=lambda s: s.key != "current",
        )
        results: list[ScenarioResult] = []
        try:
            for index, scenario in enumerate(ordered):
                if reporter.is_cancelled():
                    break
                outcome = self._run_one(base_request, scenario, window, reporter, index, len(ordered))
                if reporter.is_cancelled():
                    break                   # kịch bản đang dở: số đo không đầy đủ, không đưa vào bảng
                results.append(outcome)
        finally:
            # Trả engine dùng chung về đúng cấu hình ban đầu của ứng dụng (kịch bản có thể đã đổi batch / hướng chữ).
            try:
                self._configure_engine(base_request.ocr)
            except Exception as exc:  # noqa: BLE001 - dọn dẹp không được che lỗi gốc
                logger.warning("Không khôi phục được cấu hình OCR sau khi đo: %s", exc)
        return results

    def _run_one(
        self, base: ExtractSubtitlesRequest, scenario: BenchmarkScenario, window: ProbeWindow,
        reporter: ProgressReporterPort, index: int, total: int,
    ) -> ScenarioResult:
        result = ScenarioResult(scenario=scenario)
        request = apply_scenario(base, scenario, window)
        scenario_reporter = _ScenarioProgress(reporter, index, total, scenario.label_vi)
        try:
            scenario_reporter.report(0, 1, "Đang nạp model OCR…")
            self._configure_engine(request.ocr)
            self._engine.initialize()
            self._warm_up(request)
            if reporter.is_cancelled():
                return result
            counting = CountingOcrEngine(self._engine)
            started = time.perf_counter()
            response = self._extract_factory(counting).execute(request, scenario_reporter)
            result.wall_s = time.perf_counter() - started
            result.ocr_s = counting.infer_seconds
            result.ocr_frames = counting.frames
            result.batches = counting.batches
            result.events = tuple(
                (float(e.start_sec), float(e.end_sec), str(e.text)) for e in response.events
            )
        except (SubtitlesExtractorError, RuntimeError, OSError, ValueError, MemoryError) as exc:
            logger.warning("Kịch bản đo '%s' lỗi: %s", scenario.key, exc)
            result.error = f"{type(exc).__name__}: {exc}"
        return result

    def _warm_up(self, request: ExtractSubtitlesRequest) -> None:
        """Chạy vài khung nháp (nạp model, biên dịch kernel) — KHÔNG tính giờ."""
        roi = request.roi
        height = max(32, roi.height) if roi else 128
        width = max(64, roi.width) if roi else 640
        blank = np.full((height, width, 3), 255, np.uint8)
        count = max(1, int(getattr(request.ocr, "det_batch_size", 1) or 1))     # nóng đúng cỡ lô sẽ dùng
        for _ in range(_WARMUP_FRAMES):
            self._engine.infer_batch([blank] * count, list(range(count)), [0.0] * count)
