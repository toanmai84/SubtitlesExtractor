"""[v3.23.450] Worker QThread: TTS lại các dòng được chọn rồi ghép lại master."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Event

from PySide6.QtCore import QThread, Signal

from subtitles_extractor.application.use_cases.resynthesize_lines import (
    ResynthesizeLinesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_tts_port import (
    TTSCancelledError,
    TTSRequest,
    TTSSegmentResult,
)

logger = logging.getLogger(__name__)


class ResynthesizeWorker(QThread):
    """Chạy :class:`ResynthesizeLinesUseCase` trên nền, không chặn UI.

    Signals:
        progress_changed: (ratio 0..1, label).
        finished_ok:      list[TTSSegmentResult] đã cập nhật.
        failed:           thông điệp lỗi.
        cancelled:        [v3.23.630] người dùng huỷ (lượt tăng dần có thể kéo dài nhiều phút).
    """

    progress_changed = Signal(float, str)
    finished_ok = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        use_case: ResynthesizeLinesUseCase,
        *,
        full_request: TTSRequest,
        selected_indices: set[int],
        prior_results: list[TTSSegmentResult],
        output_path: Path,
        sample_rate: int,
        total_length_samples: int,
        remerge_only: bool = False,
        checkpoint_cb=None,
    ) -> None:
        super().__init__()
        #: [v3.23.647] ``cb(kết quả các dòng đã xong, sample rate)`` — điểm kiểm tra giữa lượt tăng dần dài.
        self._checkpoint_cb = checkpoint_cb
        self._use_case = use_case
        self._full_request = full_request
        self._selected_indices = selected_indices
        self._prior_results = prior_results
        self._output_path = output_path
        self._sample_rate = sample_rate
        self._total_length_samples = total_length_samples
        self._remerge_only = remerge_only
        self._cancel_event = Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            results = self._use_case.execute(
                full_request=self._full_request,
                selected_indices=self._selected_indices,
                prior_results=self._prior_results,
                output_path=self._output_path,
                sample_rate=self._sample_rate,
                total_length_samples=self._total_length_samples,
                progress_cb=self._on_progress,
                cancel_cb=self._cancel_event.is_set,
                remerge_only=self._remerge_only,
                checkpoint_cb=self._checkpoint_cb,
            )
            self.finished_ok.emit(results)
        except TTSCancelledError:
            logger.info("ResynthesizeWorker: người dùng huỷ.")
            self.cancelled.emit()
        except ValueError as exc:
            logger.warning("ResynthesizeWorker: %s", exc)
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 — chống treo UI nếu lỗi bất ngờ
            logger.exception("ResynthesizeWorker lỗi không xác định.")
            self.failed.emit(f"Lỗi không xác định khi TTS lại: {exc}")

    def _on_progress(self, ratio: float, label: str) -> None:
        self.progress_changed.emit(ratio, label)


__all__ = ["ResynthesizeWorker"]
