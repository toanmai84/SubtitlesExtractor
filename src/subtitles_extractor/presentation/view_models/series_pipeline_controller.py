"""[v3.23.658] Điều phối "Dây chuyền phim bộ": chạy lần lượt từng KHÂU cho mọi tập còn thiếu, dừng ở điểm dừng.

Mỗi khâu dùng lại chế độ "cả bộ" sẵn có của trang tương ứng (cùng cấu hình người dùng đã đặt: ROI, giai đoạn dịch, giọng,
chế độ xuất bản) — điều phối chỉ chọn TẬP cho từng khâu và nối các khâu với nhau. Sau mỗi khâu quét lại tệp để biết tập
nào sẵn sàng cho khâu kế; trạng thái lượt chạy ghi xuống thư mục phim bộ (tắt app giữa chừng → mở lại "Chạy tiếp tất cả").
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from subtitles_extractor.application.services.series_pipeline import (
    PUBLISH_SUFFIXES,
    EpisodeStatus,
    SeriesRunState,
    SeriesStage,
    StageState,
    episodes_for_stage,
    first_stage_with_work,
    load_run_state,
    save_run_state,
    scan_series,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StageRunResult:
    """Kết quả một khâu chạy cả bộ (trang báo về qua tín hiệu ``series_batch_finished``)."""

    failed: dict[str, str] = field(default_factory=dict)  # tên video → lý do (tập "xong" do điều phối quét lại tệp)
    cancelled: bool = False                               # người dùng Huỷ
    stop_reason: str = ""                                 # dừng cả dây chuyền (hết hạn mức, lỗi cấu hình…)


class StageRunner:
    """Giao diện tối thiểu một trang phải có để chạy một khâu cho danh sách tập.

    ``start(videos)`` trả ``""`` nếu đã bắt đầu (kết quả về sau qua ``finished``), hoặc thông báo lỗi nếu không chạy được.
    """

    def __init__(self, start: Callable[[list[Path]], str], finished: object, cancel: Callable[[], None]) -> None:
        self.start = start
        self.finished = finished  # Signal(object) phát StageRunResult
        self.cancel = cancel


class SeriesPipelineController(QObject):
    """Chạy tiếp các khâu còn thiếu của cả bộ, theo thứ tự, tới điểm dừng."""

    statuses_changed = Signal(object)        # list[EpisodeStatus]
    stage_started = Signal(str, int)         # khâu, số tập
    run_finished = Signal(str)               # thông báo tổng kết
    running_changed = Signal(bool)

    def __init__(
        self, runners: Mapping[SeriesStage, StageRunner], parent: QObject | None = None,
        prepare: Callable[[list[Path]], object] | None = None,
        publish_suffix: Callable[[], str] | None = None,
    ) -> None:
        """
        Args:
            runners: Khâu → trang chạy khâu đó.
            prepare: Gọi với mọi video của thư mục TRƯỚC khi quét khi bấm chạy (vd ghi bù ``.original.srt`` từ CSDL cho
                tập trích xuất ở bản cũ — không thì khâu Trích xuất OCR lại các tập đó hàng giờ).
        """
        super().__init__(parent)
        self._runners = dict(runners)
        self._prepare = prepare
        self._publish_suffix = publish_suffix
        self._folder: Path | None = None
        self._target_language = "vi"
        self._state = SeriesRunState()
        self._statuses: list[EpisodeStatus] = []
        self._current: SeriesStage | None = None
        self._stop_requested = False
        self._summary: list[str] = []
        self._stage_videos: list[str] = []
        self._shutting_down = False
        for stage, runner in self._runners.items():
            runner.finished.connect(lambda result, s=stage: self._on_stage_finished(s, result))  # type: ignore[attr-defined]

    # ── truy vấn ───────────────────────────────────────────────────────────
    @property
    def folder(self) -> Path | None:
        return self._folder

    @property
    def state(self) -> SeriesRunState:
        return self._state

    @property
    def statuses(self) -> list[EpisodeStatus]:
        return list(self._statuses)

    @property
    def is_running(self) -> bool:
        return self._current is not None

    @property
    def current_stage(self) -> SeriesStage | None:
        return self._current

    # ── thao tác ───────────────────────────────────────────────────────────
    def open_folder(self, folder: Path, target_language: str = "vi") -> None:
        if self.is_running:
            return
        self._folder = Path(folder)
        self._target_language = target_language or "vi"
        self._state = load_run_state(self._folder)
        self.refresh()

    def refresh(self) -> list[EpisodeStatus]:
        if self._folder is None:
            self._statuses = []
        else:
            self._statuses = scan_series(
                self._folder, target_language=self._target_language, state=self._state, publish_suffixes=self._publish_suffixes(),
            )
        self.statuses_changed.emit(list(self._statuses))
        return self._statuses

    def _publish_suffixes(self) -> tuple[str, ...]:
        """[v3.23.659] Khâu Xuất bản "xong" theo hậu tố của chế độ đang chọn (nếu có trang Xuất bản)."""
        if self._publish_suffix is not None:
            try:
                suffix = str(self._publish_suffix() or "")
            except Exception:  # noqa: BLE001 - trang lỗi: dùng mọi hậu tố như cũ
                suffix = ""
            if suffix:
                return (suffix,)
        return PUBLISH_SUFFIXES

    def set_stop_after(self, stage: SeriesStage) -> None:
        self._state.stop_after = stage
        if self._folder is not None and not self.is_running:
            save_run_state(self._folder, self._state)

    def run(self) -> str:
        """Bắt đầu "Chạy tiếp tất cả". Trả ``""`` nếu đã chạy, hoặc lý do không chạy được."""
        if self.is_running:
            return "Dây chuyền đang chạy."
        if self._folder is None:
            return "Chưa chọn thư mục phim bộ."
        if self._prepare is not None:
            from subtitles_extractor.application.services.batch_publish_plan import find_episode_videos

            try:
                self._prepare(find_episode_videos(self._folder))
            except Exception as exc:  # noqa: BLE001 - chuẩn bị hỏng không chặn lượt chạy
                logger.warning("Dây chuyền phim bộ: bước chuẩn bị lỗi: %s", exc)
        self.refresh()
        self._stop_requested = False
        self._summary = []
        self._state.running = True
        save_run_state(self._folder, self._state)
        if not self._start_next(after=None):
            self._state.running = False
            save_run_state(self._folder, self._state)
            return self._last_start_error or "Không còn khâu nào cần chạy tới điểm dừng đã chọn."
        self.running_changed.emit(True)
        return ""

    def stop(self) -> None:
        """Dừng: huỷ khâu đang chạy (trang tự dọn) rồi kết thúc lượt."""
        if not self.is_running:
            return
        self._stop_requested = True
        runner = self._runners.get(self._current) if self._current else None
        if runner is not None:
            runner.cancel()

    def begin_shutdown(self) -> None:
        """Đóng ứng dụng: bỏ qua mọi báo cáo khâu về sau, KHÔNG xoá cờ "đang chạy" (mở lại để chạy tiếp)."""
        self._shutting_down = True

    # ── nội bộ ─────────────────────────────────────────────────────────────
    _last_start_error = ""

    def _start_next(self, after: SeriesStage | None) -> bool:
        self._last_start_error = ""
        stage = first_stage_with_work(self._statuses, self._state.stop_after, after=after)
        if stage is None:
            return False
        runner = self._runners.get(stage)
        if runner is None:
            self._last_start_error = f"Khâu {stage.value} chưa hỗ trợ chạy tự động."
            return False
        videos = episodes_for_stage(self._statuses, stage)
        # Đặt khâu hiện tại TRƯỚC khi gọi start: trang có thể báo xong ngay (vd mọi tập bị bỏ qua).
        self._current = stage
        self._state.stage = stage.value
        self._stage_videos = [v.name for v in videos]
        try:
            error = runner.start(videos)
        except Exception as exc:  # noqa: BLE001 - trang ném lỗi: kết thúc lượt, không kẹt "đang chạy"
            logger.exception("Dây chuyền phim bộ: không bắt đầu được khâu %s", stage.value)
            error = str(exc) or type(exc).__name__
        if error:
            # Không chạy được (trang đang bận, thiếu cấu hình…) → dừng cả lượt, nói rõ lý do.
            self._current = None
            self._state.stage = ""
            self._last_start_error = f"{_STAGE_LABELS.get(stage, stage.value)}: {error}"
            return False
        save_run_state(self._folder, self._state)  # type: ignore[arg-type]
        self.stage_started.emit(stage.value, len(videos))
        return True

    def _on_stage_finished(self, stage: SeriesStage, result: object) -> None:
        if self._current is not stage or self._folder is None or self._shutting_down:
            return  # trang chạy "cả bộ" thủ công, không phải do dây chuyền
        res = result if isinstance(result, StageRunResult) else StageRunResult()
        # Tập "xong" = đã có tệp kết quả của khâu (quét lại thư mục) — không tin số đếm của trang.
        statuses = scan_series(self._folder, target_language=self._target_language,
                               publish_suffixes=self._publish_suffixes())
        produced = {s.name for s in statuses if s.states[stage] is StageState.DONE}
        done = [name for name in self._stage_videos if name in produced]
        self._state.clear_failures(stage, done)
        for name, reason in res.failed.items():
            if name not in produced:
                self._state.record_failures(stage, [name], reason)
        failed_count = sum(1 for name in self._stage_videos if name not in produced and name in res.failed)
        self._summary.append(
            f"{_STAGE_LABELS.get(stage, stage.value)}: xong {len(done)}/{len(self._stage_videos)} tập"
            + (f", lỗi {failed_count}" if failed_count else "")
        )
        self.refresh()
        stop = res.cancelled or self._stop_requested or bool(res.stop_reason)
        # Hoãn một nhịp: trang vừa phát tín hiệu từ trong hàm dọn hàng đợi của nó.
        QTimer.singleShot(0, lambda: self._continue(stage, stop, res.stop_reason))

    def _continue(self, stage: SeriesStage, stop: bool, stop_reason: str) -> None:
        if self._shutting_down:
            return
        # Bấm Dừng ngay sau khi khâu vừa xong (trước nhịp này): huỷ gửi tới trang đã xong nên vô tác dụng — phải dừng ở đây.
        stop = stop or self._stop_requested
        if not stop and self._start_next(after=stage):
            return
        self._current = None
        self._state.stage = ""
        reason = stop_reason or ("Đã dừng theo yêu cầu." if stop else self._last_start_error)
        finished_all = not stop and not self._last_start_error
        self._state.running = False
        save_run_state(self._folder, self._state)  # type: ignore[arg-type]
        message = "\n".join(self._summary) or "Không có tập nào cần chạy."
        if reason:
            message = f"{reason}\n\n{message}"
        elif finished_all and self._state.stop_after is SeriesStage.TRANSLATE:
            message += "\n\nĐã dừng sau khâu Dịch — soát người nói / bản dịch rồi chọn \"Chạy hết tới Xuất bản\"."
        self.running_changed.emit(False)
        self.run_finished.emit(message)


_STAGE_LABELS = {
    SeriesStage.EXTRACT: "Trích xuất",
    SeriesStage.TRANSLATE: "Dịch",
    SeriesStage.TTS: "TTS",
    SeriesStage.PUBLISH: "Xuất bản",
}

__all__ = ["SeriesPipelineController", "StageRunResult", "StageRunner"]
