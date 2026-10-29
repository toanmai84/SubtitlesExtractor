"""[v3.23.658] Điều phối dây chuyền phim bộ: chạy theo khâu, dừng sau Dịch, ghi lỗi, dừng khi hết hạn mức, chạy tiếp."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services.series_pipeline import (
    SeriesStage,
    StageState,
    load_run_state,
)
from subtitles_extractor.presentation.view_models.series_pipeline_controller import (
    SeriesPipelineController,
    StageRunner,
    StageRunResult,
)

S = SeriesStage
SRT = "1\n00:00:01,000 --> 00:00:02,000\n{}\n"


@pytest.fixture(scope="module")
def app():
    # QApplication (không phải QCoreApplication): các test trang chạy sau trong cùng tiến trình cần widget.
    return QApplication.instance() or QApplication([])


class FakeStage(QObject):
    """Giả một trang: "chạy" bằng cách ghi tệp kết quả của khâu cho các tập (trừ tập trong ``fail``)."""

    finished = Signal(object)

    def __init__(self, stage: SeriesStage, fail: set[str] | None = None, stop_reason: str = "",
                 busy: str = "") -> None:
        super().__init__()
        self.stage, self.fail, self.stop_reason, self.busy = stage, fail or set(), stop_reason, busy
        self.calls: list[list[str]] = []
        self.cancelled = False
        self.after_finish = None
        self.raise_on_start = False

    def start(self, videos: list[Path]) -> str:
        if self.raise_on_start:
            raise RuntimeError("trang lỗi")
        if self.busy:
            return self.busy
        self.calls.append([v.name for v in videos])
        QTimer.singleShot(0, lambda: self._finish(videos))
        return ""

    def _finish(self, videos: list[Path]) -> None:
        result = StageRunResult(stop_reason=self.stop_reason, cancelled=self.cancelled)
        for video in videos:
            if video.name in self.fail:
                result.failed[video.name] = "lỗi giả"
                continue
            stem = video.with_suffix("")
            if self.stage is S.EXTRACT:
                Path(f"{stem}.original.srt").write_text(SRT.format("你好"), encoding="utf-8")
            elif self.stage is S.TRANSLATE:
                Path(f"{stem}.translate.vi.srt").write_text(SRT.format("Chào"), encoding="utf-8")
            elif self.stage is S.TTS:
                Path(f"{stem}.tts.vi.srt").write_text(SRT.format("Chào"), encoding="utf-8")
                Path(f"{stem}.wav").write_bytes(b"w")
            else:
                Path(f"{stem}_phude_thuyetminh.mkv").write_bytes(b"m")
        self.finished.emit(result)
        if self.after_finish is not None:
            self.after_finish()

    def cancel(self) -> None:
        self.cancelled = True

    def runner(self) -> StageRunner:
        return StageRunner(self.start, self.finished, self.cancel)


def _folder(tmp_path: Path, n: int = 3) -> Path:
    for i in range(1, n + 1):
        (tmp_path / f"ep{i}.mp4").write_bytes(b"v")
    return tmp_path


def _controller(stages: dict[SeriesStage, FakeStage], prepare=None) -> tuple[SeriesPipelineController, list[str]]:
    controller = SeriesPipelineController({s: f.runner() for s, f in stages.items()}, prepare=prepare)
    messages: list[str] = []
    controller.run_finished.connect(messages.append)
    return controller, messages


def _wait(app, messages: list[str], timeout_ms: int = 3000) -> None:
    from PySide6.QtCore import QElapsedTimer

    timer = QElapsedTimer()
    timer.start()
    while not messages and timer.elapsed() < timeout_ms:
        app.processEvents()


def test_default_run_stops_after_translation(app, tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    fakes = {s: FakeStage(s) for s in S}
    controller, messages = _controller(fakes)
    controller.open_folder(folder)
    assert controller.state.stop_after is S.TRANSLATE
    assert controller.run() == ""
    _wait(app, messages)
    assert fakes[S.EXTRACT].calls == [["ep1.mp4", "ep2.mp4", "ep3.mp4"]]
    assert fakes[S.TRANSLATE].calls == [["ep1.mp4", "ep2.mp4", "ep3.mp4"]]
    assert fakes[S.TTS].calls == [] and fakes[S.PUBLISH].calls == []        # dừng sau Dịch
    assert "Dừng sau khâu Dịch" in messages[0] or "dừng sau khâu Dịch" in messages[0]
    assert not controller.is_running and not load_run_state(folder).running
    # Chọn chạy hết: chỉ còn TTS + Xuất bản.
    controller.set_stop_after(S.PUBLISH)
    messages.clear()
    assert controller.run() == ""
    _wait(app, messages)
    assert fakes[S.TTS].calls == [["ep1.mp4", "ep2.mp4", "ep3.mp4"]]
    assert fakes[S.PUBLISH].calls == [["ep1.mp4", "ep2.mp4", "ep3.mp4"]]
    assert all(s.next_stage() is None for s in controller.statuses)
    assert controller.run() != ""                                             # không còn gì để chạy


def test_failures_are_recorded_and_retried(app, tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    fakes = {s: FakeStage(s) for s in S}
    fakes[S.EXTRACT].fail = {"ep2.mp4"}
    controller, messages = _controller(fakes)
    controller.open_folder(folder)
    controller.run()
    _wait(app, messages)
    states = {s.name: s.states for s in controller.statuses}
    assert states["ep2.mp4"][S.EXTRACT] is StageState.FAILED
    assert fakes[S.TRANSLATE].calls == [["ep1.mp4", "ep3.mp4"]]
    assert load_run_state(folder).failures == {"ep2.mp4": {"extract": "lỗi giả"}}
    fakes[S.EXTRACT].fail = set()
    messages.clear()
    controller.run()
    _wait(app, messages)
    assert fakes[S.EXTRACT].calls[-1] == ["ep2.mp4"]
    assert load_run_state(folder).failures == {}


def test_quota_stop_ends_run_and_resume_after_restart(app, tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    fakes = {s: FakeStage(s) for s in S}
    fakes[S.TRANSLATE].fail = {"ep3.mp4"}
    fakes[S.TRANSLATE].stop_reason = "Hết hạn mức ngày."
    controller, messages = _controller(fakes)
    controller.open_folder(folder)
    controller.run()
    _wait(app, messages)
    assert messages[0].startswith("Hết hạn mức ngày.")
    # "Tắt app": bộ điều phối mới đọc lại trạng thái từ thư mục, chạy tiếp đúng tập còn thiếu.
    fakes2 = {s: FakeStage(s) for s in S}
    controller2, messages2 = _controller(fakes2)
    controller2.open_folder(folder)
    assert controller2.run() == ""
    _wait(app, messages2)
    assert fakes2[S.EXTRACT].calls == [] and fakes2[S.TRANSLATE].calls == [["ep3.mp4"]]


def test_busy_page_reports_reason_and_does_not_hang(app, tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    fakes = {s: FakeStage(s) for s in S}
    fakes[S.EXTRACT].busy = "Trang Trích xuất đang bận."
    controller, _ = _controller(fakes)
    controller.open_folder(folder)
    reason = controller.run()
    assert "đang bận" in reason and not controller.is_running and not load_run_state(folder).running


def test_stop_cancels_current_stage(app, tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    fakes = {s: FakeStage(s) for s in S}
    controller, messages = _controller(fakes)
    controller.open_folder(folder)
    controller.run()
    controller.stop()
    _wait(app, messages)
    assert fakes[S.EXTRACT].cancelled and fakes[S.TRANSLATE].calls == []
    assert "Đã dừng" in messages[0]


def test_stop_between_stages_is_honoured(app, tmp_path: Path) -> None:
    """Rà soát độc lập: bấm Dừng đúng lúc một khâu vừa xong → trước đây khâu kế (Dịch, tốn hạn mức) vẫn chạy."""
    folder = _folder(tmp_path)
    fakes = {s: FakeStage(s) for s in S}
    controller, messages = _controller(fakes)
    fakes[S.EXTRACT].after_finish = controller.stop
    controller.open_folder(folder)
    controller.run()
    _wait(app, messages)
    assert fakes[S.TRANSLATE].calls == [] and "Đã dừng" in messages[0]


def test_runner_exception_does_not_leave_run_stuck(app, tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    fakes = {s: FakeStage(s) for s in S}
    fakes[S.EXTRACT].raise_on_start = True
    controller, _ = _controller(fakes)
    controller.open_folder(folder)
    reason = controller.run()
    assert "trang lỗi" in reason
    assert not controller.is_running and not load_run_state(folder).running


def test_prepare_backfills_before_scanning(app, tmp_path: Path) -> None:
    """Rà soát độc lập: tập trích xuất ở bản cũ (phụ đề chỉ trong CSDL) bị OCR lại hàng giờ thay vì ghi bù tệp."""
    folder = _folder(tmp_path, 2)
    prepared: list[list[str]] = []

    def backfill(videos: list[Path]) -> None:
        prepared.append([v.name for v in videos])
        (folder / "ep1.original.srt").write_text(SRT.format("你好"), encoding="utf-8")

    fakes = {s: FakeStage(s) for s in S}
    controller, messages = _controller(fakes, prepare=backfill)
    controller.open_folder(folder)
    controller.run()
    _wait(app, messages)
    assert prepared[0] == ["ep1.mp4", "ep2.mp4"]
    assert fakes[S.EXTRACT].calls == [["ep2.mp4"]]
