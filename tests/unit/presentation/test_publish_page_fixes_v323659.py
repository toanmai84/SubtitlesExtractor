"""[v3.23.659] Rà BUG khâu Xuất bản (trang + cửa sổ chính)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services.series_pipeline import SeriesStage, StageState, scan_episode


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _container(tmp_path: Path):
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
    from subtitles_extractor.presentation.pages import publish_page

    return ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
        i18n_data_dir=Path(publish_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp_path,
    )


@pytest.fixture
def page(app: QApplication, tmp_path: Path) -> Iterator[Any]:
    from subtitles_extractor.presentation.pages import publish_page

    QSettings().remove("publish")
    widget = publish_page.PublishPage(_container(tmp_path))
    yield widget
    widget.close()
    QSettings().remove("publish")


def test_preview_is_ignored_in_batch(page: Any, tmp_path: Path) -> None:
    video = tmp_path / "ep1.mp4"
    video.write_bytes(b"v")
    (tmp_path / "ep1.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nA\n", encoding="utf-8")
    page._video_edit.setText(str(video))
    page._subtitle_edit.setText(str(tmp_path / "ep1.srt"))
    page._output_edit.setText(str(tmp_path / "ep1_phude.mkv"))
    page._preview_check.setChecked(True)
    assert page._build_request().clip_duration_sec > 0                 # xuất tay: vẫn xuất thử
    page._batch_items = [SimpleNamespace(video_path=video)]
    request = page._build_request()
    assert request.clip_duration_sec == 0 and request.output_path.name == "ep1_phude.mkv"
    page._batch_items = []


def test_publish_modes_survive_restart(app: QApplication, tmp_path: Path) -> None:
    from subtitles_extractor.presentation.pages import publish_page

    QSettings().remove("publish")
    first = publish_page.PublishPage(_container(tmp_path))
    first._subtitle_combo.setCurrentIndex(1)
    first._audio_combo.setCurrentIndex(1)
    suffix = first._current_output_suffix()
    first.close()
    second = publish_page.PublishPage(_container(tmp_path))
    assert second._current_output_suffix() == suffix != "_phude"       # mặc định là "_phude"
    second.close()
    QSettings().remove("publish")


def test_series_publish_done_only_for_selected_mode(tmp_path: Path) -> None:
    video = tmp_path / "ep1.mp4"
    video.write_bytes(b"v")
    (tmp_path / "ep1_phude.mkv").write_bytes(b"m")                    # bản cũ chỉ có phụ đề
    status = scan_episode(video, publish_suffixes=("_phude_thuyetminh",))
    assert status.states[SeriesStage.PUBLISH] is not StageState.DONE


def test_batch_tts_and_publish_do_not_touch_open_project(app: QApplication, tmp_path: Path) -> None:
    from subtitles_extractor.presentation.pages import publish_page

    widget = publish_page.PublishPage(_container(tmp_path))
    emitted: list[object] = []
    widget.publish_completed.connect(emitted.append)
    widget._batch_items = [SimpleNamespace(video_path=tmp_path / "ep1.mp4")]
    widget._batch_index, widget._batch_failures, widget._batch_cancelled = 0, [], False
    widget._on_render_finished(tmp_path / "ep1_phude.mkv")
    assert emitted == []
    widget._batch_items = []
    widget.close()


def test_concat_can_be_cancelled_and_is_locked_while_rendering(page: Any, tmp_path: Path, monkeypatch) -> None:
    from PySide6.QtCore import QObject, Signal

    from subtitles_extractor.presentation.workers import concat_video_worker

    class _Worker(QObject):
        progress = Signal(int)
        finished = Signal(object)
        failed = Signal(str)
        done = Signal()

        def __init__(self, *a, **k) -> None:
            super().__init__()
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

        def run(self) -> None:
            pass

    monkeypatch.setattr(concat_video_worker, "ConcatVideoWorker", _Worker)
    page._set_busy(True)                                              # đang xuất một tập
    assert not page._concat_button.isEnabled()
    page._thread = object()
    page._start_concat([tmp_path / "a.mkv", tmp_path / "b.mkv"], tmp_path / "x_tron_bo.mkv")
    assert getattr(page, "_concat_thread", None) is None              # không chạy chồng lên lượt xuất
    page._thread = None
    page._set_busy(False)
    page._start_concat([tmp_path / "a.mkv", tmp_path / "b.mkv"], tmp_path / "x_tron_bo.mkv")
    worker = page._concat_worker
    assert page._cancel_button.isEnabled()
    page._cancel_render()
    assert worker.cancelled
    page._on_concat_failed("Đã huỷ")                                  # không mời "Nén lại"
    page._cleanup_concat_thread()
    assert page._concat_button.isEnabled() and not page._cancel_button.isEnabled()
