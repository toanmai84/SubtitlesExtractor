"""[v3.23.613] Chạy tiếp đúng lúc khi luồng trước CHƯA dọn xong (Xuất bản, Trích xuất).

* "Nối cả bộ" lỗi (sao chép luồng lệch thông số) → hỏi "Nén lại?" → Có: trước đây lượt
  nén lại bị chặn im lặng vì luồng nối cũ còn đó (tín hiệu ``done`` dọn luồng chưa về)
  — bấm Có mà không có gì xảy ra.
* Xuất hàng loạt: tập kế được gọi khi tín hiệu ``done`` của tập trước có thể chưa về →
  ``_start_render`` thấy luồng cũ và thoát im lặng, hàng đợi đứng yên giữa chừng.
* Trích xuất hàng loạt: tập kế bắt đầu khi trang còn "bận" (luồng tập trước chưa báo
  thoát) → ``start_extraction`` thoát im lặng mà tập đã bị gỡ khỏi hàng chờ.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QMessageBox


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(app: QApplication, tmp_path: Path) -> Iterator[Any]:
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import publish_page

    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
        i18n_data_dir=Path(publish_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp_path,
    )
    widget = publish_page.PublishPage(container)
    yield widget
    widget.close()


class _FakeConcatWorker(QObject):
    """Worker nối giả: không chạy ffmpeg, chỉ báo xong."""

    progress = Signal(int)
    finished = Signal(object)
    failed = Signal(str)
    done = Signal()
    created: ClassVar[list[bool]] = []

    def __init__(
        self, _videos: Any, _output: Any, _duration: float, *, reencode: bool = False
    ) -> None:
        super().__init__()
        _FakeConcatWorker.created.append(reencode)

    def run(self) -> None:
        self.done.emit()


def _pump(seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)


class TestConcatReencode:
    def test_yes_starts_reencode_after_old_thread_is_cleaned(
        self, page: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from subtitles_extractor.presentation.workers import concat_video_worker

        _FakeConcatWorker.created = []
        monkeypatch.setattr(concat_video_worker, "ConcatVideoWorker", _FakeConcatWorker)
        monkeypatch.setattr(
            QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes
        )
        monkeypatch.setattr(QMessageBox, "warning", lambda *_a, **_k: None)
        monkeypatch.setattr(QMessageBox, "information", lambda *_a, **_k: None)
        tmp = Path(tempfile.mkdtemp())
        page._start_concat([tmp / "a.mp4", tmp / "b.mp4"], tmp / "all.mp4")
        assert _FakeConcatWorker.created == [False]
        # Lỗi về TRƯỚC tín hiệu done của luồng cũ (thứ tự thật của ConcatVideoWorker).
        page._on_concat_failed("Codec copy: stream parameters do not match")
        _pump(1.0)
        assert _FakeConcatWorker.created == [False, True]


class TestBatchPublish:
    def test_next_item_waits_for_previous_thread(
        self, page: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        built: list[int] = []
        monkeypatch.setattr(page, "_build_request", lambda: built.append(1))
        # [v3.23.650] _build_request trả None = tập không bắt đầu được → tính là lỗi, hết hàng đợi thì hiện tổng kết.
        monkeypatch.setattr(QMessageBox, "information", lambda *_a, **_k: None)
        tmp = Path(tempfile.mkdtemp())
        item = SimpleNamespace(
            video_path=tmp / "t2.mp4", subtitle_path=None, audio_path=None,
            output_path=tmp / "t2.out.mp4",
        )
        page._batch_items = [item, item]
        page._batch_index = 1
        page._batch_cancelled = False
        page._batch_failures = []
        page._thread = object()  # luồng tập trước: "done" chưa về
        page._run_next_publish()
        assert built == []
        page._thread = None  # "done" về, luồng đã dọn
        _pump(0.5)
        assert built == [1]


class TestBatchExtraction:
    def test_next_episode_waits_until_not_busy(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        from subtitles_extractor.composition.container import ApplicationContainer
        from subtitles_extractor.infrastructure.settings.json_file_repository import (
            JsonFileRepository,
        )
        from subtitles_extractor.presentation.pages import extract_page

        container = ApplicationContainer(
            settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
            i18n_data_dir=Path(extract_page.__file__).resolve().parents[2] / "data",
            user_data_dir=tmp_path,
        )
        page = extract_page.ExtractPage(container)
        view_model = page._view_model
        busy = {"value": True}
        monkeypatch.setattr(
            type(view_model), "is_busy", property(lambda _s: busy["value"])
        )
        started: list[bool] = []
        monkeypatch.setattr(
            view_model, "start_extraction", lambda: started.append(busy["value"])
        )
        page._pending_batch_item = SimpleNamespace(roi=None)
        page._start_batch_item_extraction()
        assert started == []
        busy["value"] = False  # luồng tập trước đã báo thoát
        _pump(0.6)
        assert started == [False]
        page.close()
