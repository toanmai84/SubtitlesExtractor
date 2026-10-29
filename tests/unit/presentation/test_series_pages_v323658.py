"""[v3.23.658] Các trang chạy một khâu cho dây chuyền phim bộ (không hộp thoại) + lỗi Huỷ lúc đang nạp tập kế."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _container(tmp_path: Path):
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
    from subtitles_extractor.presentation.pages import extract_page

    return ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
        i18n_data_dir=Path(extract_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp_path,
    )


def _pump(seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)


@pytest.fixture
def no_dialogs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    shown: list[str] = []
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: shown.append(str(a[1:2]))))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    return shown


@pytest.fixture
def extract(app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    from subtitles_extractor.presentation.pages import extract_page

    page = extract_page.ExtractPage(_container(tmp_path))
    vm = page._view_model
    monkeypatch.setattr(type(vm), "is_busy", property(lambda _s: False))
    page.loaded: list[str] = []
    page.started: list[str] = []

    def load(path: Path) -> None:
        page.loaded.append(Path(path).name)
        QTimer.singleShot(0, page._start_batch_item_extraction)

    def start() -> None:
        page.started.append(page.loaded[-1])
        QTimer.singleShot(0, page._advance_batch)

    monkeypatch.setattr(vm, "load_video", load)
    monkeypatch.setattr(vm, "start_extraction", start)
    monkeypatch.setattr(vm, "cancel_extraction", lambda: None)
    yield page
    page.close()


def _videos(tmp_path: Path, n: int = 3) -> list[Path]:
    videos = []
    for i in range(1, n + 1):
        video = tmp_path / f"ep{i}.mp4"
        video.write_bytes(b"v")
        videos.append(video)
    return videos


def test_cancel_while_next_episode_loads_does_not_extract_it(extract: Any, tmp_path: Path, no_dialogs) -> None:
    """Lỗi có sẵn: bấm Huỷ đúng lúc trang đang nạp tập kế → tập đó VẪN bị trích xuất rồi mới dừng."""
    from subtitles_extractor.application.services.batch_extraction_plan import build_batch_plan

    plan = build_batch_plan(_videos(tmp_path), reference_roi=None, reference_size=None)
    extract._batch_queue = [item for item in plan if item.will_run]
    extract._batch_index = 0
    extract._batch_failures = []
    extract._run_next_batch_item()            # bắt đầu nạp ep1
    extract._on_cancel_clicked()              # Huỷ trước khi nạp xong
    _pump(0.3)
    assert extract.started == []
    assert not extract._batch_queue


def test_series_batch_runs_without_dialogs_and_reports(extract: Any, tmp_path: Path, no_dialogs) -> None:
    results: list[Any] = []
    extract.series_batch_finished.connect(results.append)
    assert extract.start_series_batch(_videos(tmp_path)) == ""
    _pump(0.5)
    assert extract.started == ["ep1.mp4", "ep2.mp4", "ep3.mp4"]
    assert len(results) == 1 and results[0].failed == {} and not results[0].cancelled
    assert no_dialogs == []
    # Đang chạy cả bộ → từ chối lượt dây chuyền mới (không chồng hai hàng đợi).
    extract._batch_queue = [object()]
    assert extract.start_series_batch(_videos(tmp_path)) != ""


def _page(kind: str, tmp_path: Path):
    from subtitles_extractor.presentation.pages import publish_page, translate_page, tts_page

    cls = {"translate": translate_page.TranslatePage, "tts": tts_page.TTSPage,
           "publish": publish_page.PublishPage}[kind]
    return cls(_container(tmp_path))


@pytest.mark.parametrize("kind", ["translate", "tts", "publish"])
def test_nothing_runnable_reports_missing_inputs(app, tmp_path: Path, kind: str, no_dialogs) -> None:
    page = _page(kind, tmp_path)
    results: list[Any] = []
    page.series_batch_finished.connect(results.append)
    if kind == "translate":
        page._collect_enabled_stages = lambda: [object()]          # có giai đoạn dịch bật
        page._backfill_original_from_db = lambda videos: 0
    videos = _videos(tmp_path, 2)                                     # chưa có tệp đầu vào nào
    assert page.start_series_batch(videos) == ""
    _pump(0.1)
    assert len(results) == 1
    assert set(results[0].failed) == {"ep1.mp4", "ep2.mp4"}           # thiếu đầu vào → ghi lỗi, không treo
    assert no_dialogs == []
    page.close()


@pytest.mark.parametrize("kind", ["translate", "tts", "publish"])
def test_finish_in_series_mode_emits_instead_of_dialog(app, tmp_path: Path, kind: str, no_dialogs) -> None:
    from types import SimpleNamespace

    page = _page(kind, tmp_path)
    results: list[Any] = []
    page.series_batch_finished.connect(results.append)
    page._series_batch = True
    page._series_invalid = {}
    page._batch_items = [SimpleNamespace(video_path=tmp_path / "ep1.mp4"), SimpleNamespace(video_path=tmp_path / "ep2.mp4")]
    page._batch_index = 2
    page._batch_failures = ["ep2.mp4"]
    page._batch_cancelled = False
    page._batch_done = 1
    page._batch_token_mark = {}
    if kind == "translate":
        page._batch_quota_stopped = True
    finish = getattr(page, {"translate": "_finish_batch_translate", "tts": "_finish_batch_tts",
                            "publish": "_finish_batch_publish"}[kind])
    finish()
    assert no_dialogs == []
    assert results and set(results[0].failed) == {"ep2.mp4"}
    if kind == "translate":
        assert "hạn mức" in results[0].stop_reason                 # hết hạn mức → dừng cả dây chuyền
    assert not page._batch_items and not getattr(page, "_series_batch", False)
    # Lượt "cả bộ" thủ công sau đó vẫn hiện tổng kết như cũ.
    page._batch_items = [SimpleNamespace(video_path=tmp_path / "ep1.mp4")]
    page._batch_index, page._batch_failures, page._batch_cancelled = 1, [], False
    finish()
    assert len(results) == 1 and no_dialogs
    page.close()


def test_series_page_shows_table_and_runs(app, tmp_path: Path, no_dialogs) -> None:
    from PySide6.QtCore import QSettings

    from subtitles_extractor.application.services.series_pipeline import SeriesStage
    from subtitles_extractor.presentation.pages.series_page import SeriesPage
    from subtitles_extractor.presentation.view_models.series_pipeline_controller import SeriesPipelineController
    from tests.unit.presentation.test_series_controller_v323658 import FakeStage

    QSettings().remove("series/last_folder")
    fakes = {s: FakeStage(s) for s in SeriesStage}
    controller = SeriesPipelineController({s: f.runner() for s, f in fakes.items()})
    page = SeriesPage(controller, _container(tmp_path).translator)
    page._open(tmp_path)
    _videos(tmp_path, 2)
    controller.refresh()
    assert page._table.rowCount() == 2 and page._table.item(0, 1).text() == "▶"
    page._on_run()
    assert not page._btn_run.isEnabled() and page._btn_stop.isEnabled()
    _pump(0.5)
    assert page._btn_run.isEnabled()
    assert [page._table.item(0, c).text() for c in range(1, 5)] == ["✓", "✓", "▶", "·"]   # dừng sau Dịch
    assert no_dialogs                                                  # tổng kết hiện một lần
    page.close()
    QSettings().remove("series/last_folder")


def test_manual_buttons_do_not_wipe_running_queue(extract: Any, tmp_path: Path, no_dialogs) -> None:
    """Rà soát độc lập: Bắt đầu / Hàng loạt bấm giữa hai tập (nút được bật lại) xoá hàng đợi đang chạy → treo."""
    queue = [object()]
    extract._batch_queue = queue
    extract._series_batch = True
    from PySide6.QtWidgets import QFileDialog

    video = _videos(tmp_path, 1)[0]
    QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([str(video)], ""))  # type: ignore[method-assign]
    extract._on_start_clicked()
    extract._on_batch_clicked()
    assert extract._batch_queue is queue


@pytest.mark.parametrize("kind", ["tts", "publish"])
def test_manual_batch_does_not_replace_running_queue(app, tmp_path: Path, kind: str, no_dialogs, monkeypatch) -> None:
    from PySide6.QtWidgets import QFileDialog

    page = _page(kind, tmp_path)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path)))
    _videos(tmp_path, 1)
    srt = "1\n00:00:01,000 --> 00:00:02,000\nChào\n"
    (tmp_path / "ep1.translate.vi.srt").write_text(srt, encoding="utf-8")   # TTS chạy được
    (tmp_path / "ep1.tts.srt").write_text(srt, encoding="utf-8")            # Xuất bản chạy được
    (tmp_path / "ep1.wav").write_bytes(b"w")
    if kind == "tts":
        (tmp_path / "ep1.wav").unlink()
    queue = [object()]
    page._batch_items = queue
    getattr(page, {"tts": "_on_batch_tts_clicked", "publish": "_on_batch_publish_clicked"}[kind])()
    assert page._batch_items is queue
    page._batch_items = []
    page.close()


def test_tts_start_failure_reports_reason(app, tmp_path: Path, no_dialogs) -> None:
    """Rà soát độc lập: TTS không khởi động được (engine/khoá) → trước đây báo "Đã dừng theo yêu cầu"."""
    from types import SimpleNamespace

    page = _page("tts", tmp_path)
    results: list[Any] = []
    page.series_batch_finished.connect(results.append)
    page._series_batch, page._series_invalid = True, {}
    page._batch_items = [SimpleNamespace(video_path=tmp_path / "ep1.mp4")]
    page._batch_index, page._batch_failures, page._batch_cancelled = 0, [], False
    page._on_generate = lambda: None                                  # không khởi động được
    page._generate_for_batch()
    assert results and results[0].stop_reason and not results[0].cancelled
    page.close()


def test_empty_extraction_counts_as_failure(extract: Any, tmp_path: Path, no_dialogs) -> None:
    """Rà soát độc lập: tập ra 0 câu không ghi tệp và không tính lỗi → mỗi lượt OCR lại tập đó."""
    from types import SimpleNamespace

    from subtitles_extractor.application.services.batch_extraction_plan import build_batch_plan

    results: list[Any] = []
    extract.series_batch_finished.connect(results.append)
    plan = build_batch_plan(_videos(tmp_path, 1), reference_roi=None, reference_size=None)
    extract._series_batch, extract._series_invalid = True, {}
    extract._batch_queue = [item for item in plan if item.will_run]
    extract._batch_index, extract._batch_failures, extract._batch_cancelled = 0, [], False
    extract._on_extraction_finished(SimpleNamespace(events=[]))
    _pump(0.2)
    assert results and "ep1.mp4" in results[0].failed


def test_opus_tts_output_is_detected(tmp_path: Path) -> None:
    from subtitles_extractor.application.services.batch_publish_plan import find_tts_audio
    from subtitles_extractor.application.services.batch_tts_plan import find_existing_audio

    video = tmp_path / "ep1.mp4"
    video.write_bytes(b"v")
    (tmp_path / "ep1.opus").write_bytes(b"o")
    assert find_tts_audio(video) == tmp_path / "ep1.opus"
    assert find_existing_audio(video) == tmp_path / "ep1.opus"


def test_cancelled_episode_is_not_a_failure_and_eta_resets(extract: Any, tmp_path: Path, no_dialogs) -> None:
    """[v3.23.659] Huỷ giữa tập: không tính tập đó là lỗi; tập kế bắt đầu với ETA mới (trước đây "~00:00" mãi)."""
    from types import SimpleNamespace

    from subtitles_extractor.application.services.batch_extraction_plan import build_batch_plan

    results: list[Any] = []
    extract.series_batch_finished.connect(results.append)
    plan = build_batch_plan(_videos(tmp_path, 2), reference_roi=None, reference_size=None)
    extract._series_batch, extract._series_invalid = True, {}
    extract._batch_queue = [item for item in plan if item.will_run]
    extract._batch_index, extract._batch_failures, extract._batch_cancelled = 0, [], True
    extract._eta_display_sec = 0.0
    extract._on_extraction_finished(SimpleNamespace(events=[]))
    _pump(0.2)
    assert results and results[0].failed == {} and results[0].cancelled
    extract._eta_display_sec = 0.0
    extract._pending_batch_item = SimpleNamespace(roi=None, video_path=tmp_path / "ep2.mp4")
    extract._batch_cancelled = False
    extract.loaded.append("ep2.mp4")
    extract._view_model.start_extraction = lambda: None                 # chỉ kiểm ETA, không chạy tiếp hàng đợi
    extract._start_batch_item_extraction()
    assert extract._eta_display_sec is None
