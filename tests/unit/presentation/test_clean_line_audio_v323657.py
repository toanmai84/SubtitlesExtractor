"""[v3.23.657] Nút "Dọn audio từng dòng": người dùng bấm mới dọn (không tự dọn sau xuất bản).

Xoá seg_*.wav + thùng rác + cache dòng; giữ master + SRT; sau dọn hết "TTS lại dòng chọn".
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox

from subtitles_extractor.infrastructure.tts.line_cache_store import CACHE_FILENAME
from subtitles_extractor.infrastructure.tts.segment_store import (
    TRASH_DIR_NAME,
    clean_line_audio,
    line_audio_usage,
)
from tests.unit.presentation.test_dubbing_add_delete_lines_v323625 import _view_model
from tests.unit.presentation.test_dubbing_line_edits_v323598 import app, page  # noqa: F401


def _fill(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "seg_000000.wav").write_bytes(b"x" * 1000)
    (folder / "seg_000001.wav").write_bytes(b"x" * 500)
    (folder / CACHE_FILENAME).write_text("{}", encoding="utf-8")
    (folder / TRASH_DIR_NAME).mkdir(exist_ok=True)
    (folder / TRASH_DIR_NAME / "a.wav").write_bytes(b"y" * 200)


def test_usage_and_clean_keep_master_and_foreign_files(tmp_path: Path) -> None:
    master = tmp_path / "ep.tts.wav"
    master.write_bytes(b"m" * 4000)
    srt = tmp_path / "ep.tts.srt"
    srt.write_text("1\n", encoding="utf-8")
    folder = tmp_path / "ep.tts_segments"
    _fill(folder)
    size, count = line_audio_usage(folder)
    assert count == 4 and size == 1000 + 500 + 2 + 200
    assert clean_line_audio(folder) == size
    assert not folder.exists()                       # rỗng → xoá luôn thư mục
    assert master.read_bytes() == b"m" * 4000 and srt.exists()
    _fill(folder)
    (folder / "ghi_chu.txt").write_text("của người dùng", encoding="utf-8")
    clean_line_audio(folder)
    assert [p.name for p in folder.iterdir()] == ["ghi_chu.txt"]  # tệp lạ không đụng tới
    assert line_audio_usage(tmp_path / "khong_co") == (0, 0)
    assert clean_line_audio(tmp_path / "khong_co") == 0


def test_view_model_clean_disables_resynth(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    vm, _ = _view_model(tmp_path, monkeypatch)
    folder = Path(vm._last_results[0].segment_audio_path).parent
    assert vm.can_resynthesize
    used = vm.line_audio_usage()
    assert used > 0
    vm._is_busy = True
    assert vm.clean_line_audio() == -1 and vm.can_resynthesize   # đang chạy: không dọn
    vm._is_busy = False
    assert vm.clean_line_audio() == used
    assert not vm.can_resynthesize
    assert all(not r.segment_audio_path for r in vm.last_results)
    assert not list(folder.glob("seg_*.wav"))
    assert vm._last_output_path.exists()              # master giữ nguyên
    assert vm.line_audio_usage() == 0


def test_page_button_confirms_then_cleans(
    page, tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    vm, _ = _view_model(tmp_path, monkeypatch)
    page._view_model = vm
    page._last_results = list(vm.last_results)
    page._refresh_line_audio_usage()
    assert page._btn_clean_line_audio.isEnabled()
    assert "MB" in page._btn_clean_line_audio.text()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    page._on_clean_line_audio()
    assert vm.can_resynthesize                        # huỷ xác nhận: không xoá gì
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    page._on_clean_line_audio()
    assert not vm.can_resynthesize
    assert not page._btn_clean_line_audio.isEnabled()
    assert not page._btn_resynth.isEnabled()


def test_segments_in_video_folder_count_only_app_files(tmp_path: Path) -> None:
    (tmp_path / "phim.mp4").write_bytes(b"v" * 9000)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "khac.wav").write_bytes(b"k" * 700)
    (tmp_path / "seg_000003.wav").write_bytes(b"x" * 300)
    assert line_audio_usage(tmp_path) == (300, 1)
    assert clean_line_audio(tmp_path) == 300
    assert (tmp_path / "phim.mp4").exists() and (tmp_path / "sub" / "khac.wav").exists()


def test_settings_page_round_trips_detector_override(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.application_settings import ApplicationSettings
    from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
    from subtitles_extractor.presentation.pages import settings_page as sp

    monkeypatch.setattr(ApplicationContainer, "preload_ocr_engine_async", lambda self: None)
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
        i18n_data_dir=Path(sp.__file__).resolve().parents[1] / "data", user_data_dir=tmp_path,
    )
    page_ = sp.SettingsPage(container)
    snapshot = ApplicationSettings()
    snapshot.ocr.detection_model_override = "PP-OCRv6_small_det"
    page_._populate_form(snapshot)
    assert page_._build_patch()["ocr"]["detection_model_override"] == "PP-OCRv6_small_det"
    # Rà soát độc lập: người dùng đổi phiên bản OCR → bộ dò chữ riêng về "Theo phiên bản" (không âm thầm ghép
    # bộ dò v6 với bộ nhận dạng của phiên bản khác).
    page_._cb_ocr_version.setCurrentIndex(page_._cb_ocr_version.findData("PP-OCRv5_server"))
    assert page_._build_patch()["ocr"]["detection_model_override"] == ""
    snapshot.ocr.detection_model_override = ""
    page_._populate_form(snapshot)
    assert page_._build_patch()["ocr"]["detection_model_override"] == ""


def test_new_source_forgets_previous_run_audio(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Rà soát độc lập: nạp phụ đề mới thì nút Dọn không được nhắm vào audio của lượt TTS cũ."""
    from tests.unit.presentation.test_dubbing_line_edits_v323598 import _events

    vm, _ = _view_model(tmp_path, monkeypatch)
    default_dir = tmp_path / "out_segments"                 # thư mục audio dòng mặc định của out.wav
    _fill(default_dir)
    assert vm.line_audio_usage() > 0
    vm.set_source_events(_events())
    assert vm.line_audio_usage() == 0
    assert vm.clean_line_audio() == 0
    assert list(default_dir.glob("seg_*.wav"))       # audio của lượt cũ còn nguyên


def test_leftover_resynth_temp_dirs_are_cleaned(tmp_path: Path) -> None:
    folder = tmp_path / "ep.tts_segments"
    _fill(folder)
    (folder / "resynth_seg_abc").mkdir()
    (folder / "resynth_seg_abc" / "seg_000000.wav").write_bytes(b"z" * 50)
    assert line_audio_usage(folder) == (1000 + 500 + 2 + 200 + 50, 5)
    clean_line_audio(folder)
    assert not folder.exists()
