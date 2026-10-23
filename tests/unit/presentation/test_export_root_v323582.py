"""[v3.23.582] Ba nút xuất (chuẩn đoán phân tích/dịch, nhật ký) xử lý NHƯ NHAU.

Người dùng phản ánh: "xuất chuẩn đoán phân tích và xuất chuẩn đoán dịch rồi xuất file nhật
ký có vẻ không nhất quán ... lưu không cùng một thư mục gốc", và "xuất chuẩn đoán dịch mỗi
lần lại mở thư mục lên thật phiền phức". v3.23.581:

* chuẩn đoán phân tích: tên trơn → hộp thoại mở ở thư mục làm việc hiện tại;
* chuẩn đoán dịch: cạnh file video, rồi TỰ MỞ thư mục sau khi lưu;
* nhật ký: "app_log.txt" (ghi đè lần trước) qua hộp thoại HỆ ĐIỀU HÀNH (nhớ thư mục lần
  trước; cũng là loại hộp thoại từng gây treo khi có MPV — EXPORT-HANG v3.7).

Nay: vẫn hỏi nơi lưu mỗi lần, nhưng cả ba mở ở CÙNG thư mục gốc của ứng dụng, tên có
timestamp, hộp thoại Qt thuần, không tự mở thư mục.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QFileDialog

import subtitles_extractor

#: Thư mục dự án khi chạy từ mã nguồn: <dự án>/src/subtitles_extractor/__init__.py.
PROJECT_ROOT = Path(subtitles_extractor.__file__).resolve().parents[2]
VIDEO = "/films/he thong sung the.mp4"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict]:
    """Thay hộp thoại lưu: ghi lại tham số, trả về file trong ``tmp_path``."""
    calls: list[dict] = []

    def _fake_save(parent, caption, directory="", file_filter="", **kwargs):
        calls.append({"directory": directory, "options": kwargs.get("options")})
        return str(tmp_path / Path(directory).name), file_filter

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(_fake_save))
    return calls


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, *args) -> None:
        self.calls.append(args)


def _text(value: str) -> SimpleNamespace:
    return SimpleNamespace(text=lambda: value, toPlainText=lambda: value,
                           currentText=lambda: value, currentData=lambda: value)


def _translation_page_stub(tmp_path: Path) -> SimpleNamespace:
    def _export(path: Path) -> Path:
        path.write_text("{}", encoding="utf-8")
        return path

    return SimpleNamespace(
        _view_model=SimpleNamespace(
            has_result=True, export_translation_diagnostics=_export,
            source_events=[], name_readings={},
        ),
        _source_video_path=VIDEO,
        # [v3.23.609] Video của phụ đề lấy qua một chỗ (cả khi không đính video).
        _session_video_path=lambda: VIDEO,
        _translator=SimpleNamespace(translate=lambda key: key),
        _show_success=_Recorder(), _show_warning=_Recorder(), _show_info=_Recorder(),
        _reveal_in_explorer=_Recorder(),
        _lang_combo=_text("Vietnamese"), _source_lang_edit=_text("Chinese"),
        _overview_edit=_text("Lâm Dật nhận được hệ thống hoàn tiền."),
        _characters_edit=_text(""), _glossary_edit=_text(""), _visual_cues_edit=_text(""),
        _tags_check=SimpleNamespace(isChecked=lambda: False),
        _desc_check=SimpleNamespace(isChecked=lambda: False),
        _attach_video_check=SimpleNamespace(isChecked=lambda: False),
        _analyze_model_combo=_text("gemini"), _media_res_combo=_text(""),
        _analysis_think_combo=_text(""),
    )


def _export_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    from subtitles_extractor.infrastructure.logging.qt_log_handler import QtLogBridge
    from subtitles_extractor.presentation.pages import log_page as log_module
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage

    page_stub = _translation_page_stub(tmp_path)
    TranslatePage._on_export_context_diagnostics(page_stub)
    TranslatePage._on_export_diagnostics_clicked(page_stub)
    translator = SimpleNamespace(translate=lambda key: key)
    log_page = log_module.LogPage(QtLogBridge(), translator)
    monkeypatch.setattr(log_module._feedback, "show_success", lambda *args: None)
    log_page._on_export()
    return page_stub


class TestSameRootForAllThree:
    def test_all_dialogs_open_in_the_application_root(
        self, app, dialog_calls: list[dict], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _export_all(tmp_path, monkeypatch)
        assert [Path(call["directory"]).parent for call in dialog_calls] == [
            PROJECT_ROOT, PROJECT_ROOT, PROJECT_ROOT
        ]

    def test_names_carry_a_timestamp(
        self, app, dialog_calls: list[dict], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _export_all(tmp_path, monkeypatch)
        names = [Path(call["directory"]).name for call in dialog_calls]
        stamp = r"\.\d{8}_\d{6}\."
        film = r"he thong sung the"
        assert re.fullmatch(rf"{film}\.context_diagnostics{stamp}json", names[0])
        assert re.fullmatch(rf"{film}\.translation_debug{stamp}json", names[1])
        assert re.fullmatch(rf"app_log{stamp}txt", names[2])

    def test_all_dialogs_are_qt_dialogs(
        self, app, dialog_calls: list[dict], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _export_all(tmp_path, monkeypatch)
        assert [call["options"] for call in dialog_calls] == [
            QFileDialog.Option.DontUseNativeDialog
        ] * 3

    def test_translation_diagnostics_no_longer_opens_the_folder(
        self, app, dialog_calls: list[dict], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        page_stub = _export_all(tmp_path, monkeypatch)
        assert page_stub._reveal_in_explorer.calls == []
        assert len(page_stub._show_success.calls) == 1
        saved = sorted(path.suffix for path in tmp_path.iterdir())
        assert saved == [".json", ".json", ".txt"]


class TestApplicationRootDir:
    def test_packaged_build_uses_the_executable_folder(self, tmp_path: Path) -> None:
        from subtitles_extractor.infrastructure.app_paths import application_root_dir

        executable = tmp_path / "SubtitlesExtractor" / "SubtitlesExtractor.exe"
        assert application_root_dir(frozen=True, executable=str(executable)) == (
            executable.parent.resolve()
        )

    def test_source_run_uses_the_project_folder(self, tmp_path: Path) -> None:
        from subtitles_extractor.infrastructure.app_paths import application_root_dir

        package = tmp_path / "proj" / "src" / "subtitles_extractor" / "__init__.py"
        assert application_root_dir(frozen=False, package_file=str(package)) == (
            (tmp_path / "proj").resolve()
        )
        assert application_root_dir(frozen=False) == PROJECT_ROOT

    def test_model_store_shares_the_packaged_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import sys

        from subtitles_extractor.infrastructure import model_store
        from subtitles_extractor.infrastructure.app_paths import application_root_dir

        executable = tmp_path / "dist" / "SubtitlesExtractor.exe"
        monkeypatch.setattr(sys, "executable", str(executable))
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        assert model_store._frozen_runtime_root() == (
            application_root_dir(frozen=True) / "models"
        )
        assert application_root_dir() == executable.parent.resolve()


class TestAppLogFilename:
    def test_fixed_time(self) -> None:
        from datetime import datetime

        from subtitles_extractor.application.services.diagnostic_naming import (
            build_app_log_filename,
        )

        assert build_app_log_filename(now=datetime(2026, 10, 1, 8, 49, 35)) == (
            "app_log.20261001_084935.txt"
        )
