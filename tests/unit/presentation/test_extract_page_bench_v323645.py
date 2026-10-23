"""[v3.23.645] Nút "Đo tốc độ OCR" của trang Trích xuất: khoá khi bận, báo khi chưa có video, mở hộp thoại khi đủ điều kiện."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox


@pytest.fixture
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _container(tmp: Path) -> Any:
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
    from subtitles_extractor.presentation.pages import editor_page

    return ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
        i18n_data_dir=Path(editor_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp,
    )


def test_bench_button_follows_busy_state_and_requires_a_video(
    app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from subtitles_extractor.presentation.pages.extract_page import ExtractPage
    from subtitles_extractor.presentation.widgets import ocr_benchmark_dialog

    page = ExtractPage(_container(tmp_path))
    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(str(a[2])))
    opened: list[object] = []
    monkeypatch.setattr(ocr_benchmark_dialog.OcrBenchmarkDialog, "exec", lambda self: opened.append(self) or 0)

    page._on_bench_clicked()                       # chưa mở video → thông báo, không mở hộp thoại
    assert shown and not opened

    page._on_busy_changed(True)
    assert not page._bench_button.isEnabled()
    page._on_busy_changed(False)
    assert page._bench_button.isEnabled()

    monkeypatch.setattr(type(page._view_model), "video", property(lambda self: object()))
    monkeypatch.setattr(page, "_run_preflight", lambda: True)
    page._on_bench_clicked()
    assert len(opened) == 1
    page.close()
