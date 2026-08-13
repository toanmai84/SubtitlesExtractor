"""Test [v3.23.51] mở thư mục sau khi xuất (an toàn, không crash)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.composition.bootstrap import bootstrap_for_gui


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_reveal_does_not_crash_on_nonexistent(app) -> None:
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage
    page = TranslatePage(bootstrap_for_gui())
    # Không tồn tại — không được ném ngoại lệ.
    page._reveal_in_explorer(Path("/nonexistent/path/file.srt"))


def test_reveal_handles_directory(app) -> None:
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage
    page = TranslatePage(bootstrap_for_gui())
    page._reveal_in_explorer(Path("/tmp"))
