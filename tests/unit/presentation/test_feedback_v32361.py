"""Test [v3.23.61] lớp thông báo chung (feedback) — không crash, đúng API."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PySide6.QtWidgets import QApplication, QWidget


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_all_feedback_functions_exist(app) -> None:
    from subtitles_extractor.presentation.theme import feedback
    for name in ("show_success", "show_info", "show_warning", "show_error"):
        assert hasattr(feedback, name)


def test_feedback_does_not_crash(app) -> None:
    from subtitles_extractor.presentation.theme import feedback
    parent = QWidget()
    feedback.show_success(parent, "Tiêu đề", "Nội dung")
    feedback.show_info(parent, "Tiêu đề")
    feedback.show_warning(parent, "Cảnh báo", "x")
    feedback.show_error(parent, "Lỗi", "y")


def test_durations_ordered(app) -> None:
    from subtitles_extractor.presentation.theme import feedback
    # Lỗi hiển thị lâu nhất, thông tin ngắn nhất.
    assert feedback._DURATION_ERROR >= feedback._DURATION_WARNING
    assert feedback._DURATION_WARNING >= feedback._DURATION_SUCCESS
    assert feedback._DURATION_SUCCESS >= feedback._DURATION_INFO
