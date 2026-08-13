"""Test [v3.23.44] validate sớm tại UI trang dịch (key trống, không có stage)."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.composition.bootstrap import bootstrap_for_gui


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _make_page(app):
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage
    page = TranslatePage(bootstrap_for_gui())
    calls = []
    page._view_model.start_translation = lambda **k: calls.append(k) or True
    return page, calls


def test_empty_api_key_blocks_translation(app) -> None:
    page, calls = _make_page(app)
    page._api_key_edit.setText("")
    page._stage_literal.enable_check.setChecked(True)
    page._on_run_clicked()
    assert calls == []  # không gọi dịch khi thiếu key


def test_no_stage_blocks_translation(app) -> None:
    page, calls = _make_page(app)
    page._api_key_edit.setText("FAKE")
    for panel in (page._stage_preprocess, page._stage_literal,
                  page._stage_style, page._stage_localize):
        panel.enable_check.setChecked(False)
    page._on_run_clicked()
    assert calls == []  # không gọi dịch khi không bật stage nào


def test_valid_inputs_proceed(app) -> None:
    page, calls = _make_page(app)
    page._api_key_edit.setText("FAKE")
    page._stage_literal.enable_check.setChecked(True)
    page._on_run_clicked()
    assert len(calls) == 1
    assert calls[0]["api_key"] == "FAKE"


def test_video_enabled_without_file_blocks(app) -> None:
    page, calls = _make_page(app)
    page._api_key_edit.setText("FAKE")
    page._stage_literal.enable_check.setChecked(True)
    page._attach_video_check.setChecked(True)
    page._video_stage_literal.setChecked(True)
    page._video_path = ""  # chưa chọn file
    page._on_run_clicked()
    assert calls == []  # cảnh báo, không dịch


def test_video_enabled_with_file_proceeds(app) -> None:
    page, calls = _make_page(app)
    page._api_key_edit.setText("FAKE")
    page._stage_literal.enable_check.setChecked(True)
    page._attach_video_check.setChecked(True)
    page._video_stage_literal.setChecked(True)
    page._video_path = "/fake/video.mp4"
    page._on_run_clicked()
    assert len(calls) == 1
