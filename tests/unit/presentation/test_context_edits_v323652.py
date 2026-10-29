"""[v3.23.652] Ô Visual Cues sửa tay hỏng JSON phải chặn dịch (không im lặng xoá cues); sửa tay ngữ cảnh phải được lưu."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services.visual_cue_serializer import visual_cues_syntax_error
from subtitles_extractor.composition.bootstrap import bootstrap_for_gui


@pytest.mark.parametrize(
    ("raw", "ok"),
    [
        ("", True),
        ("   ", True),
        (None, True),
        ('[{"start_ms": 0, "end_ms": 10, "description": "a"}]', True),
        ('[{"start_ms": 0 "end_ms": 10}]', False),  # thiếu dấu phẩy
        ('{"start_ms": 0}', False),  # không phải danh sách
    ],
)
def test_syntax_error(raw, ok) -> None:
    assert (visual_cues_syntax_error(raw) is None) is ok


def test_syntax_error_reports_line() -> None:
    error = visual_cues_syntax_error('[\n{"a": 1}\n{"b": 2}]')
    assert error is not None and "dòng 3" in error


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _make_page(app):
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage

    page = TranslatePage(bootstrap_for_gui())
    calls: list = []
    warnings: list = []
    page._view_model.start_translation = lambda **k: calls.append(k) or True
    page._show_warning = lambda t, c: warnings.append((t, c))
    return page, calls, warnings


def test_invalid_cues_block_translation(app) -> None:
    page, calls, warnings = _make_page(app)
    page._api_key_edit.setText("FAKE")
    page._stage_literal.enable_check.setChecked(True)
    page._visual_cues_edit.setPlainText('[{"start_ms": 0 "end_ms": 10}]')
    page._on_run_clicked()
    assert calls == []
    assert warnings and "dòng 1" in warnings[-1][1]


def test_user_edit_is_saved_but_programmatic_load_is_not(app) -> None:
    page, _calls, _warnings = _make_page(app)
    saved: list = []
    page._view_model.save_context_edits = lambda path, **k: saved.append((path, k))
    page._source_video_path = "/tmp/ep1.mp4"
    page._characters_edit.setPlainText("Lâm Phong — nam chính")  # nạp phiên: không được ghi ngược
    page._save_context_edits()
    assert saved == []
    page._characters_edit.document().setModified(True)  # như người dùng gõ
    page._save_context_edits()
    assert saved and saved[-1][0] == "/tmp/ep1.mp4"
    assert saved[-1][1]["characters"] == "Lâm Phong — nam chính"
    page._save_context_edits()  # không sửa thêm → không lưu lại
    assert len(saved) == 1


def test_broken_cues_not_autosaved(app) -> None:
    page, _calls, _warnings = _make_page(app)
    saved: list = []
    page._view_model.save_context_edits = lambda path, **k: saved.append(k)
    page._source_video_path = "/tmp/ep1.mp4"
    page._visual_cues_edit.setPlainText("[{")
    page._visual_cues_edit.document().setModified(True)
    page._save_context_edits()
    assert saved == []


def test_view_model_persists_to_session_store(tmp_path) -> None:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import TranslatePageViewModel

    stored: list = []

    class _Store:
        def save_analysis_texts(self, key, **kw):
            stored.append((key, kw))

    class _Container:
        translation_session_store = _Store()

    vm = TranslatePageViewModel.__new__(TranslatePageViewModel)
    vm._container = _Container()
    vm.save_context_edits("/v.mp4", characters="A", overview="", glossary="", visual_cues="")
    assert stored == [("/v.mp4", {"characters": "A", "overview": "", "glossary": "", "visual_cues": ""})]


def test_pending_edit_goes_to_the_video_it_was_typed_for(app) -> None:
    page, _calls, _warnings = _make_page(app)
    saved: list = []
    page._view_model.save_context_edits = lambda path, **k: saved.append((path, k))
    page._source_video_path = "/tmp/ep1.mp4"
    page._characters_edit.setPlainText("Nhân vật tập 1")
    page._characters_edit.document().setModified(True)
    page._on_context_text_changed()                 # người dùng gõ → hẹn giờ 1,5 s
    page.clear_context_fields()                     # đổi sang phim khác TRƯỚC khi hẹn giờ chạy
    assert saved and saved[-1][0] == "/tmp/ep1.mp4"
    assert saved[-1][1]["characters"] == "Nhân vật tập 1"


def test_clearing_all_boxes_is_saved(app) -> None:
    stored: list = []

    class _Store:
        def save_analysis_texts(self, key, **kw):
            stored.append(kw)

    from subtitles_extractor.presentation.view_models.translate_page_view_model import TranslatePageViewModel

    vm = TranslatePageViewModel.__new__(TranslatePageViewModel)
    vm._container = type("C", (), {"translation_session_store": _Store()})()
    vm.save_context_edits("/v.mp4", characters="", overview="", glossary="", visual_cues="")
    assert stored == [{"characters": "", "overview": "", "glossary": "", "visual_cues": ""}]
