"""[v3.23.693] Tách trang Chỉnh sửa (U6): hộp thoại, Re-OCR, tìm/thay/gộp, xuất/xem trước ASS — hành vi không đổi."""

from __future__ import annotations

import ast

import pytest

from subtitles_extractor.presentation.pages import (
    editor_constants,
    editor_dialogs,
    editor_export,
    editor_find_replace,
    editor_page,
    editor_reocr,
)
from subtitles_extractor.presentation.pages.editor_export import EditorExportMixin
from subtitles_extractor.presentation.pages.editor_find_replace import EditorFindReplaceMixin
from subtitles_extractor.presentation.pages.editor_page import EditorPage
from subtitles_extractor.presentation.pages.editor_reocr import EditorReOcrMixin
from tests.unit.editor_page_source import EDITOR_PAGE_SOURCE

EXPECTED = {
    EditorReOcrMixin: {
        "_on_video_roi_changed", "_on_reocr_progress", "set_extraction_running", "_refresh_reocr_buttons",
        "_warn_extraction_running", "_on_reocr_busy", "_on_reocr_clicked", "_on_fast_reocr_clicked", "cancel_reocr",
    },
    EditorFindReplaceMixin: {
        "_apply_filters", "_find_next", "_find_prev", "_find", "_replace_one", "_replace_in_text_safe",
        "_replace_all", "_on_auto_fix_timeline", "_on_merge_similar_dialog",
    },
    EditorExportMixin: {
        "_on_export_clicked", "_do_export_clicked", "_on_export_finished", "_schedule_preview",
        "_generate_ass_preview", "_cleanup_old_preview_tracks", "_build_ass_content",
    },
}


@pytest.mark.parametrize("mixin", list(EXPECTED))
def test_mixin_methods_are_complete_and_not_shadowed(mixin: type) -> None:
    assert mixin in EditorPage.__mro__
    methods = {name for name in vars(mixin) if not name.startswith("__")}
    assert methods == EXPECTED[mixin]
    assert not methods & set(vars(EditorPage))
    for other in EXPECTED:
        if other is not mixin:
            assert not methods & set(vars(other))


def test_static_helper_keeps_staticmethod() -> None:
    assert isinstance(vars(EditorFindReplaceMixin)["_replace_in_text_safe"], staticmethod)
    assert EditorPage._replace_in_text_safe("a", "b", "aaa", 1) == "baa"


def test_old_names_still_importable_from_editor_page() -> None:
    # Mã/plug-in cũ nhập từ editor_page vẫn chạy.
    assert editor_page.AdvancedReOcrDialog is editor_dialogs.AdvancedReOcrDialog
    assert editor_page.EditorSettingsDialog is editor_dialogs.EditorSettingsDialog
    assert editor_page._DetachedVideoWindow is editor_dialogs._DetachedVideoWindow
    assert editor_page._fmt_mmss(125.4) == "02:05"
    assert editor_page._SLIDER_RESOLUTION == editor_constants._SLIDER_RESOLUTION == 1000
    assert editor_reocr.AdvancedReOcrDialog is editor_dialogs.AdvancedReOcrDialog


def test_split_modules_log_with_loguru_like_the_page() -> None:
    assert editor_export.logger is editor_page.logger
    # Hai phần còn lại không ghi log — không giữ import thừa.
    assert not hasattr(editor_reocr, "logger") and not hasattr(editor_find_replace, "logger")


def test_joined_source_parses_and_has_every_method_once() -> None:
    tree = ast.parse(EDITOR_PAGE_SOURCE.read_text())
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    assert {"AdvancedReOcrDialog", "EditorSettingsDialog", "_DetachedVideoWindow", "EditorPage"} <= set(classes)
    page_methods = [f.name for name in ("EditorPage", "EditorReOcrMixin", "EditorFindReplaceMixin", "EditorExportMixin")
                    for f in classes[name].body if isinstance(f, ast.FunctionDef)]
    assert len(page_methods) == len(set(page_methods))
