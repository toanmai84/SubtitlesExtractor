"""[v3.23.694] Tách tiếp trang Dịch (U6): dựng giao diện, key/hạn mức, model, phân tích, bảng kết quả, công cụ."""

from __future__ import annotations

import ast
import logging

import pytest

from subtitles_extractor.presentation.pages import (
    translate_analysis,
    translate_constants,
    translate_keys,
    translate_models,
    translate_page,
    translate_table,
    translate_tools,
    translate_ui,
    translate_widgets,
)
from subtitles_extractor.presentation.pages.translate_analysis import TranslateAnalysisMixin
from subtitles_extractor.presentation.pages.translate_keys import TranslateKeysMixin
from subtitles_extractor.presentation.pages.translate_models import TranslateModelsMixin
from subtitles_extractor.presentation.pages.translate_page import TranslatePage
from subtitles_extractor.presentation.pages.translate_table import TranslateTableMixin
from subtitles_extractor.presentation.pages.translate_tools import TranslateToolsMixin
from subtitles_extractor.presentation.pages.translate_ui import TranslateUiMixin
from tests.unit.page_sources import TRANSLATE_PAGE_SOURCE

EXPECTED = {
    TranslateUiMixin: {
        "_make_config_tab", "_build_ui", "_build_source_group", "_build_ai_config_group", "_build_context_group",
        "_build_video_context_group", "_build_stages_group", "_build_actions_group", "_build_result_group",
        "_wrap_layout",
    },
    TranslateKeysMixin: {"_all_keys", "_combined_keys_text", "_masked_key", "_keys_quota_status_text",
                         "_on_edit_quota_clicked"},
    TranslateModelsMixin: {
        "_on_refresh_models_clicked", "_fallback_selection", "_fallback_choice", "_on_fallback_enabled_toggled",
        "_fill_fallback_combo", "_audit_choice", "_fill_audit_combo", "_on_audit_model_changed",
        "_on_fallback_model_changed", "_on_models_ready", "_custom_mode", "_custom_provider_ready",
        "_sync_custom_provider", "_warn_custom_incomplete", "shorten_model",
        "_model_combos", "_model_snapshot", "_apply_model_snapshot", "_swap_provider_models",   # [v3.23.700]
    },
    TranslateAnalysisMixin: {
        "_on_analyze_context_clicked", "_fill_context_fields", "_on_analyze_context_ready",
        "_on_visual_cues_topped_up", "_on_analysis_restored", "_on_refine_analysis_clicked",
        "_update_speaker_button", "_on_complete_speakers_clicked", "_on_speaker_completion_done",
        "_on_analyze_error",
    },
    TranslateTableMixin: {
        "_on_column_resized", "_on_table_selection_changed", "_on_copy_selection_clicked", "_on_table_context_menu",
        "_on_retranslate_rows", "_on_edit_translation_row", "_copy_column", "_mark_edited_row",
        "_on_send_to_editor_clicked",
    },
    TranslateToolsMixin: {
        "_on_manage_tm_clicked", "_on_check_glossary_clicked", "_on_compare_stages_clicked",
        "_on_clear_cache_clicked", "_on_clear_cloud_clicked", "_on_stage_cache_clicked",
    },
}


@pytest.mark.parametrize("mixin", list(EXPECTED))
def test_mixin_methods_are_complete_and_not_shadowed(mixin: type) -> None:
    assert mixin in TranslatePage.__mro__
    methods = {name for name in vars(mixin) if not name.startswith("__")}
    assert methods == EXPECTED[mixin]
    assert not methods & set(vars(TranslatePage))
    for other in TranslatePage.__mro__[1:]:
        if other is not mixin and other.__name__.endswith("Mixin"):
            assert not methods & set(vars(other)), other


def test_old_names_still_importable_from_translate_page() -> None:
    assert translate_page._StagePanel is translate_widgets._StagePanel
    assert translate_page.parse_quota_rows is translate_widgets.parse_quota_rows
    assert translate_page.replace_model_items is translate_widgets.replace_model_items
    for name in ("_TARGET_LANGUAGES", "_DEFAULT_MODELS", "NO_FALLBACK_CHOICE", "DEFAULT_REFINE_MODEL",
                 "_STYLE_PRESETS", "_RESIZE_ROWS_THRESHOLD", "EDITS_ONLY_DEFAULT"):
        assert getattr(translate_page, name) is getattr(translate_constants, name)
    assert translate_page.DEFAULT_REFINE_MODEL == "gemini-3.1-flash-lite"
    assert translate_page.NO_FALLBACK_CHOICE == "-" and translate_page.EDITS_ONLY_DEFAULT is True


def test_settings_keys_unchanged() -> None:
    # Đổi hai tên này sẽ mất cài đặt đã lưu của người dùng (QSettings).
    assert (translate_constants._SETTINGS_ORG, translate_constants._SETTINGS_APP) == ("SubtitlesExtractor",
                                                                                      "TranslatePage")


def test_split_modules_keep_page_logger() -> None:
    page_logger = logging.getLogger(translate_page.__name__)
    for module in (translate_ui, translate_keys, translate_models, translate_analysis, translate_table,
                   translate_tools):
        assert getattr(module, "logger", page_logger) is page_logger


def test_joined_source_parses_and_has_every_method_once() -> None:
    tree = ast.parse(TRANSLATE_PAGE_SOURCE.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    page = [f.name for c in classes if c.name == "TranslatePage" or c.name.endswith("Mixin")
            for f in c.body if isinstance(f, ast.FunctionDef)]
    assert len(page) == len(set(page))
