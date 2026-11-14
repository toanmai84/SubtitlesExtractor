"""[v3.23.695] Tách nốt trang Lồng tiếng (U6): dựng giao diện, engine, bảng kết quả, duyệt nhanh, cài đặt."""

from __future__ import annotations

import ast
import logging

import pytest

from subtitles_extractor.presentation.pages import (
    tts_constants,
    tts_engines,
    tts_page,
    tts_results,
    tts_review,
    tts_settings,
    tts_ui,
    tts_voice_loader,
)
from subtitles_extractor.presentation.pages.tts_engines import TTSEnginesMixin
from subtitles_extractor.presentation.pages.tts_page import TTSPage
from subtitles_extractor.presentation.pages.tts_results import TTSResultsMixin
from subtitles_extractor.presentation.pages.tts_review import TTSReviewMixin
from subtitles_extractor.presentation.pages.tts_settings import TTSSettingsMixin
from subtitles_extractor.presentation.pages.tts_ui import TTSUiMixin
from tests.unit.page_sources import TTS_PAGE_SOURCE

EXPECTED = {
    TTSUiMixin: {
        "_make_config_tab", "_build_ui", "_build_dubbing_group", "_build_source_group", "_build_engine_selector",
        "_build_engine_stack", "_build_edge_panel", "_build_gemini_panel", "_build_vieneu_panel",
        "_build_common_config", "_build_output_group", "_build_actions_group", "_build_result_group",
        "_build_result_actions", "_wrap", "_row_one",
    },
    TTSEnginesMixin: {
        "_on_vieneu_catalog_loaded", "_on_engine_changed", "_sync_edge_only_controls", "_load_vieneu_voices",
        "_voice_loader_running", "_on_vieneu_voices_loaded", "_on_vieneu_voices_failed", "_reset_vieneu_voice_combo",
        "_vieneu_int8_tooltip", "_vieneu_precision", "_sync_vieneu_mode_controls", "_on_vieneu_mode_changed",
        "_on_gemini_model_changed", "_update_edge_speakers", "_on_browse_vieneu_ref", "_refresh_install_hint",
        "_on_engine_available",
    },
    TTSResultsMixin: {
        "_selected_event_indices", "_result_by_event_index", "_on_result_selection_changed", "_on_result_item_edited",
        "_on_play_selected", "_play_audio_file", "_make_preview_wav", "_on_media_status", "_on_media_error",
        "_commit_active_edit", "_populate_table", "_apply_row_filter",
    },
    TTSReviewMixin: {
        "_install_review_shortcuts", "_speaker_review_rows", "_schedule_review_count", "review_count",
        "_review_event_indices", "_jump_to_review_line", "_abnormal_event_indices", "_on_select_abnormal",
    },
    TTSSettingsMixin: {"_save_settings", "_restore_settings", "_on_reset_defaults"},
}


@pytest.mark.parametrize("mixin", list(EXPECTED))
def test_mixin_methods_are_complete_and_not_shadowed(mixin: type) -> None:
    assert mixin in TTSPage.__mro__
    methods = {name for name in vars(mixin) if not name.startswith("__")}
    assert methods == EXPECTED[mixin]
    assert not methods & set(vars(TTSPage))
    for other in TTSPage.__mro__[1:]:
        if other is not mixin and other.__name__.endswith("Mixin"):
            assert not methods & set(vars(other)), other


def test_old_names_still_importable_from_tts_page() -> None:
    assert tts_page._VieNeuVoiceLoader is tts_voice_loader._VieNeuVoiceLoader
    for name in ("_DEFAULT_MAX_SPEED", "_OVERLAP_WARN_S", "_EDGE_ONLY_HINT", "_ENGINES", "_EDGE_LANGUAGES",
                 "_EDGE_LANGUAGE_DISPLAY", "_ENGINE_VIENEU"):
        assert getattr(tts_page, name) is getattr(tts_constants, name)
    assert tts_page._DEFAULT_MAX_SPEED == 1.5 and tts_page._OVERLAP_WARN_S == 0.5
    assert [engine for engine, _label, _desc in tts_page._ENGINES] == ["edge", "gemini", "vieneu"]
    assert "vi-VN" in tts_page._EDGE_LANGUAGES


def test_split_modules_keep_page_logger() -> None:
    page_logger = logging.getLogger(tts_page.__name__)
    for module in (tts_ui, tts_engines, tts_results, tts_review, tts_settings, tts_voice_loader):
        assert getattr(module, "logger", page_logger) is page_logger


def test_joined_source_parses_and_has_every_method_once() -> None:
    tree = ast.parse(TTS_PAGE_SOURCE.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    assert {"_VieNeuVoiceLoader", "TTSPage"} <= {c.name for c in classes}
    page = [f.name for c in classes if c.name == "TTSPage" or c.name.endswith("Mixin")
            for f in c.body if isinstance(f, ast.FunctionDef)]
    assert len(page) == len(set(page))
