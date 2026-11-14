"""[v3.23.691] Tách tiếp trang Dịch (U6): xuất tệp và Visual Cues sang lớp trộn — hành vi không đổi."""

from __future__ import annotations

import logging

import pytest

from subtitles_extractor.presentation.pages import translate_cues, translate_export, translate_page
from subtitles_extractor.presentation.pages.translate_cues import TranslateCuesMixin
from subtitles_extractor.presentation.pages.translate_export import TranslateExportMixin
from subtitles_extractor.presentation.pages.translate_page import TranslatePage

EXPECTED = {
    TranslateExportMixin: {
        "_on_export_clicked", "_on_export_bilingual_clicked", "_on_export_diagnostics_clicked",
        "_reveal_in_explorer", "_on_export_context_diagnostics",
    },
    TranslateCuesMixin: {
        "_cue_source_rows", "_on_cue_recheck_clicked", "refresh_cue_issues", "_open_cue_table", "_exec_cue_dialog",
        "_with_user_cues", "apply_user_speakers", "apply_user_cues", "_visual_cues_valid",
    },
}


@pytest.mark.parametrize("mixin", list(EXPECTED))
def test_mixin_methods_are_complete_and_not_shadowed(mixin: type) -> None:
    assert mixin in TranslatePage.__mro__
    methods = {name for name in vars(mixin) if not name.startswith("__")}
    assert methods == EXPECTED[mixin]
    assert not methods & set(vars(TranslatePage))
    # Không hai lớp trộn nào cùng định nghĩa một tên (thứ tự kế thừa sẽ lặng lẽ chọn một bản).
    others = [m for m in EXPECTED if m is not mixin]
    for other in others:
        assert not methods & set(vars(other))


def test_static_method_stays_static() -> None:
    assert isinstance(vars(TranslateCuesMixin)["_exec_cue_dialog"], staticmethod)


def test_shared_constant_and_logger_unchanged() -> None:
    assert translate_page._LANGUAGE_CODES is translate_export._LANGUAGE_CODES
    assert translate_export._LANGUAGE_CODES["Vietnamese"] == "vi"
    for module in (translate_export, translate_cues):
        assert module.logger is logging.getLogger(translate_page.__name__)
