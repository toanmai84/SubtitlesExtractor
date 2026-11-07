"""[v3.23.681] Ô "Chỉ trả dòng cần sửa" chỉ có ở giai đoạn tinh chỉnh / bản địa hoá, lưu/khôi phục được."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationStageKind


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


class _Tr:
    def translate(self, key: str) -> str:
        return key


def _panel(kind, **kw):
    from subtitles_extractor.presentation.pages.translate_page import _StagePanel

    return _StagePanel(
        "t", kind, translator=_Tr(), default_enabled=True, default_model="gemini-3.5-flash-lite",
        default_temp=0.3, default_batch=40, default_ctx=12, **kw,
    )


def test_only_refine_stages_offer_edits_only(app: QApplication) -> None:
    from subtitles_extractor.presentation.pages.translate_page import EDITS_ONLY_DEFAULT

    literal = _panel(TranslationStageKind.LITERAL)
    style = _panel(TranslationStageKind.STYLE, show_style=True)
    localize = _panel(TranslationStageKind.LOCALIZE, show_locale=True)
    assert literal.edits_only_check is None and literal.build_config().edits_only is False
    for panel in (style, localize):
        assert panel.edits_only_check is not None
        assert panel.build_config().edits_only is EDITS_ONLY_DEFAULT


def test_edits_only_saved_and_restored(app: QApplication, tmp_path) -> None:
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    style = _panel(TranslationStageKind.STYLE, show_style=True)
    style.edits_only_check.setChecked(False)
    style.save_state(settings, "style")
    again = _panel(TranslationStageKind.STYLE, show_style=True)
    again.restore_state(settings, "style")
    assert again.build_config().edits_only is False
