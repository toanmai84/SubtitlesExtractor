"""[v3.23.620] Dịch thô mặc định bật Thinking "low" (đo: bẫy xưng hô 0-7/14 → 11/14)."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationStageKind


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


class _Tr:
    def translate(self, key: str) -> str:
        return key


def _panel(**kw):
    from subtitles_extractor.presentation.pages.translate_page import _StagePanel

    return _StagePanel(
        "t", TranslationStageKind.LITERAL, translator=_Tr(), default_enabled=True,
        default_model="gemini-3.1-flash-lite", default_temp=0.1, default_batch=50,
        default_ctx=10, **kw,
    )


def test_default_thinking_enables_low_level(app: QApplication) -> None:
    panel = _panel(default_thinking=True)
    config = panel.build_config()
    assert panel.thinking_check.isChecked()
    assert panel.thinking_level_combo.isEnabled()
    assert panel.thinking_level_combo.currentData() == "low"
    assert config.enable_thinking and config.thinking_level == "low"


def test_panel_default_still_off(app: QApplication) -> None:
    panel = _panel()
    assert not panel.thinking_check.isChecked()
    assert not panel.thinking_level_combo.isEnabled()
