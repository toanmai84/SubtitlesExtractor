"""[v3.23.593] Trang TTS, tab "🎭 Lồng tiếng": nạp vai, phân vai, giọng từng dòng.

Dữ liệu: bảng nhân vật + Visual Cues + mốc thời gian chép nguyên văn từ lượt dịch thật
v3.23.592 (``0c0c7c9a``), qua :mod:`tests.unit.application.test_dubbing_cast_v323593`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QComboBox

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.i18n.json_translator import JsonTranslator
from subtitles_extractor.infrastructure.tts.edge_voice_variants import edge_voice_catalog
from subtitles_extractor.presentation.pages.tts_dubbing import (
    SETTINGS_CASTING_PREFIX,
    TTSDubbingCoordinator,
)
from subtitles_extractor.presentation.view_models.tts_page_view_model import (
    TTSPageViewModel,
)
from subtitles_extractor.presentation.widgets.dubbing_cast_panel import DubbingCastPanel
from tests.unit.application.test_dubbing_cast_v323593 import ROWS, _source

DATA_DIR = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "data"
EDGE = edge_voice_catalog(["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"])


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "tts.ini"), QSettings.Format.IniFormat)


def _events() -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=position, text=text, interval=TimeInterval(start, final_end))
        for position, (_line, start, _e, final_end, _s, _t, text) in enumerate(ROWS, 1)
    ]


def _coordinator(
    settings: QSettings, catalog: list = EDGE, engine_key: str = "edge_vi-VN"
) -> tuple[TTSDubbingCoordinator, DubbingCastPanel, TTSPageViewModel]:
    translator = JsonTranslator(data_dir=DATA_DIR, default_locale="vi")
    panel = DubbingCastPanel(translator)
    view_model = TTSPageViewModel(SimpleNamespace())
    view_model.set_source_events(_events())
    coordinator = TTSDubbingCoordinator(
        panel, view_model, settings, translator.translate, lambda: (engine_key, catalog),
    )
    coordinator.set_source_provider(_source)
    return coordinator, panel, view_model


class TestDubbingTab:
    def test_off_by_default_means_one_voice(self, app, settings) -> None:
        coordinator, panel, _vm = _coordinator(settings)
        coordinator.load_cast()
        assert panel.has_cast()
        assert coordinator.line_voices() == ()

    def test_real_case_cast_and_line_voices(self, app, settings) -> None:
        coordinator, panel, _vm = _coordinator(settings)
        panel._enable_check.setChecked(True)
        coordinator.load_cast()
        # 5 vai nam dùng chung 3 giọng nam của Edge (thường/trầm/cao) + 1 vai nữ → 4
        # giọng.
        assert panel._status.text() == (
            "6 vai · 19/19 dòng biết người nói · 4 giọng khác nhau."
        )
        voices = coordinator.line_voices()
        casting = panel.casting()
        assert voices == tuple(casting[row[4]] for row in ROWS)
        assert casting["余小蓮"].startswith("vi-VN-HoaiMy")

    def test_user_choice_survives_reload(self, app, settings) -> None:
        coordinator, panel, _vm = _coordinator(settings)
        panel._enable_check.setChecked(True)
        coordinator.load_cast()
        row = [member.name for member in panel._members].index("老莫")
        combo = panel._table.cellWidget(row, 3)
        assert isinstance(combo, QComboBox)
        # [v3.23.594] Chọn giọng KHÁC giọng đang giao — chọn trùng thì không có gì để lưu
        # và test qua được cả khi việc lưu hỏng.
        chosen = next(p.voice_id for p in EDGE if p.voice_id != combo.currentData())
        combo.setCurrentIndex(combo.findData(chosen))
        again, again_panel, _ = _coordinator(settings)
        again.load_cast()
        assert again_panel.casting()["老莫"] == chosen
        assert again_panel.is_dubbing_enabled()

    def test_saved_voice_of_another_engine_is_not_applied(self, app, settings) -> None:
        coordinator, panel, _vm = _coordinator(settings)
        panel._enable_check.setChecked(True)
        coordinator.load_cast()
        coordinator.auto_cast()
        settings.setValue(f"{SETTINGS_CASTING_PREFIX}gemini", '{"老莫": "Charon"}')
        gemini, gemini_panel, _ = _coordinator(
            settings, catalog=[EDGE[0]], engine_key="gemini"
        )
        gemini.load_cast()
        assert gemini_panel.casting()["老莫"] == "vi-VN-HoaiMyNeural"

    def test_no_speaker_context_says_what_is_missing(self, app, settings) -> None:
        coordinator, panel, _vm = _coordinator(settings)
        coordinator.set_source_provider(lambda: _source().__class__())
        coordinator.load_cast()
        assert not panel.has_cast()
        assert "Visual Cues" in panel._status.text()

    def test_engine_without_voice_list_yet(self, app, settings) -> None:
        coordinator, panel, _vm = _coordinator(settings, catalog=[])
        panel._enable_check.setChecked(True)
        coordinator.load_cast()
        assert "chưa có danh sách giọng" in panel._status.text()
        assert coordinator.line_voices() == ()

    def test_view_model_puts_line_voices_into_the_request(self, app, settings) -> None:
        """``start_tts`` nhận ``line_voices`` và chuyển nguyên vào ``TTSRequest``."""
        import inspect

        signature = inspect.signature(TTSPageViewModel.start_tts)
        assert signature.parameters["line_voices"].default == ()
        source = inspect.getsource(TTSPageViewModel.start_tts)
        assert "line_voices=tuple(line_voices)" in source


class TestPageWiring:
    def test_tts_page_sends_line_voices_and_has_the_tab(self) -> None:
        import subtitles_extractor.presentation.pages.tts_page as page_module

        source = Path(page_module.__file__).read_text(encoding="utf-8")
        assert 'kwargs["line_voices"] = self._dubbing.line_voices()' in source
        assert 'self._translator.translate("tts.tab_dubbing")' in source
        assert "loader.catalog_loaded.connect(self._on_vieneu_catalog_loaded)" in source

    def test_main_window_links_the_translate_page(self) -> None:
        import subtitles_extractor.presentation.main_window as main_module

        source = Path(main_module.__file__).read_text(encoding="utf-8")
        assert "set_dubbing_source_provider(self._collect_dubbing_source)" in source
        assert "self._translate_page.get_speaker_context()" in source
