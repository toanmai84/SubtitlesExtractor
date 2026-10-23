"""[v3.23.596] Giao diện lồng tiếng: nói rõ khi giọng đã lưu ghi đè phân vai tự động;
"Elastic timing" / "Nén êm" mở cho VieNeu và Gemini.

Lượt thật 10/10/2026: 19 vai mà chỉ 2 giọng (Minh Đức cho mọi vai nam và vai chưa rõ giới,
Ngọc Huyền cho mọi vai nữ). Chạy lại đúng phụ đề + Visual Cues của lượt đó qua phân vai tự
động cho 19 giọng khác nhau — 2 giọng đến từ lựa chọn đã lưu, vốn được áp lặng lẽ.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from subtitles_extractor.presentation.pages.tts_dubbing import SETTINGS_CASTING_PREFIX
from tests.unit.presentation.test_dubbing_bugfixes_v323594 import (
    ENGINE_KEY,
    _coordinator,
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "tts.ini"), QSettings.Format.IniFormat)


def _one_voice_for_everyone(settings: QSettings) -> int:
    coordinator, panel, _view_model = _coordinator(settings)
    assert coordinator.load_cast()
    names = list(panel.casting())
    first_voice = next(voice for voice in panel.casting().values() if voice)
    settings.setValue(
        f"{SETTINGS_CASTING_PREFIX}{ENGINE_KEY}",
        json.dumps(dict.fromkeys(names, first_voice), ensure_ascii=False),
    )
    auto = panel.casting()
    return sum(1 for name in names if auto[name] != first_voice)


def test_status_says_when_saved_voices_override_auto_cast(
    app: QApplication, settings: QSettings
) -> None:
    overridden = _one_voice_for_everyone(settings)
    assert overridden > 0
    coordinator, panel, _view_model = _coordinator(settings)
    assert coordinator.load_cast()
    assert len({voice for voice in panel.casting().values() if voice}) == 1
    assert f"{overridden} vai đang dùng giọng bạn chọn từ trước" in panel.status()

    coordinator.auto_cast()
    assert "giọng bạn chọn từ trước" not in panel.status()
    assert len({voice for voice in panel.casting().values() if voice}) > 1


def test_fresh_cast_has_no_override_note(app: QApplication, settings: QSettings) -> None:
    coordinator, panel, _view_model = _coordinator(settings)
    assert coordinator.load_cast()
    assert "giọng bạn chọn từ trước" not in panel.status()


def test_elastic_controls_enabled_for_vieneu_and_gemini(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import tts_page as tts_page_module

    monkeypatch.setattr(
        tts_page_module, "QSettings",
        lambda *_args: QSettings(str(tmp_path / "page.ini"), QSettings.Format.IniFormat),
    )
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "config.json"),
        i18n_data_dir=Path(tts_page_module.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp_path,
    )
    page = tts_page_module.TTSPage(container)
    for engine in (tts_page_module._ENGINE_VIENEU, tts_page_module._ENGINE_GEMINI):
        page._engine_combo.setCurrentIndex(page._engine_combo.findData(engine))
        assert page._elastic_timing.isEnabled()
        assert page._comfort_ratio.isEnabled()
        assert not page._double_pass.isEnabled()  # vẫn chỉ Edge
