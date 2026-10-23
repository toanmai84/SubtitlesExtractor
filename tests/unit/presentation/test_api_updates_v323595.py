"""[v3.23.595] Giao diện theo cập nhật API: VieNeu int8/v3 Turbo, Gemini 3.8, PP-OCRv6.

* Ô "CPU nhanh — int8" chỉ bật ở v3 Turbo và đi tới tận adapter.
* VieNeu 3.8 bỏ ``style`` ở v3 Turbo → ô sắc thái tắt, luôn gửi "natural" (dùng chung
  engine đã nạp vì khoá cache gồm sắc thái).
* Trang Gỡ lỗi OCR: combo model chỉ có v5/v4 trong khi model thật là PP-OCRv6 → hiện sai
  model đang dùng.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.settings.json_file_repository import (
    JsonFileRepository,
)
from subtitles_extractor.infrastructure.tts import cpu_features
from subtitles_extractor.presentation.pages import translate_page
from subtitles_extractor.presentation.pages import tts_page as tts_page_module
from subtitles_extractor.presentation.view_models.tts_page_view_model import (
    TTSPageViewModel,
)

DATA_DIR = Path(tts_page_module.__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def container(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ApplicationContainer:
    ini_path = tmp_path / "tts_page.ini"
    monkeypatch.setattr(
        tts_page_module, "QSettings",
        lambda *_args: QSettings(str(ini_path), QSettings.Format.IniFormat),
    )
    return ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "config.json"),
        i18n_data_dir=DATA_DIR, user_data_dir=tmp_path,
    )


def _select(combo: Any, data: str) -> None:
    index = combo.findData(data)
    assert index >= 0, data
    combo.setCurrentIndex(index)


def _tts_page(container: ApplicationContainer) -> Any:
    return tts_page_module.TTSPage(container)


class TestVieNeuOptions:
    def test_int8_only_on_v3turbo(self, qapp: QApplication, container: Any) -> None:
        page = _tts_page(container)
        _select(page._vieneu_mode, "v3turbo")
        assert page._vieneu_int8.isEnabled()
        assert not page._vieneu_emotion.isEnabled()
        assert page._vieneu_precision() == ""
        page._vieneu_int8.setChecked(True)
        assert page._vieneu_precision() == "int8"
        for mode in ("standard", "turbo"):
            _select(page._vieneu_mode, mode)
            assert not page._vieneu_int8.isEnabled()
            assert page._vieneu_emotion.isEnabled()
            assert page._vieneu_precision() == ""

    @pytest.mark.parametrize(("support", "key"), [
        (True, "tts.vnni_yes"), (False, "tts.vnni_no"), (None, "tts.vnni_unknown"),
    ])
    def test_tooltip_reports_vnni(
        self, qapp: QApplication, container: Any, monkeypatch: pytest.MonkeyPatch,
        support: bool | None, key: str,
    ) -> None:
        monkeypatch.setattr(cpu_features, "cpu_vnni_support", lambda: support)
        page = _tts_page(container)
        assert container.translator.translate(key) in page._vieneu_int8.toolTip()

    def test_generate_sends_precision_and_natural_style(
        self, qapp: QApplication, container: Any, monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        page = _tts_page(container)
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            page._view_model, "start_tts", lambda **kwargs: captured.update(kwargs)
        )
        page._view_model.set_source_events([
            SubtitleEvent(index=1, text="Trưởng đồn.", interval=TimeInterval(1.0, 2.0)),
        ])
        _select(page._engine_combo, tts_page_module._ENGINE_VIENEU)
        page._engine_avail = {tts_page_module._ENGINE_VIENEU: True}
        _select(page._vieneu_mode, "v3turbo")
        _select(page._vieneu_emotion, "storytelling")
        page._vieneu_int8.setChecked(True)
        page._output.setText(str(tmp_path / "out.wav"))
        page._on_generate()
        assert captured["vieneu_precision"] == "int8"
        assert captured["vieneu_emotion"] == "natural"

        captured.clear()
        _select(page._vieneu_mode, "standard")
        page._on_generate()
        assert captured["vieneu_precision"] == ""
        assert captured["vieneu_emotion"] == "storytelling"

    def test_int8_choice_is_remembered(self, qapp: QApplication, container: Any) -> None:
        page = _tts_page(container)
        _select(page._vieneu_mode, "v3turbo")
        page._vieneu_int8.setChecked(True)
        page._save_settings()
        assert _tts_page(container)._vieneu_int8.isChecked()

    def test_view_model_builds_adapter_with_precision(self) -> None:
        made: dict[str, Any] = {}

        def make_vieneu_tts_adapter(**kwargs: Any) -> Any:
            made.update(kwargs)
            return SimpleNamespace(
                is_available=lambda: False, get_engine_name=lambda: "x"
            )

        view_model = TTSPageViewModel(
            SimpleNamespace(make_vieneu_tts_adapter=make_vieneu_tts_adapter)
        )
        view_model.set_source_events([
            SubtitleEvent(index=1, text="Trưởng đồn.", interval=TimeInterval(1.0, 2.0)),
        ])
        view_model.start_tts(
            engine="vieneu", output_path=Path("out.wav"), vieneu_mode="v3turbo",
            vieneu_precision="int8",
        )
        assert made["precision"] == "int8"


class TestGeminiModels:
    def test_tts_page_defaults_to_38_flash_lite(
        self, qapp: QApplication, container: Any
    ) -> None:
        page = _tts_page(container)
        assert page._gemini_model.itemText(0) == "gemini-3.8-flash-lite-tts"
        assert page._gemini_model.findText("gemini-3.8-flash-tts") >= 0

    def test_translate_page_offers_38_flash(self) -> None:
        models = translate_page._DEFAULT_MODELS
        assert "gemini-3.8-flash" in models
        assert models.index("gemini-3.8-flash") < models.index("gemini-3.6-flash")
        # [v3.23.622] Bí danh đã ngừng không còn được mời chọn.
        assert "gemini-3.7-flash" not in models and "gemini-3.5-flash" not in models


class TestDebugPageModel:
    def test_combo_shows_ppocrv6_and_follows_settings(
        self, qapp: QApplication, container: Any
    ) -> None:
        from subtitles_extractor.presentation.pages.debug_page import DebugPage

        page = DebugPage(container)
        combo = page._cb_model
        assert combo.itemData(0) == "PP-OCRv6_medium"
        for name in ("PP-OCRv6_small", "PP-OCRv6_tiny", "PP-OCRv5_server"):
            assert combo.findData(name) >= 0
        assert not combo.isEnabled()
        assert combo.toolTip() == container.translator.translate(
            "debug.dbg_model_from_settings"
        )


def test_default_models_prefer_31_flash_lite_and_refine_default_has_high_quota() -> None:
    """[v3.23.635] 3.1-flash-lite đứng đầu (đo thật: dịch tốt, nhanh gấp đôi 3.5-flash-lite);
    giai đoạn tinh chỉnh mặc định dùng model 500 request/ngày thay vì model Flash 20/ngày."""
    models = translate_page._DEFAULT_MODELS
    assert models[0] == "gemini-3.1-flash-lite"
    assert "gemini-3.5-flash-lite" in models
    assert translate_page.DEFAULT_REFINE_MODEL == "gemini-3.1-flash-lite"
