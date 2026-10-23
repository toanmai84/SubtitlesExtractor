"""[v3.23.612] Ô "Xưng hô thuần Việt" trên trang Dịch.

Bật ô là đổi NGAY bảng nhân vật và bản dịch đang có (không phải phân tích/dịch lại),
lưu vào phiên; mở lại phiên khi ô đang bật thì bản dịch đã lưu cũng được đổi.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from tests.unit.application.test_native_address_v323612 import ROSTER

SOURCES = ["劍哥", "我還進去看看劍哥", "洪大嫂我不想吃"]
TRANSLATED = [
    "Kiếm ca.",
    "Tôi vào xem Kiếm ca thế nào đã.",
    "Hồng đại tẩu, tôi không muốn ăn.",
]
NATIVE = [
    "Anh Kiếm.",
    "Tôi vào xem anh Kiếm thế nào đã.",
    "Chị Hồng, tôi không muốn ăn.",
]
CUES = json.dumps([
    {"id": 1, "spk": "秦可秀", "to": "甄劍聲"},
    {"id": 2, "spk": "秦可秀", "to": "朱正軍"},
    {"id": 3, "spk": "秦可秀", "to": "洪大嫂"},
], ensure_ascii=False)


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _page(tmp: Path | None = None):
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import translate_page

    tmp = tmp or Path(tempfile.mkdtemp())
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
        i18n_data_dir=Path(translate_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp,
    )
    page = translate_page.TranslatePage(container)
    page._persist_inputs = lambda: None  # không ghi QSettings thật của máy
    page._lang_combo.setCurrentText("Vietnamese")
    return page, container


def _events(texts: list[str]) -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=i, text=text, interval=TimeInterval(i, i + 0.9))
        for i, text in enumerate(texts, start=1)
    ]


def _load(page, tmp_path: Path) -> str:
    view_model = page._view_model
    video = str(tmp_path / "v.mp4")
    view_model.set_source_events(_events(SOURCES))
    view_model._last_translate_video_path = video
    view_model._translated_events = [
        replace(event, text=text)
        for event, text in zip(view_model.source_events, TRANSLATED, strict=True)
    ]
    page._characters_edit.setPlainText(ROSTER)
    page._visual_cues_edit.setPlainText(CUES)
    return video


class TestCheckbox:
    def test_off_by_default_and_saved(self, app: QApplication, tmp_path: Path) -> None:
        page, container = _page(tmp_path)
        assert page._native_address_check.isChecked() is False
        page._native_address_check.setChecked(True)
        assert container.settings_service.current.translation.native_address is True
        page._native_address_check.setChecked(False)
        assert container.settings_service.current.translation.native_address is False
        page.close()

    def test_checking_converts_roster_and_translation_now(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        page, _container = _page(tmp_path)
        _load(page, tmp_path)
        page._native_address_check.setChecked(True)
        assert [e.text for e in page._view_model.translated_events] == NATIVE
        roster = page._characters_edit.toPlainText()
        assert "(Chân Kiếm Thanh / anh Kiếm)" in roster
        assert "Kiếm ca" not in roster
        page.close()

    def test_unchecking_keeps_current_translation(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        page, _container = _page(tmp_path)
        _load(page, tmp_path)
        page._native_address_check.setChecked(True)
        page._native_address_check.setChecked(False)
        assert [e.text for e in page._view_model.translated_events] == NATIVE
        page.close()

    def test_converted_translation_is_saved_to_session(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        page, _container = _page(tmp_path)
        video = _load(page, tmp_path)
        page._native_address_check.setChecked(True)
        view_model = page._view_model
        view_model._translated_events = []
        view_model.restore_stage_outputs_for_video(video)
        assert [e.text for e in view_model.translated_events] == NATIVE
        page.close()

    def test_not_applied_while_busy(self, app: QApplication, tmp_path: Path) -> None:
        page, _container = _page(tmp_path)
        _load(page, tmp_path)
        page._view_model._is_busy = True
        page._native_address_check.setChecked(True)
        page._view_model._is_busy = False
        assert [e.text for e in page._view_model.translated_events] == TRANSLATED
        page.close()


class TestViewModel:
    def test_effective_context_follows_the_option(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        """Vân tay bộ nhớ đệm phải tính trên đúng bảng nhân vật sẽ gửi đi khi dịch."""
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            TranslationContext,
        )

        page, _container = _page(tmp_path)
        view_model = page._view_model
        context = TranslationContext(target_lang="Vietnamese", characters=ROSTER)
        assert "Kiếm ca" in view_model._effective_context(context).characters
        view_model.set_native_address(True)
        assert "Kiếm ca" not in view_model._effective_context(context).characters
        page.close()

    def test_restoring_a_session_converts_saved_translation(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        from subtitles_extractor.application.services.translation_session_hashing import (
            hash_analysis_input,
        )

        page, container = _page(tmp_path)
        view_model = page._view_model
        video = _load(page, tmp_path)
        view_model._persist_current_result()
        container.translation_session_store.save_analysis(
            video, characters=ROSTER, overview="", source_lang="zh",
            target_lang="Vietnamese",
            input_hash=hash_analysis_input(SOURCES, "Vietnamese"),
            glossary="", visual_cues=CUES,
        )
        view_model.set_native_address(True)
        view_model._translated_events = []
        assert view_model.try_restore_saved_analysis(video, "Vietnamese")
        assert [e.text for e in view_model.translated_events] == NATIVE
        page.close()

    def test_diagnostics_record_the_option(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from subtitles_extractor.presentation.utils import safe_dialogs

        page, _container = _page(tmp_path)
        page._native_address_check.setChecked(True)
        page._characters_edit.setPlainText(ROSTER)
        target = tmp_path / "context.json"
        monkeypatch.setattr(safe_dialogs, "save_file", lambda *_a, **_k: str(target))
        monkeypatch.setattr(page, "_show_info", lambda *_a: None)
        page._on_export_context_diagnostics()
        saved = json.loads(target.read_text(encoding="utf-8"))
        assert saved["analysis_settings"]["native_address"] is True
        page.close()
