"""[v3.23.623] Ô "Gán người nói từ phụ đề" nối tới yêu cầu dịch / phân tích."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationStageConfig,
    TranslationStageKind,
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _page():
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import translate_page

    tmp = Path(tempfile.mkdtemp())
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
        i18n_data_dir=Path(translate_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp,
    )
    return translate_page.TranslatePage(container), tmp


def _stage() -> TranslationStageConfig:
    return TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="gemini-3.1-flash-lite")


def test_default_on_and_request_carries_flags(app: QApplication) -> None:
    page, tmp = _page()
    assert page._text_cues_check.isChecked()
    video = tmp / "ep.mp4"
    video.write_bytes(b"x")
    page._source_video_path = str(video)
    page._sync_text_cues()
    request = page._view_model._translation_request(
        [_stage()], TranslationContext(target_lang="Vietnamese"),
        video_path=None, enable_visual_cues=False)
    assert request.text_cues and request.voice_source_path == str(video)


def test_off_clears_voice_source_and_missing_video_is_ignored(app: QApplication) -> None:
    page, tmp = _page()
    page._source_video_path = str(tmp / "missing.mp4")
    page._sync_text_cues()
    request = page._view_model._translation_request(
        [_stage()], TranslationContext(target_lang="Vietnamese"),
        video_path=None, enable_visual_cues=False)
    assert request.text_cues and request.voice_source_path == ""
    page._text_cues_check.setChecked(False)
    page._sync_text_cues()
    request = page._view_model._translation_request(
        [_stage()], TranslationContext(target_lang="Vietnamese"),
        video_path=None, enable_visual_cues=False)
    assert not request.text_cues


def test_strings_exist_in_both_languages() -> None:
    import json

    data_dir = Path(__file__).resolve().parents[3] / "src/subtitles_extractor/data"
    for lang in ("vi", "en"):
        table = json.loads((data_dir / f"strings_{lang}.json").read_text(encoding="utf-8"))
        flat = {k: v for section in table.values() if isinstance(section, dict) for k, v in section.items()}
        assert flat.get("opt_text_cues") and flat.get("opt_text_cues_tip")
