"""[v3.23.703] Nút bật/tắt lượt "khớp giọng đọc" (tự rút gọn dòng dài hơn khung TTS) trên trang Dịch.

Trước bản này ``TranslateSubtitlesRequest.fit_for_speech`` luôn là ``True`` (mặc định dataclass) — người dùng
không tắt được bước tốn thêm ~50 % thời gian dịch khi chỉ cần phụ đề.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication


def _container(tmp_path: Path):
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
    from subtitles_extractor.presentation.pages import editor_page

    return ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
        i18n_data_dir=Path(editor_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp_path,
    )


def test_setting_defaults_on(tmp_path: Path) -> None:
    container = _container(tmp_path)
    assert container.settings_service.current.translation.fit_for_speech is True


def test_checkbox_persists_and_reaches_request(tmp_path: Path) -> None:
    from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationContext
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage

    QApplication.instance() or QApplication([])
    container = _container(tmp_path)
    page = TranslatePage(container)
    try:
        assert page._fit_speech_check.isChecked()
        vm = page._view_model
        ctx = TranslationContext(target_lang="Vietnamese")
        assert vm._translation_request([], ctx, video_path=None, enable_visual_cues=False).fit_for_speech is True
        page._fit_speech_check.setChecked(False)
        assert container.settings_service.current.translation.fit_for_speech is False
        assert vm._translation_request([], ctx, video_path=None, enable_visual_cues=False).fit_for_speech is False
    finally:
        page.close()
    # Mở lại trang: ô giữ trạng thái đã lưu.
    page2 = TranslatePage(container)
    try:
        assert not page2._fit_speech_check.isChecked()
    finally:
        page2.close()


def test_strings_exist_in_both_languages() -> None:
    import json

    from subtitles_extractor.presentation.pages import editor_page

    data_dir = Path(editor_page.__file__).resolve().parents[2] / "data"
    for lang in ("vi", "en"):
        group = json.loads((data_dir / f"strings_{lang}.json").read_text(encoding="utf-8"))["translate"]
        for key in ("tr_fit_for_speech", "fit_for_speech", "fit_for_speech_tip"):
            assert group[key].strip(), (lang, key)
