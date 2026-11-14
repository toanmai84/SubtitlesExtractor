"""[v3.23.704] Tiền xử lý OCR: chỉ GỢI Ý cài đặt tốt nhất, người dùng toàn quyền.

* Mở trang Cài đặt KHÔNG tự đổi cấu hình đã lưu (kể cả khi 4 bước đang bật).
* Nút "Dùng cài đặt gợi ý" tắt 4 bước nhưng CHƯA lưu — phải bấm Lưu; "Huỷ thay đổi" trả lại như cũ.
* Nút chỉ bấm được khi còn bước nào đang bật.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

_KEYS = ("add_white_border", "apply_sharpen", "apply_contrast_boost")   # [v3.23.721] phóng to không còn bị gợi ý tắt


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _container(tmp_path: Path):
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
    from subtitles_extractor.presentation.pages import translate_page

    return ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
        i18n_data_dir=Path(translate_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp_path,
    )


def _preprocess(container) -> dict[str, bool]:
    current = container.settings_service.current.preprocess
    return {key: bool(getattr(current, key)) for key in _KEYS}


def test_button_disabled_when_already_recommended(app: QApplication, tmp_path: Path) -> None:
    from subtitles_extractor.presentation.pages.settings_page import SettingsPage

    page = SettingsPage(_container(tmp_path))
    try:
        assert not page._btn_preprocess_recommended.isEnabled()
    finally:
        page.close()


def test_suggestion_never_changes_saved_settings_by_itself(app: QApplication, tmp_path: Path) -> None:
    from subtitles_extractor.presentation.pages.settings_page import SettingsPage

    container = _container(tmp_path)
    container.settings_service.update(preprocess={key: True for key in _KEYS})
    page = SettingsPage(container)
    try:
        # Mở trang: không đổi gì, nút gợi ý bấm được.
        assert all(_preprocess(container).values())
        assert page._btn_preprocess_recommended.isEnabled()
        page._btn_preprocess_recommended.click()
        assert not any(check.isChecked() for check in page._recommended_off_preprocess_checks())
        assert not page._btn_preprocess_recommended.isEnabled()
        # Chưa bấm Lưu → cấu hình đã lưu vẫn như người dùng đặt.
        assert all(_preprocess(container).values())
        assert page._save_button.isEnabled()
        # Người dùng vẫn bật lại được từng bước.
        page._chk_sharpen.setChecked(True)
        assert page._btn_preprocess_recommended.isEnabled()
    finally:
        page.close()


def test_strings_exist_in_both_languages() -> None:
    import json

    from subtitles_extractor.presentation.pages import translate_page

    data_dir = Path(translate_page.__file__).resolve().parents[2] / "data"
    for lang in ("vi", "en"):
        group = json.loads((data_dir / f"strings_{lang}.json").read_text(encoding="utf-8"))["settings"]
        for key in ("btn_preprocess_recommended", "tip_preprocess_recommended", "tip_preprocess_off_recommended"):
            assert group[key].strip(), (lang, key)
