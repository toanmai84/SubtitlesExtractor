"""[v3.23.721] Nút "Dùng cài đặt gợi ý" không còn tắt "Phóng to".

Gợi ý 3.23.704 dựa trên số đo 3.23.713, lúc phóng to còn vô hiệu (vùng chữ 136 px với đích ≤ 272 px không bao giờ được
phóng). Phóng thật ×2 là bước tốt nhất đo được (4064 khung: đúng 98,28 % → 99,53 %), nên nút không được tắt nó nữa.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from tests.unit.presentation.test_preprocess_recommendation_v323704 import _container


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_recommendation_keeps_upscale(app: QApplication, tmp_path: Path) -> None:
    from subtitles_extractor.presentation.pages.settings_page import SettingsPage

    container = _container(tmp_path)
    container.settings_service.update(preprocess={"upscale_small_text": True, "apply_sharpen": True})
    page = SettingsPage(container)
    try:
        assert page._btn_preprocess_recommended.isEnabled()
        page._btn_preprocess_recommended.click()
        assert page._chk_upscale.isChecked()                 # mã cũ: bị bỏ chọn
        assert not page._chk_sharpen.isChecked()
        assert page._chk_upscale.toolTip().strip()
    finally:
        page.close()


def test_only_upscale_on_needs_no_recommendation(app: QApplication, tmp_path: Path) -> None:
    from subtitles_extractor.presentation.pages.settings_page import SettingsPage

    container = _container(tmp_path)
    container.settings_service.update(preprocess={"upscale_small_text": True})
    page = SettingsPage(container)
    try:
        assert not page._btn_preprocess_recommended.isEnabled()
    finally:
        page.close()


def test_default_target_is_twice_the_measured_roi() -> None:
    from subtitles_extractor.infrastructure.settings.application_settings import ApplicationSettings

    assert ApplicationSettings().preprocess.upscale_target_height_px == 272
