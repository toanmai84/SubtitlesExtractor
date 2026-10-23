"""[v3.23.601] Trang Dịch: không nút nào bị cắt chữ, thẻ Ngữ cảnh không cuộn ngang.

Ảnh chụp offscreen v3.23.600 ở cửa sổ 1500 px: hàng nút kết quả (8 nút) bị bóp còn ~80 px
mỗi nút ("…ửi sang Biên", "…o sánh giai đ"), ở 1920 px vẫn cắt 4 nút; hàng nút phân tích
ngữ cảnh rộng hơn cột trái nên cả thẻ phải cuộn ngang và nút cuối bị che. Bản này thêm nút
"Bổ sung người nói" vào hàng phân tích — phải tách hàng thay vì làm tệ thêm.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (
    QApplication,
    QPushButton,
    QScrollArea,
    QTabWidget,
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _page(width: int):
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import translate_page
    from subtitles_extractor.presentation.widgets.section_card import SectionCard

    tmp = Path(tempfile.mkdtemp())
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
        i18n_data_dir=Path(translate_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp,
    )
    page = translate_page.TranslatePage(container)
    page.resize(width, 1000)
    page.show()
    for card in page.findChildren(SectionCard):
        card.set_collapsed(False)
    for tabs in page.findChildren(QTabWidget):
        for index in range(tabs.count()):
            if "cảnh" in tabs.tabText(index):
                tabs.setCurrentIndex(index)
    QApplication.processEvents()
    page.grab()  # buộc bố cục chốt kích thước như khi hiện lên màn hình thật
    QApplication.processEvents()
    return page


@pytest.mark.parametrize("width", [1500, 1920])
def test_khong_nut_nao_bi_cat_chu(app: QApplication, width: int) -> None:
    page = _page(width)
    cut = [
        button.text()
        for button in page.findChildren(QPushButton)
        if button.isVisible() and button.width() + 2 < button.sizeHint().width()
    ]
    assert cut == []
    page.close()


def test_the_ngu_canh_khong_cuon_ngang(app: QApplication) -> None:
    page = _page(1500)
    assert page._btn_complete_speakers.isVisible()
    scrolls = [
        area for area in page.findChildren(QScrollArea)
        if area.isVisible() and area.isAncestorOf(page._btn_complete_speakers)
    ]
    assert scrolls
    assert all(not area.horizontalScrollBar().isVisible() for area in scrolls)
    page.close()


def test_trang_tts_khong_nut_nao_bi_cat_chu(
    app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """v3.23.600 ở 1500 px: bốn nút bị cắt chữ (sáu nút trên một hàng).

    "▶ Nghe dòng chọn", "⚠ Chọn dòng bất thường", "📊 Xuất chi tiết (debug)", "🔄 TTS lại
    dòng chọn".
    """
    from PySide6.QtCore import QSettings

    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import tts_page
    from subtitles_extractor.presentation.widgets.section_card import SectionCard

    monkeypatch.setattr(
        tts_page,
        "QSettings",
        lambda *_args: QSettings(str(tmp_path / "p.ini"), QSettings.Format.IniFormat),
    )
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
        i18n_data_dir=Path(tts_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp_path,
    )
    page = tts_page.TTSPage(container)
    page.resize(1500, 1000)
    page.show()
    for card in page.findChildren(SectionCard):
        card.set_collapsed(False)
    QApplication.processEvents()
    page.grab()
    QApplication.processEvents()
    cut = [
        button.text()
        for button in page.findChildren(QPushButton)
        if button.isVisible() and button.width() + 2 < button.sizeHint().width()
    ]
    assert cut == []
    page.close()
