"""[v3.23.677] Khoá API + "Số lần thử lại" về Cài đặt (dùng chung Dịch / Gemini TTS), gọn nút trang Lồng tiếng.

Trước bản này:
* Khoá Gemini nằm ở trang Dịch, Gemini TTS có ô "API Key" RIÊNG (lưu chỗ khác) dù chú thích ghi "dùng chung".
* "Số lần thử lại" có ở CẢ trang Dịch (1–20, mặc định 5) lẫn Cài đặt (1–10, mặc định 3); ô trang Dịch luôn đè Cài đặt.
* Bảng dòng trang Lồng tiếng: 22 nút xếp 6 hàng; chú thích Edge dồn thành một đoạn kỹ thuật ("Elastic timing",
  "pip install pedalboard"…).
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, QStandardPaths
from PySide6.QtWidgets import QApplication, QPushButton


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def isolated(tmp_path: Path) -> Iterator[Path]:
    QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, str(tmp_path))
    yield tmp_path
    QSettings.setPath(
        QSettings.Format.NativeFormat, QSettings.Scope.UserScope,
        QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericConfigLocation),
    )


def _container():
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
    from subtitles_extractor.presentation.pages import translate_page

    tmp = Path(tempfile.mkdtemp())
    return ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
        i18n_data_dir=Path(translate_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp,
    )


# ── Kho khoá dùng chung ────────────────────────────────────────────────────────────────────


def test_kho_doc_dung_khoa_cu_cua_trang_dich(app: QApplication, isolated: Path) -> None:
    from subtitles_extractor.presentation.ai_access_store import AiAccessStore

    legacy = QSettings("SubtitlesExtractor", "TranslatePage")   # bản cũ lưu ở đây
    legacy.setValue("api_key", "AIzaA")
    legacy.setValue("api_keys_extra", "AIzaB\nAIzaA\n\nAIzaC")
    legacy.sync()
    store = AiAccessStore()
    assert store.gemini_keys() == ["AIzaA", "AIzaB", "AIzaC"]
    seen: list[int] = []
    store.changed.connect(lambda: seen.append(1))
    store.set_gemini_keys(["X", "X", " Y "])
    assert AiAccessStore().gemini_keys() == ["X", "Y"] and seen == [1]
    store.set_provider("khác")
    assert store.provider() == "gemini"
    store.set_custom("https://oxionai.com/v1", "sk")
    assert store.custom_complete()


def test_chuyen_cau_hinh_cu_mot_lan(app: QApplication, isolated: Path) -> None:
    from subtitles_extractor.presentation.ai_access_store import AiAccessStore, migrate_legacy_ai_settings

    store = AiAccessStore()
    store.set_gemini_keys(["AIzaA"])
    tts = QSettings("SubtitlesExtractorPaddleOCR", "TTSPage_v2")
    tts.setValue("gemini_key", "AIzaT1, AIzaA")
    page = QSettings("SubtitlesExtractor", "TranslatePage")
    page.setValue("retry_count", 15)
    tts.sync()
    page.sync()
    patches: list[dict] = []
    service = SimpleNamespace(update=lambda **p: patches.append(p))
    notes = migrate_legacy_ai_settings(store, service)
    assert len(notes) == 2
    assert store.gemini_keys() == ["AIzaA", "AIzaT1"]
    assert patches == [{"translation": {"retry_count": 10}}]     # 1–20 cũ → 1–10 của Cài đặt
    assert not QSettings("SubtitlesExtractorPaddleOCR", "TTSPage_v2").contains("gemini_key")
    assert not QSettings("SubtitlesExtractor", "TranslatePage").contains("retry_count")
    assert migrate_legacy_ai_settings(store, service) == []      # lần sau: không làm lại


def test_cau_hinh_hong_khong_chan_khoi_dong(app: QApplication, isolated: Path) -> None:
    from subtitles_extractor.domain.exceptions import ConfigurationError
    from subtitles_extractor.presentation.ai_access_store import AiAccessStore, migrate_legacy_ai_settings

    page = QSettings("SubtitlesExtractor", "TranslatePage")
    page.setValue("retry_count", 4)
    page.sync()

    def broken(**_p):
        raise ConfigurationError("hỏng")

    assert migrate_legacy_ai_settings(AiAccessStore(), SimpleNamespace(update=broken)) == []
    assert QSettings("SubtitlesExtractor", "TranslatePage").contains("retry_count")   # để lần sau thử lại


# ── Trang Dịch / Lồng tiếng / Cài đặt ──────────────────────────────────────────────────────


def test_trang_dich_khong_con_o_key_va_thu_lai(app: QApplication, isolated: Path) -> None:
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage

    page = TranslatePage(_container())
    assert not hasattr(page, "_api_key_edit") and not hasattr(page, "_retry_spin")
    opened: list[int] = []
    page.set_open_settings_callback(lambda: opened.append(1))
    page._show_warning = lambda *_a: None  # type: ignore[method-assign]
    calls: list[dict] = []
    page._view_model.start_translation = lambda **k: calls.append(k) or True  # type: ignore[method-assign]
    page._stage_literal.enable_check.setChecked(True)
    page._ai_store.set_gemini_keys([])
    page._on_run_clicked()
    assert calls == [] and opened == [1]                         # thiếu key → mở Cài đặt
    assert "⚠️" in page._ai_summary.label.text()
    page._ai_store.set_gemini_keys(["AIzaKEY1", "AIzaKEY2"])
    assert "2 key Gemini" in page._ai_summary.label.text()      # tóm tắt tự cập nhật
    page._on_run_clicked()
    assert calls and calls[0]["api_key"] == "AIzaKEY1\nAIzaKEY2" and calls[0]["retry_count"] is None
    page.close()


def test_retry_none_lay_tu_cai_dat(app: QApplication, isolated: Path) -> None:
    container = _container()
    container.settings_service.update(translation={"retry_count": 7})
    translator = container.make_subtitle_translator("AIzaKEY", None)
    assert translator._retry_count == 7


def test_gemini_tts_dung_key_chung(app: QApplication, isolated: Path) -> None:
    from subtitles_extractor.presentation.pages.tts_page import TTSPage

    page = TTSPage(_container())
    page._ai_store.set_gemini_keys(["AIzaA", "AIzaB"])
    assert page._gemini_api_keys() == "AIzaA\nAIzaB"
    assert "2 key Gemini" in page._gemini_key_summary.label.text()
    page.close()


def test_cai_dat_co_muc_khoa_api_va_luu_ngay(app: QApplication, isolated: Path) -> None:
    from subtitles_extractor.presentation.ai_access_store import shared_ai_access_store
    from subtitles_extractor.presentation.pages.settings_page import SettingsPage

    page = SettingsPage(_container())
    titles = [page._tabs.tabText(i) for i in range(page._tabs.count())]
    assert "Khoá API và AI" in titles
    page.show_ai_access()
    assert page._tabs.tabText(page._tabs.currentIndex()) == "Khoá API và AI"
    panel = page.ai_access_panel
    panel.gemini_key_edit.setText("AIzaNEW")
    panel.gemini_key_edit.editingFinished.emit()
    assert shared_ai_access_store().gemini_keys() == ["AIzaNEW"]
    panel.provider_combo.setCurrentIndex(panel.provider_combo.findData("custom"))
    assert shared_ai_access_store().is_custom()
    assert panel.form.isRowVisible(panel.custom_url_edit)
    assert panel.form.isRowVisible(panel.gemini_key_edit.parentWidget())   # key Gemini vẫn cần cho Gemini TTS
    for button in (panel.gemini_eye, panel.custom_eye, panel.multi_key_button):
        assert button.accessibleName()
    page.close()


# ── Nút trang Lồng tiếng ──────────────────────────────────────────────────────────────────


def test_menu_phan_chieu_trang_thai_nut(app: QApplication) -> None:
    from subtitles_extractor.presentation.widgets.menu_button import MenuButton

    clicked: list[str] = []
    a, b = QPushButton("Gộp dòng"), QPushButton("Tách dòng (2)")
    a.clicked.connect(lambda: clicked.append("a"))
    b.setEnabled(False)
    menu = MenuButton("Sửa dòng", [a, b])
    menu.show()
    assert not a.isVisible() and not b.isVisible()
    menu.menu().aboutToShow.emit()
    actions = menu.menu().actions()
    assert [x.text() for x in actions] == ["Gộp dòng", "Tách dòng (2)"]
    assert [x.isEnabled() for x in actions] == [True, False]
    actions[0].trigger()
    assert clicked == ["a"]


def test_trang_tts_it_nut_hon_va_chu_thich_de_hieu(app: QApplication, isolated: Path) -> None:
    from PySide6.QtWidgets import QLabel, QToolButton

    from subtitles_extractor.presentation.pages.tts_page import TTSPage

    page = TTSPage(_container())
    page.resize(1366, 768)
    page.show()
    app.processEvents()
    table_pane = page._table.parentWidget()
    visible = [w for w in table_pane.findChildren(QPushButton) if w.isVisible()]
    menus = [w for w in table_pane.findChildren(QToolButton) if w.isVisible() and w.text().endswith("▾")]
    assert len(visible) <= 9 and len(menus) == 3, [w.text() for w in visible]
    # Mọi nút cũ vẫn tồn tại (logic bật/tắt giữ nguyên), chỉ chuyển vào menu.
    hidden = {b.text() for m in menus for b in m.buttons}
    assert len(hidden) == 16                                   # [v3.23.679] + "Kiểm tra phát âm (AI)"
    texts = " ".join(lb.text() for lb in page.findChildren(QLabel) if lb.isVisible())
    assert "pip install" not in texts and "Elastic timing" not in texts
    page.close()
