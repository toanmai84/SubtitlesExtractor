"""[v3.23.700] Cài đặt AI / trang Dịch: ô key chính không làm mất key, đổi nhà cung cấp không lẫn model / tick video."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, QStandardPaths
from PySide6.QtWidgets import QApplication


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


class _Tr:
    def translate(self, key: str, **_kw: object) -> str:
        return key


def _panel(keys: list[str]):
    from subtitles_extractor.presentation.ai_access_store import AiAccessStore
    from subtitles_extractor.presentation.widgets.ai_access_panel import AiAccessPanel

    store = AiAccessStore()
    store.set_gemini_keys(keys)
    panel = AiAccessPanel(_Tr(), store=store)
    return store, panel


class TestMainKeyField:
    def test_clear_enter_then_focus_out_removes_only_the_shown_key(self, app, isolated) -> None:
        store, panel = _panel(["k1", "k2", "k3"])
        focused = {"on": True}
        panel.gemini_key_edit.hasFocus = lambda: focused["on"]       # ô đang có tiêu điểm: load() không ghi đè ô
        panel.gemini_key_edit.setText("")
        panel._on_main_key_edited()                                   # Enter
        assert store.gemini_keys() == ["k2", "k3"]
        panel._on_main_key_edited()                                   # bấm ra ngoài: editingFinished lần hai
        assert store.gemini_keys() == ["k2", "k3"]                    # bản cũ: ["k3"] — mất key 2

    def test_replacing_the_shown_key_keeps_the_others(self, app, isolated) -> None:
        store, panel = _panel(["k1", "k2"])
        panel.gemini_key_edit.hasFocus = lambda: True
        panel.gemini_key_edit.setText("k9")
        panel._on_main_key_edited()
        panel._on_main_key_edited()
        assert store.gemini_keys() == ["k9", "k2"]

    def test_unchanged_text_is_a_no_op_and_paste_list_splits(self, app, isolated) -> None:
        store, panel = _panel(["k1", "k2"])
        panel._on_main_key_edited()
        assert store.gemini_keys() == ["k1", "k2"]
        panel.gemini_key_edit.setText("a1, a2")
        panel._on_main_key_edited()
        assert store.gemini_keys() == ["a1", "a2", "k2"]


def _page():
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
    from subtitles_extractor.presentation.pages import translate_page

    tmp = Path(tempfile.mkdtemp())
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
        i18n_data_dir=Path(translate_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp,
    )
    from subtitles_extractor.presentation import ai_access_store

    # Kho mới cho trang này: trang Dịch của các test khác (có ``_settings`` giả) vẫn nối vào kho dùng chung cũ.
    ai_access_store._shared = None
    page = translate_page.TranslatePage(container)
    page._show_warning = lambda *_a: None  # type: ignore[method-assign]
    return page


class TestTranslatePageProviderSwitch:
    def test_each_provider_keeps_its_own_models(self, app, isolated) -> None:
        page = _page()
        store = page._ai_store
        gemini_literal = page._stage_literal.model_combo.currentText()
        gemini_analyze = page._analyze_model_combo.currentText()
        assert gemini_literal.startswith("gemini")

        store.set_custom("https://example.com/v1", "sk-abc")
        store.set_provider("custom")
        page._stage_literal.model_combo.setEditText("gpt-4o-mini")
        page._analyze_model_combo.setEditText("gpt-4o")

        store.set_provider("gemini")          # bản cũ: vẫn gửi "gpt-4o-mini" cho Gemini
        assert page._stage_literal.model_combo.currentText() == gemini_literal
        assert page._analyze_model_combo.currentText() == gemini_analyze

        store.set_provider("custom")          # quay lại: model tuỳ chỉnh đã chọn trở lại
        assert page._stage_literal.model_combo.currentText() == "gpt-4o-mini"
        assert page._analyze_model_combo.currentText() == "gpt-4o"

    def test_key_edits_do_not_touch_models(self, app, isolated) -> None:
        page = _page()
        page._stage_literal.model_combo.setEditText("gemini-x-test")
        page._ai_store.set_gemini_keys(["k1", "k2"])           # store.changed nhưng không đổi nhà cung cấp
        assert page._stage_literal.model_combo.currentText() == "gemini-x-test"

    def test_attach_video_tick_returns_with_gemini(self, app, isolated) -> None:
        page = _page()
        store = page._ai_store
        page._video_path = "/tmp/phim.mp4"
        page._attach_video_check.setChecked(True)
        store.set_custom("https://example.com/v1", "sk-abc")
        store.set_provider("custom")
        assert not page._attach_video_check.isChecked() and not page._attach_video_check.isEnabled()
        store.set_provider("gemini")
        assert page._attach_video_check.isEnabled() and page._attach_video_check.isChecked()

    def test_new_film_does_not_inherit_the_old_tick(self, app, isolated) -> None:
        page = _page()
        store = page._ai_store
        page._video_path = "/tmp/phim.mp4"
        page._attach_video_check.setChecked(True)
        store.set_custom("https://example.com/v1", "sk-abc")
        store.set_provider("custom")
        page._reset_context_and_video()                        # nạp phim mới (video bị gỡ)
        store.set_provider("gemini")
        assert not page._attach_video_check.isChecked()
