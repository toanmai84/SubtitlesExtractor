"""[v3.23.666] Trang Dịch: chọn nhà cung cấp AI tuỳ chỉnh (API kiểu OpenAI, vd OxionAI)."""

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
def isolated_qsettings(tmp_path: Path) -> Iterator[Path]:
    """QSettings của trang ghi vào thư mục tạm (không đụng cấu hình thật của máy)."""
    QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, str(tmp_path))
    yield tmp_path
    QSettings.setPath(
        QSettings.Format.NativeFormat, QSettings.Scope.UserScope,
        QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericConfigLocation),
    )


def _container():
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import translate_page

    tmp = Path(tempfile.mkdtemp())
    return ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
        i18n_data_dir=Path(translate_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp,
    )


def _page(container):
    from subtitles_extractor.presentation.pages import translate_page

    page = translate_page.TranslatePage(container)
    warnings: list[str] = []
    page._show_warning = lambda title, _body: warnings.append(title)  # type: ignore[method-assign]
    page._test_warnings = warnings  # type: ignore[attr-defined]
    return page


def test_chon_tuy_chinh_hien_o_nhap_va_ap_xuong_bo_dich(
    app: QApplication, isolated_qsettings: Path
) -> None:
    container = _container()
    page = _page(container)
    form = page._ai_form
    assert page._provider_combo.currentData() == "gemini"
    assert form.isRowVisible(page._gemini_key_row)
    assert not form.isRowVisible(page._custom_url_edit)

    page._provider_combo.setCurrentIndex(page._provider_combo.findData("custom"))
    assert form.isRowVisible(page._custom_url_edit) and form.isRowVisible(page._custom_key_row)
    assert not form.isRowVisible(page._gemini_key_row)
    assert not page._attach_video_check.isEnabled()
    # Chưa nhập địa chỉ/key: không lặng lẽ dùng key Gemini, cảnh báo rõ.
    assert container.custom_text_provider is None
    assert page._combined_keys_text() == ""
    assert page._warn_custom_incomplete() is True
    assert page._test_warnings

    page._custom_url_edit.setText("https://oxionai.com/v1/chat/completions")
    page._custom_key_edit.setText("sk-abc")
    page._custom_key_edit.editingFinished.emit()
    provider = container.custom_text_provider
    assert provider is not None and provider.normalized_base_url == "https://oxionai.com/v1"
    assert page._combined_keys_text() == "sk-abc"
    assert page._warn_custom_incomplete() is False
    translator = container.make_subtitle_translator(page._combined_keys_text())
    assert translator.uses_custom_provider and translator._quota_manager is None

    # Mở lại ứng dụng: khôi phục lựa chọn.
    container2 = _container()
    page2 = _page(container2)
    assert page2._provider_combo.currentData() == "custom"
    assert page2._custom_url_edit.text() == "https://oxionai.com/v1/chat/completions"
    assert container2.custom_text_provider is not None

    page2._provider_combo.setCurrentIndex(page2._provider_combo.findData("gemini"))
    assert container2.custom_text_provider is None
    assert page2._attach_video_check.isEnabled()
    page.close()
    page2.close()


def test_tinh_nang_can_video_bao_ro_khi_dung_tuy_chinh(
    app: QApplication, isolated_qsettings: Path
) -> None:
    container = _container()
    page = _page(container)
    vm = page._view_model
    assert vm.set_custom_provider(True, "https://oxionai.com/v1", "sk-x") is True
    vm._source_events = [object()]  # chỉ cần "có phụ đề nguồn" để tới bước kiểm nhà cung cấp
    errors: list[str] = []
    vm.analyze_error.connect(errors.append)
    assert vm.start_speaker_completion("sk-x", "gemini-3.7-flash", "/x.mp4", "[]", "", "Vietnamese") is False
    assert vm.start_cue_recheck(api_key="sk-x", model_name="m", video_path="/x.mp4", visual_cues="[]",
                                characters="", target_lang="Vietnamese") is False
    assert len(errors) == 2 and all("Gemini" in e for e in errors)
    assert vm.set_custom_provider(False, "https://oxionai.com/v1", "sk-x") is False
    assert container.custom_text_provider is None
    page.close()


def test_tuy_chinh_bo_tick_dinh_video_va_khong_dung_ten_gemini_co_dinh(
    app: QApplication, isolated_qsettings: Path
) -> None:
    container = _container()
    page = _page(container)
    page._attach_video_check.setChecked(True)
    assert page._visual_cues_check.isChecked()
    page._custom_url_edit.setText("https://oxionai.com/v1")
    page._custom_key_edit.setText("sk-abc")
    page._provider_combo.setCurrentIndex(page._provider_combo.findData("custom"))
    # Không còn ô "tick mà xám": bỏ tick kéo theo các ô phụ thuộc video.
    assert not page._attach_video_check.isChecked() and not page._attach_video_check.isEnabled()
    assert not page._visual_cues_check.isChecked()
    assert page._selected_video_path() is None

    page._analyze_model_combo.setCurrentText("gemini-3.1-pro")
    assert page.shorten_model() == "gemini-3.1-pro"
    vm = page._view_model
    vm.set_fallback_model("")
    vm.set_audit_model("")
    assert vm.analysis_fallback_model() == ""
    assert vm.audit_chain("gemini-3.1-pro") == ("gemini-3.1-pro", ())

    page._provider_combo.setCurrentIndex(page._provider_combo.findData("gemini"))
    assert page.shorten_model() == "gemini-3.1-flash-lite"
    assert vm.analysis_fallback_model() == "gemini-3.1-flash-lite"
    page.close()
