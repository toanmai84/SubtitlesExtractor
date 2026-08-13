"""Test [v3.18] các fix UX/logic của trang Trích xuất và Dịch.

Yêu cầu PyQt6 + qfluentwidgets + container GUI; tự bỏ qua nếu môi trường thiếu.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("qfluentwidgets")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def container(qapp):
    try:
        from subtitles_extractor.composition.bootstrap import bootstrap_for_gui

        return bootstrap_for_gui()
    except Exception as exc:  # pragma: no cover - thiếu dependency tuỳ chọn
        pytest.skip(f"Không dựng được container GUI: {exc}")


@pytest.fixture
def extract_page(container):
    try:
        from subtitles_extractor.presentation.pages.extract_page import ExtractPage

        return ExtractPage(container)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Không dựng được ExtractPage: {exc}")


@pytest.fixture
def translate_page(container):
    try:
        from subtitles_extractor.presentation.pages.translate_page import TranslatePage

        return TranslatePage(container)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Không dựng được TranslatePage: {exc}")


class TestExtractPageUxFixes:
    def test_slider_resolution_is_10000(self) -> None:
        from subtitles_extractor.presentation.pages import extract_page as ep

        assert ep._SLIDER_RESOLUTION == 10000  # [#10]

    def test_draw_button_enabled_by_default(self, extract_page) -> None:
        assert extract_page._draw_toggle.isEnabled() is True  # [#6]

    def test_default_preset_is_auto_subtitle(self, extract_page) -> None:
        # [#4] Mục đầu của ComboBox preset là "Tự nhận diện".
        assert extract_page._preset_combo.itemData(0) == "auto_subtitle"

    def test_sync_preset_combo_does_not_trigger_detection(self, extract_page) -> None:
        # [#5] Đồng bộ ComboBox không kích hoạt detection (chặn tín hiệu).
        extract_page._sync_preset_combo_to("full")
        assert extract_page._preset_combo.currentData() == "full"

    def test_draw_toggle_switches_combo_to_custom(self, extract_page) -> None:
        # [#6] Bật vẽ → ComboBox tự gạt sang "Tuỳ chỉnh".
        extract_page._on_draw_toggle_toggled(True)
        assert extract_page._preset_combo.currentData() == "custom"

    def test_cancel_flag_yields_warning_not_completion(self, extract_page) -> None:
        # [#7] Khi đã huỷ, hoàn tất hiển thị cảnh báo vàng, không báo thành công.
        extract_page._is_cancelled_by_user = True

        class _Resp:
            events: list = []
            elapsed_seconds = 1.0

        extract_page._on_extraction_finished(_Resp())
        assert "hủy" in extract_page._status_label.text().lower()
        assert extract_page._is_cancelled_by_user is False  # cờ được reset

    def test_speed_handler_runs_without_error(self, extract_page) -> None:
        # [#9] Đổi tốc độ không gây lỗi (canvas có set_playback_speed).
        extract_page._speed_combo.setCurrentIndex(3)
        extract_page._on_speed_changed(3)
        assert extract_page._speed_combo.currentData() == 2.0


class TestTranslatePageContextFixes:
    def test_reset_context_clears_everything(self, translate_page) -> None:
        # [#2] Tẩy não toàn bộ ngữ cảnh phim cũ.
        translate_page._characters_edit.setPlainText("A, B")
        translate_page._overview_edit.setPlainText("cốt truyện")
        translate_page._source_lang_edit.setText("zh")
        translate_page._video_path = "/old/A.mp4"
        translate_page._attach_video_check.setChecked(True)

        translate_page._reset_context_and_video()

        assert translate_page._characters_edit.toPlainText() == ""
        assert translate_page._overview_edit.toPlainText() == ""
        assert translate_page._source_lang_edit.text() == ""
        assert translate_page._video_path == ""
        assert translate_page._attach_video_check.isChecked() is False

    def test_pull_from_editor_auto_attaches_video(self, translate_page) -> None:
        # [#11] Lấy dữ liệu từ Biên tập → tự đính video + reset ngữ cảnh cũ.
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        events = [SubtitleEvent(index=1, text="hi", interval=TimeInterval(start_sec=0.0, end_sec=1.0))]
        translate_page.set_editor_event_provider(lambda: events)
        translate_page.set_editor_video_provider(lambda: "/phim/B.mkv")
        translate_page._show_info = lambda *a, **k: None
        translate_page._source_lang_edit.setText("ja")  # ngữ cảnh cũ

        translate_page._on_pull_editor_clicked()

        assert translate_page._video_path == "/phim/B.mkv"
        assert translate_page._video_path_label.text() == "B.mkv"
        assert translate_page._attach_video_check.isChecked() is True
        assert translate_page._source_lang_edit.text() == ""  # ngữ cảnh cũ đã reset
