"""[v3.23.656] Khi đang trích xuất (kể cả giữa hai tập của lượt hàng loạt): khoá Re-OCR và chạy thử OCR ở trang Gỡ lỗi."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.composition.bootstrap import bootstrap_for_gui


@pytest.fixture(scope="module")
def container():
    QApplication.instance() or QApplication([])
    return bootstrap_for_gui()


def test_editor_reocr_buttons_follow_extraction(container, monkeypatch) -> None:
    from subtitles_extractor.presentation.pages import editor_page as module

    page = module.EditorPage(container)
    shown: list = []
    monkeypatch.setattr(module.InfoBar, "warning", lambda *a, **k: shown.append(a))
    page.set_extraction_running(True)
    assert not page._btn_reocr.isEnabled() and not page._btn_fast_reocr.isEnabled()
    assert "trích xuất" in page._btn_reocr.toolTip().lower()
    page._on_reocr_busy(False)                       # Re-OCR xong cũng không được mở khoá khi đang trích xuất
    assert not page._btn_reocr.isEnabled()
    page._on_reocr_clicked()                         # gọi thẳng (phím tắt) vẫn bị chặn
    page._on_fast_reocr_clicked()
    assert len(shown) == 2
    page.set_extraction_running(False)
    assert page._btn_reocr.isEnabled() and page._btn_fast_reocr.isEnabled()


def test_extract_page_running_covers_batch_gap(container) -> None:
    from subtitles_extractor.presentation.pages.extract_page import ExtractPage

    page = ExtractPage(container)
    seen: list[bool] = []
    page.extraction_running_changed.connect(seen.append)
    page._batch_queue = ["tap01", "tap02"]          # đang ở giữa hai tập (worker chưa chạy)
    page._emit_extraction_running()
    assert seen == [True] and page.is_extraction_running()
    page._batch_queue = []
    page._emit_extraction_running()
    assert seen == [True, False]


def test_debug_page_reocr_locked(container) -> None:
    from subtitles_extractor.presentation.pages.debug_page import DebugPage

    page = DebugPage(container)
    page._set_reocr_frame_enabled(True)
    page.set_extraction_running(True)
    assert not page._btn_reocr_frame.isEnabled()
    page._set_reocr_frame_enabled(True)               # trang muốn bật lại (chọn khung khác): vẫn khoá
    assert not page._btn_reocr_frame.isEnabled()
    page.set_extraction_running(False)
    assert page._btn_reocr_frame.isEnabled()


def test_extract_buttons_unlock_after_batch_and_lock_during_reocr(container) -> None:
    from subtitles_extractor.presentation.pages.extract_page import ExtractPage

    page = ExtractPage(container)
    page._batch_queue = ["tap01"]
    page._emit_extraction_running()
    page._on_busy_changed(False)                   # tập cuối xong, hàng đợi chưa dọn
    assert not page._bench_button.isEnabled()
    page._batch_queue = []
    page._emit_extraction_running()                # _finish_batch dọn hàng đợi
    assert page._bench_button.isEnabled()          # rà soát: trước đây kẹt khoá tới khi nạp video khác

    page.set_reocr_running(True)                   # Re-OCR ở trang Chỉnh sửa
    assert not page._bench_button.isEnabled() and not page._batch_button.isEnabled()
    assert not page._start_button.isEnabled()
    page.set_reocr_running(False)
    assert page._bench_button.isEnabled() and page._batch_button.isEnabled()
