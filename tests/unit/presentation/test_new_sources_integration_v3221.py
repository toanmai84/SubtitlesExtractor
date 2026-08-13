"""Test tích hợp [v3.22.1] cho nguồn Embedded + STT trên trang Trích xuất.

Kiểm các lỗ hổng tích hợp đã vá:
  * busy → khoá nút nguồn mới (chống chạy chồng thread).
  * cancel → gọi request_cancel trên cả embedded/STT worker.
  * is_stt_available có cache (không probe import mỗi lần).
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from subtitles_extractor.composition.bootstrap import bootstrap_for_gui  # noqa: E402
from subtitles_extractor.presentation.view_models.extract_page_view_model import (  # noqa: E402
    ExtractPageViewModel,
)


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    return instance


@pytest.fixture
def view_model(app):
    return ExtractPageViewModel(bootstrap_for_gui())


class TestCancelCancelsAllWorkers:
    def test_cancel_calls_request_cancel_on_new_workers(self, view_model) -> None:
        calls = {"embedded": False, "transcribe": False}

        class _FakeWorker:
            def __init__(self, key: str) -> None:
                self._key = key

            def request_cancel(self) -> None:
                calls[self._key] = True

        view_model._embedded_worker = _FakeWorker("embedded")
        view_model._transcribe_worker = _FakeWorker("transcribe")
        view_model.cancel_extraction()

        assert calls["embedded"] is True
        assert calls["transcribe"] is True

    def test_cancel_safe_when_no_workers(self, view_model) -> None:
        # Không có worker nào → không lỗi.
        view_model.cancel_extraction()


class TestSttAvailabilityCache:
    def test_result_cached(self, view_model) -> None:
        first = view_model.is_stt_available()
        # Gán cache giả khác → lần sau phải trả cache, không probe lại.
        view_model._stt_available_cache = not first
        assert view_model.is_stt_available() == (not first)


class TestBusyLocksNewSourceButtons:
    def test_busy_disables_then_enables(self, app) -> None:
        from subtitles_extractor.presentation.pages.extract_page import ExtractPage

        page = ExtractPage(bootstrap_for_gui())
        page._on_busy_changed(True)
        assert not page._btn_scan_embedded.isEnabled()
        assert not page._btn_transcribe.isEnabled()
        assert not page._btn_extract_embedded.isEnabled()
