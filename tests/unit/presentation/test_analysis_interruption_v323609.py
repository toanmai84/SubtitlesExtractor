"""[v3.23.609] Phân tích ngữ cảnh bị gián đoạn mà người dùng KHÔNG bấm huỷ phải báo lỗi.

Khâu dịch đã phân biệt hai trường hợp từ v3.23.175 (``cancellation_outcome``): lỗi
``TranslationCancelledError`` cũng phát sinh khi chờ Gemini quá lâu / mất mạng. Worker
phân tích (và worker "Bổ sung người nói" kế thừa nó) thì coi mọi trường hợp là "người dùng
huỷ" → chỉ hiện "Đã huỷ phân tích." ở dòng trạng thái, lỗi thật bị giấu.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationCancelledError,
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


class _InterruptedUseCase:
    def execute(self, **_kwargs: object) -> object:
        raise TranslationCancelledError("Hết thời gian chờ Gemini")


def _run(cancel_first: bool) -> tuple[list[str], list[int]]:
    from subtitles_extractor.presentation.workers.analyze_context_worker import (
        AnalyzeContextWorker,
    )

    worker = AnalyzeContextWorker(_InterruptedUseCase(), [], "Vietnamese")
    failed: list[str] = []
    cancelled: list[int] = []
    worker.failed.connect(failed.append)
    worker.cancelled.connect(lambda: cancelled.append(1))
    if cancel_first:
        worker.request_cancel()
    worker.run()  # chạy ngay trên luồng test
    return failed, cancelled


def test_interruption_without_user_cancel_is_reported_as_failure(
    app: QApplication,
) -> None:
    failed, cancelled = _run(cancel_first=False)
    assert cancelled == []
    assert failed and "không phải do bạn huỷ" in failed[0]


def test_user_cancel_stays_quiet(app: QApplication) -> None:
    failed, cancelled = _run(cancel_first=True)
    assert failed == []
    assert cancelled == [1]
