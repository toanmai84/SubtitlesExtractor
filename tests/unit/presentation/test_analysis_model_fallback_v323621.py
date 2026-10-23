"""[v3.23.621] Phân tích tự lùi về model hạn mức cao khi model đã chọn hết hạn mức NGÀY.

Mặc định phân tích nay là gemini-3.5-flash (nhận người nói trong video chính xác nhất khi
chấm mù trên phim mẫu) nhưng chỉ 20 request/ngày/key — phim bộ 86 tập sẽ chạm trần.
Trước đây hết hạn mức là báo lỗi và dừng.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    QuotaExhaustedError,
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


class _QuotaUseCase:
    def __init__(self, exhausted: set[str]) -> None:
        self.exhausted = exhausted
        self.models: list[str] = []

    def execute(self, **kwargs: object) -> object:
        model = str(kwargs["model_name"])
        self.models.append(model)
        if model in self.exhausted:
            raise QuotaExhaustedError(f"Model '{model}' đã hết hạn mức request/ngày")
        return f"ok:{model}"


def _run(exhausted: set[str], fallback: str) -> tuple[_QuotaUseCase, list, list[str]]:
    from subtitles_extractor.presentation.workers.analyze_context_worker import (
        AnalyzeContextWorker,
    )

    use_case = _QuotaUseCase(exhausted)
    worker = AnalyzeContextWorker(
        use_case, [], "Vietnamese", model_name="gemini-3.5-flash", fallback_model=fallback
    )
    done: list = []
    failed: list[str] = []
    worker.finished_ok.connect(done.append)
    worker.failed.connect(failed.append)
    worker.run()
    return use_case, done, failed


def test_falls_back_when_daily_quota_exhausted(app: QApplication) -> None:
    use_case, done, failed = _run({"gemini-3.5-flash"}, "gemini-3.1-flash-lite")
    assert use_case.models == ["gemini-3.5-flash", "gemini-3.1-flash-lite"]
    assert done == ["ok:gemini-3.1-flash-lite"] and failed == []


def test_reports_when_fallback_also_exhausted(app: QApplication) -> None:
    use_case, done, failed = _run(
        {"gemini-3.5-flash", "gemini-3.1-flash-lite"}, "gemini-3.1-flash-lite"
    )
    assert len(use_case.models) == 2 and done == []
    assert failed and "hạn mức" in failed[0]


def test_no_fallback_keeps_old_behaviour(app: QApplication) -> None:
    use_case, done, failed = _run({"gemini-3.5-flash"}, "")
    assert use_case.models == ["gemini-3.5-flash"] and failed


def test_view_model_passes_fallback_and_page_default() -> None:
    from subtitles_extractor.presentation.pages import translate_page
    from subtitles_extractor.presentation.view_models import translate_page_view_model as vm

    assert translate_page.DEFAULT_ANALYSIS_MODEL == "gemini-3.8-flash"
    assert vm.ANALYSIS_FALLBACK_MODEL == "gemini-3.1-flash-lite"
