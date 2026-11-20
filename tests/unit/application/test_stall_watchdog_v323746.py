"""[v3.23.746] Lượt "Xuất dữ liệu thô OCR" treo 4 giờ sau "Nạp bộ đọc dòng PaddleOCR-VL-1.6-0.9B (gpu)" (log 18/11 21:34,
GPU 3 %, VRAM 7,6/8,0 GB) mà nhật ký không cho biết luồng nào kẹt ở đâu.

* Đồng hồ canh treo: không có tiến độ ≥ 3 phút → ghi ngăn xếp mọi luồng vào nhật ký (lần sau có bằng chứng).
* Trước bước soát câu bằng VL / gói kiểm định: trả bộ nhớ đệm VRAM bước OCR để lại + ghi VRAM còn trống.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.stall_watchdog import (
    MAX_DUMPS,
    StallWatchdog,
    WatchedReporter,
    thread_stacks,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_dumps_all_thread_stacks_after_silence_and_resets_on_progress() -> None:
    clock, dumps = _Clock(), []
    watchdog = StallWatchdog(stall_sec=180, repeat_sec=600, clock=clock,
                             dump=lambda message, idle, stacks: dumps.append((message, idle, stacks)))
    watchdog.touch("Soát 40 câu nghi ngờ")
    clock.now = 179
    assert not watchdog.check()
    clock.now = 181
    assert watchdog.check()
    message, idle, stacks = dumps[0]
    assert message == "Soát 40 câu nghi ngờ" and idle == pytest.approx(181)
    assert "test_dumps_all_thread_stacks_after_silence" in stacks          # ngăn xếp luồng đang chạy có trong bản ghi
    clock.now = 500
    assert not watchdog.check()                                             # lặp lại sau 600 s, không dồn dập
    for step in range(1, 5):
        clock.now = 181 + 600 * step
        watchdog.check()
    assert len(dumps) == MAX_DUMPS
    watchdog.touch("tiếp tục")
    clock.now += 100
    assert not watchdog.check()


def test_stacks_include_blocked_worker_thread() -> None:
    gate = threading.Event()

    def stuck_in_vl_generate() -> None:
        gate.wait(10)

    worker = threading.Thread(target=stuck_in_vl_generate, name="ocr-worker")
    worker.start()
    try:
        stacks = thread_stacks()
    finally:
        gate.set()
        worker.join()
    assert "Luồng ocr-worker" in stacks and "stuck_in_vl_generate" in stacks


def test_dump_written_to_log_as_warning(caplog: pytest.LogCaptureFixture) -> None:
    clock = _Clock()
    watchdog = StallWatchdog(stall_sec=1, clock=clock)
    clock.now = 5
    with caplog.at_level(logging.WARNING):
        watchdog.check()
    assert any("không có tiến độ" in r.getMessage() and "ngăn xếp" in r.getMessage() for r in caplog.records)


def test_reporter_wrapper_feeds_watchdog_and_forwards() -> None:
    clock, seen = _Clock(), []
    watchdog = StallWatchdog(stall_sec=10, clock=clock, dump=lambda *a: seen.append("dump"))
    inner = SimpleNamespace(report=lambda *a, **k: seen.append(a), is_cancelled=lambda: False)
    reporter = WatchedReporter(inner, watchdog)
    clock.now = 9
    reporter.report(96, 100, "Đang soát lại câu nghi ngờ 1/40…")
    clock.now = 15
    assert not watchdog.check() and seen == [(96, 100, "Đang soát lại câu nghi ngờ 1/40…")]
    assert reporter.is_cancelled() is False


def test_use_case_execute_runs_under_watchdog() -> None:
    source = (Path(__file__).resolve().parents[3] / "src/subtitles_extractor/application/use_cases/"
              "extract_subtitles.py").read_text(encoding="utf-8")
    assert "with StallWatchdog() as watchdog:" in source
    assert "self._execute(request, WatchedReporter(" in source


def test_gpu_cache_trimmed_and_logged_before_vl_stages(caplog: pytest.LogCaptureFixture) -> None:
    from subtitles_extractor.application.use_cases.extract_subtitles import (
        ExtractSubtitlesUseCase,
    )

    use_case = ExtractSubtitlesUseCase.__new__(ExtractSubtitlesUseCase)
    calls: list[str] = []
    use_case._ocr_engine_port = SimpleNamespace(trim_gpu_memory=lambda: calls.append("trim") or "VRAM trống 5.1/8.0 GB")
    with caplog.at_level(logging.INFO):
        use_case._trim_gpu_memory("trước bước soát câu")
    assert calls == ["trim"]
    assert any("Trước bước soát câu: VRAM trống 5.1/8.0 GB" in r.getMessage() for r in caplog.records)
    source = (Path(__file__).resolve().parents[3] / "src/subtitles_extractor/application/use_cases/"
              "extract_subtitles.py").read_text(encoding="utf-8")
    assert 'self._trim_gpu_memory("trước bước soát câu")' in source
    assert 'self._trim_gpu_memory("trước gói kiểm định")' in source


def test_adapter_trim_keeps_main_model() -> None:
    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import (
        PaddleOcrAdapter,
    )

    adapter = PaddleOcrAdapter.__new__(PaddleOcrAdapter)
    adapter._infer_lock = threading.RLock()
    adapter._model = sentinel = object()
    emptied: list[str] = []
    adapter._empty_paddle_vram = lambda: emptied.append("paddle")      # type: ignore[method-assign]
    adapter._empty_cupy_pool = lambda: emptied.append("cupy")           # type: ignore[method-assign]
    assert isinstance(adapter.trim_gpu_memory(), str)
    assert emptied == ["paddle", "cupy"] and adapter._model is sentinel
