"""[v3.23.650] ``is_initialized``/``release`` của PaddleOcrAdapter không được chặn luồng gọi khi ``initialize()`` đang nạp mô hình.

Lỗi gốc: ``is_initialized`` lấy ``_init_lock`` mà ``initialize()`` giữ suốt lúc import paddle + nạp mô hình; hàm này lại
được gọi từ luồng giao diện (showEvent) → cửa sổ đơ 7–19 s lúc khởi động và 4–6 s lúc đóng ứng dụng.
"""

from __future__ import annotations

import threading
import time

from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter


def _adapter() -> PaddleOcrAdapter:
    return PaddleOcrAdapter(OcrEngineConfig())


def _hold_lock(adapter: PaddleOcrAdapter) -> tuple[threading.Event, threading.Thread]:
    ready, done = threading.Event(), threading.Event()

    def holder() -> None:
        with adapter._init_lock:
            ready.set()
            done.wait(10)

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert ready.wait(5)
    return done, thread


def _timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def test_is_initialized_does_not_block_while_initialize_holds_lock_v323650() -> None:
    adapter = _adapter()
    done, thread = _hold_lock(adapter)
    try:
        assert _timed(lambda: adapter.is_initialized) < 0.2
        assert adapter.is_initialized is False
    finally:
        done.set()
        thread.join(5)


def test_release_does_not_block_and_is_applied_after_initialize_v323650() -> None:
    adapter = _adapter()
    done, thread = _hold_lock(adapter)
    try:
        assert _timed(adapter.release) < 0.2
        assert adapter._release_after_init is True
    finally:
        done.set()
        thread.join(5)


def test_stale_release_request_is_dropped_on_next_initialize_v323650() -> None:
    """release() bị hoãn rồi không ai tiêu thụ cờ (đường thoát sớm): lần initialize sau không được tự xoá mô hình."""
    adapter = _adapter()
    sentinel = object()
    adapter._model = sentinel
    adapter._release_after_init = True          # tồn dư từ lần release() hoãn trước đó
    adapter.initialize()                        # mô hình đã có → đường thoát sớm
    assert adapter._model is sentinel
    assert adapter._release_after_init is False
