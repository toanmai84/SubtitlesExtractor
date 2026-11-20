"""[v3.23.747] PaddleOCR-VL treo vĩnh viễn ở lần sinh chữ đầu tiên (log 18/11 23:11, đồng hồ canh treo 3.23.746 ghi
ngăn xếp: luồng trích xuất đứng trong ``_C_ops.embedding`` của ``greedy_search``; GPU 3 %, VRAM còn 5,9 GB).

Bộ đọc VL chạy trên luồng riêng có thời hạn: quá hạn → :class:`VlReadTimeoutError` (kèm ngăn xếp luồng VL trong nhật ký),
bộ chuyển đổi tắt VL cho phiên và giữ bản đọc PP-OCR — lượt chạy đi tiếp thay vì treo hàng giờ.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pytest

from subtitles_extractor.infrastructure.ocr import paddle_vl_reader as vl


class _Model:
    def __init__(self, hang: threading.Event | None = None) -> None:
        self.hang = hang
        self.threads: list[str] = []

    def predict(self, payload, max_new_tokens):  # noqa: ANN001, ANN201
        self.threads.append(threading.current_thread().name)
        if self.hang is not None:
            self.hang.wait(30)                         # giả lập lệnh GPU không bao giờ xong
        return [{"result": "门没关"}]


@pytest.fixture
def reader(monkeypatch: pytest.MonkeyPatch) -> vl.PaddleVlLineReader:
    monkeypatch.setattr(vl, "_bind_thread_device", lambda device: None)
    return vl.PaddleVlLineReader()


def test_normal_read_runs_on_dedicated_vl_thread(reader: vl.PaddleVlLineReader, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _Model()
    monkeypatch.setattr(reader, "_ensure", lambda device: model)
    assert reader.read([np.zeros((40, 200, 3), np.uint8)], device="gpu") == ["门没关"]
    assert model.threads == ["paddle-vl"]
    assert reader._thread is not None and reader._thread.thread.daemon         # không chặn ứng dụng thoát


def test_hung_generation_times_out_and_vl_disabled(
    reader: vl.PaddleVlLineReader, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    gate = threading.Event()
    model = _Model(hang=gate)
    monkeypatch.setattr(reader, "_ensure", lambda device: model)
    monkeypatch.setattr(vl, "VL_READ_TIMEOUT_GPU_SEC", 0.5)
    started = time.monotonic()
    try:
        with caplog.at_level(logging.WARNING), pytest.raises(vl.VlReadTimeoutError, match="quá"):
            reader.read([np.zeros((40, 200, 3), np.uint8)], device="gpu")
        assert time.monotonic() - started < 10
        assert any("Ngăn xếp luồng VL" in r.getMessage() and "predict" in r.getMessage() for r in caplog.records)
        with pytest.raises(vl.VlReadTimeoutError):                           # lần sau: báo ngay, không đợi lại
            reader.read([np.zeros((40, 200, 3), np.uint8)], device="gpu")
        reader.release()                                                     # luồng kẹt: không chờ
    finally:
        gate.set()


def test_load_has_its_own_generous_deadline(reader: vl.PaddleVlLineReader, monkeypatch: pytest.MonkeyPatch) -> None:
    assert vl.VL_LOAD_TIMEOUT_SEC >= 1800 > vl.VL_READ_TIMEOUT_GPU_SEC        # lần đầu còn tải ~2 GB mô hình
    loads: list[str] = []

    cached: list[_Model] = []

    def slow_ensure(device: str) -> _Model:
        if not cached:                                   # như ``_ensure`` thật: nạp một lần rồi dùng lại
            loads.append(threading.current_thread().name)
            time.sleep(0.8)                              # nạp lâu hơn thời hạn ĐỌC nhưng trong thời hạn NẠP
            cached.append(_Model())
        return cached[0]

    monkeypatch.setattr(reader, "_ensure", slow_ensure)
    monkeypatch.setattr(vl, "VL_READ_TIMEOUT_GPU_SEC", 0.5)
    assert reader.read([np.zeros((40, 200, 3), np.uint8)], device="gpu") == ["门没关"]
    assert set(loads) == {"paddle-vl"}


def test_adapter_falls_back_to_ppocr_on_timeout() -> None:
    import inspect

    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import (
        PaddleOcrAdapter,
    )

    source = inspect.getsource(PaddleOcrAdapter._vl_reread)
    assert "self._vl_reader.read(crops" in source and "self._vl_unavailable = True" in source
    assert issubclass(vl.VlReadTimeoutError, Exception)                      # rơi vào nhánh "VL lỗi → giữ bản nháp"
