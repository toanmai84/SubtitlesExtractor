"""[v3.23.754] Bật TensorRT → "Khởi tạo OCR trên GPU thất bại — tự động lùi về CPU" (log 19/11 10:00): dựng lỗi sau
0,12 s, cả phim OCR trên CPU (lô 512, 16 luồng) — ứng dụng ì, chậm; log không ghi lý do.

Chuỗi lùi đúng (CLAUDE.md): GPU + TensorRT → GPU không TensorRT → CPU, ghi lý do ở từng bước; TensorRT lỗi một lần thì
cả phiên không thử lại.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from subtitles_extractor.domain.exceptions import OcrModelLoadError
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.domain.value_objects.device_kind import DeviceKind
from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter


def _adapter(monkeypatch: pytest.MonkeyPatch, fails: Any) -> tuple[PaddleOcrAdapter, list[dict[str, Any]]]:
    adapter = PaddleOcrAdapter.__new__(PaddleOcrAdapter)
    adapter._config = OcrEngineConfig(device=DeviceKind.GPU, use_tensorrt=True)
    adapter._tensorrt_unavailable = False
    calls: list[dict[str, Any]] = []

    def create(_cls: Any, kwargs: dict[str, Any]) -> object:
        calls.append(kwargs)
        if fails(kwargs):
            raise OcrModelLoadError(f"Không khởi tạo được mô hình PaddleOCR: lỗi giả ({kwargs['device']}).")
        return object()

    monkeypatch.setattr(adapter, "_try_create_model_with_retry", create)
    # [v3.23.766] TensorRT đã tải + tự kiểm đạt (chưa tải thì nay không thử TensorRT trần) — kiểm đúng nhánh lùi khi DỰNG.
    from pathlib import Path

    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter as module
    from subtitles_extractor.infrastructure.ocr.tensorrt_selftest import SelfTestOutcome

    monkeypatch.setattr(module, "_prepare_paddle_tensorrt", lambda: Path("trt"))
    monkeypatch.setattr(module, "_tensorrt_crashed_last_time", lambda trt_dir: False)
    monkeypatch.setattr(module, "_tensorrt_thread_ok", lambda: True)                 # [v3.23.767] đang ở luồng Paddle
    monkeypatch.setattr(module, "_tensorrt_selftest", lambda *a: SelfTestOutcome(True, "giả lập: đạt"))
    return adapter, calls


def test_direct_tensorrt_unavailable_falls_back_to_plain_gpu_not_cpu(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """[v3.23.774] Tự kiểm đạt → mô hình GPU thường + runner TensorRT trực tiếp (predictor TensorRT của Paddle sập với mọi
    engine — không dựng nữa). Không gắn được → vẫn GPU (không CPU), nhớ cho cả phiên."""
    from subtitles_extractor.infrastructure.ocr import trt_direct_runner

    def fail(model, **_kw):  # noqa: ANN001, ANN003, ANN202
        raise trt_direct_runner.DirectTrtUnavailable("lỗi giả")

    monkeypatch.setattr(trt_direct_runner, "attach_direct_tensorrt", fail)
    adapter, calls = _adapter(monkeypatch, lambda kw: False)
    with caplog.at_level(logging.WARNING):
        adapter._create_model_with_device_fallback(object, use_gpu=True)
    assert [(c["device"], bool(c.get("use_tensorrt"))) for c in calls] == [("gpu", False)]
    assert adapter._active_device == DeviceKind.GPU and adapter._tensorrt_unavailable
    assert any("TensorRT trực tiếp không gắn được" in r.getMessage() and "lỗi giả" in r.getMessage()
               for r in caplog.records)


def test_cpu_only_when_plain_gpu_also_fails_and_reason_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    adapter, calls = _adapter(monkeypatch, lambda kw: kw["device"] == "gpu")
    with caplog.at_level(logging.WARNING):
        adapter._create_model_with_device_fallback(object, use_gpu=True)
    assert [c["device"] for c in calls] == ["gpu", "cpu"] and adapter._active_device == DeviceKind.CPU
    assert any("lùi về CPU" in r.getMessage() and "lỗi giả (gpu)" in r.getMessage() for r in caplog.records)
