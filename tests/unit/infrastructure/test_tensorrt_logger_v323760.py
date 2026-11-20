"""[v3.23.760] Lỗi gốc của TensorRT vào log ứng dụng.

Log 19/11 12:39 (bản 3.23.758): cầu NumPy ↔ TensorRT chạy, dựng TensorRT lỗi "__len__() should return >= 0" (tensor mang
số chiều -1 — một lớp mạng phía trước dựng LỖI) nhưng lỗi gốc mất: Paddle tạo ``trt.Logger(trt.Logger.ERROR)``, bộ ghi
mặc định in ra stderr — exe cửa sổ không có stderr. Kiểm trên bindings TensorRT 10.9 thật: lớp con ``trt.ILogger`` nhận
được thông báo (``init_libnvinfer_plugins`` ghi ~40 dòng qua nó).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from subtitles_extractor.infrastructure.ocr import tensorrt_runtime_plan as plan


class _Severity:
    INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE = "internal", "error", "warning", "info", "verbose"


def _fake_trt():  # noqa: ANN202
    class ILogger:
        def __init__(self) -> None:
            self.base_init = True

    class Logger(ILogger):
        INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE = (_Severity.INTERNAL_ERROR, _Severity.ERROR, _Severity.WARNING,
                                                         _Severity.INFO, _Severity.VERBOSE)
        Severity = _Severity

        def __init__(self, min_severity=None) -> None:  # noqa: ANN001
            super().__init__()

    class Weights:
        def __init__(self, *args) -> None:  # noqa: ANN002
            pass

    return SimpleNamespace(ILogger=ILogger, Logger=Logger, Weights=Weights, DataType=SimpleNamespace(),
                           INetworkDefinition=type("Network", (), {}), Builder=type("Builder", (), {}))


@pytest.fixture(autouse=True)
def _clear() -> None:  # noqa: PT004
    plan._RECENT_ERRORS.clear()


def test_paddle_style_logger_forwards_messages(caplog: pytest.LogCaptureFixture) -> None:
    trt = _fake_trt()
    assert plan.patch_tensorrt_numpy_bridge(trt)
    lg = trt.Logger(trt.Logger.ERROR)                        # đúng như paddle/tensorrt/converter.py:143
    assert isinstance(lg, trt.ILogger) and lg.base_init and trt.Logger.Severity is _Severity
    with caplog.at_level(logging.DEBUG, logger=plan.logger.name):
        lg.log(trt.Logger.ERROR, "ShuffleLayer: reshape volume mismatch")
        lg.log(trt.Logger.WARNING, "tactic chậm")
        lg.log(trt.Logger.INFO, "MemUsageChange")
    levels = {(record.levelname, record.getMessage()) for record in caplog.records}
    assert ("WARNING", "[TensorRT] ShuffleLayer: reshape volume mismatch") in levels
    assert ("DEBUG", "[TensorRT] tactic chậm") in levels                # dưới mức Paddle chọn (ERROR): debug
    assert plan.tensorrt_recent_errors() == ["ShuffleLayer: reshape volume mismatch"]


def test_recent_errors_are_bounded_and_logger_never_raises() -> None:
    trt = _fake_trt()
    plan.patch_tensorrt_numpy_bridge(trt)
    lg = trt.Logger()
    for index in range(50):
        lg.log(trt.Logger.INTERNAL_ERROR, f"lỗi {index}")
    assert len(plan.tensorrt_recent_errors()) == plan._RECENT_ERRORS_MAX
    assert plan.tensorrt_recent_errors()[-1] == "lỗi 49"
    lg.log(object(), object())                                 # mức lạ: không ném ngoại lệ về C++


def test_logger_installed_once() -> None:
    trt = _fake_trt()
    plan.patch_tensorrt_numpy_bridge(trt)
    installed = trt.Logger
    plan.patch_tensorrt_numpy_bridge(trt)
    assert trt.Logger is installed and installed._se_app_logger


def test_adapter_logs_trt_errors_and_traceback_on_fallback(caplog: pytest.LogCaptureFixture) -> None:
    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter

    plan._RECENT_ERRORS.append("[shuffleNode.cpp::symbolicExecute] reshape lỗi")
    try:
        try:
            raise ValueError("__len__() should return >= 0")
        except ValueError as cause:
            raise RuntimeError("Cấu hình PaddleOCR không hợp lệ") from cause
    except RuntimeError as error:
        with caplog.at_level(logging.DEBUG, logger=paddle_ocr_adapter.logger.name):
            paddle_ocr_adapter._log_tensorrt_failure_details(error)
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "reshape lỗi" in text and "__len__() should return >= 0" in text and "Traceback" in text
