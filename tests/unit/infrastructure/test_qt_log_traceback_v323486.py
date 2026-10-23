"""[v3.23.486] Sink trang Nhật ký phải KÈM traceback.

Trước đây sink chỉ lấy ``record["message"]`` → bản log người dùng xuất từ trang Nhật ký chỉ có
MỘT dòng "Lỗi không lường trước..." (app_log), không đủ chẩn đoán crash.
"""
from __future__ import annotations

import sys
import types

import pytest


def _stub_pyside() -> None:
    """Sandbox/CI có thể không có Qt — chỉ cần logic sink, không cần Qt thật."""
    if "PySide6.QtCore" in sys.modules:
        return
    try:
        # [v3.23.613] Có Qt thật thì dùng Qt thật: thay PySide6 bằng bản giả ở đây làm
        # hỏng mọi test Qt chạy SAU trong cùng phiên (phụ thuộc thứ tự chạy).
        import PySide6.QtCore  # noqa: F401
    except ImportError:
        pass
    else:
        return
    qtcore = types.ModuleType("PySide6.QtCore")

    class _QObject:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _Signal:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __get__(self, obj, cls):
            return self

        def emit(self, *args) -> None:
            pass

    qtcore.QObject = _QObject
    qtcore.Signal = lambda *a, **k: _Signal()
    pyside = types.ModuleType("PySide6")
    pyside.QtCore = qtcore
    sys.modules["PySide6"] = pyside
    sys.modules["PySide6.QtCore"] = qtcore


@pytest.fixture
def bridge():
    _stub_pyside()
    from loguru import logger

    from subtitles_extractor.infrastructure.logging.qt_log_handler import QtLogBridge

    instance = QtLogBridge()
    logger.remove()
    logger.add(instance.loguru_sink, level="INFO")
    yield instance
    logger.remove()


def test_exception_includes_traceback(bridge) -> None:
    from loguru import logger

    try:
        raise ValueError("crash mô phỏng")
    except ValueError:
        logger.exception("Lỗi không lường trước khi dịch phụ đề.")
    message = bridge.snapshot()[-1].message
    assert "Traceback" in message
    assert "ValueError" in message
    assert "crash mô phỏng" in message


def test_normal_log_unchanged(bridge) -> None:
    from loguru import logger

    logger.info("dòng bình thường")
    assert bridge.snapshot()[-1].message == "dòng bình thường"


def test_long_traceback_is_truncated() -> None:
    _stub_pyside()
    from subtitles_extractor.infrastructure.logging.qt_log_handler import (
        _MAX_TRACEBACK_CHARS, _format_exception,
    )

    text = _format_exception((ValueError, ValueError("x" * 9000), None))
    assert len(text) <= _MAX_TRACEBACK_CHARS + 100
    assert "đã cắt bớt" in text


def test_malformed_exception_does_not_raise() -> None:
    _stub_pyside()
    from subtitles_extractor.infrastructure.logging.qt_log_handler import _format_exception

    assert _format_exception("không phải bộ ba") == "không phải bộ ba"
