"""[v3.23.613] Trang Nhật ký phải giữ ĐỦ traceback khi sink chạy qua hàng đợi.

Ứng dụng gắn sink trang Nhật ký với ``enqueue=True``; Loguru pickle bản ghi để đưa qua
hàng đợi và BỎ khung stack của exception. Sink v3.23.486 dựng lại traceback từ bản ghi
đã mất khung nên chỉ còn "ValueError: …" — app_log thật của người dùng có ba lần
"Lỗi không mong đợi khi mở hộp thoại xuất tệp." mà không một dòng nào cho biết lỗi ở đâu.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from loguru import logger
from PySide6.QtCore import QCoreApplication

from subtitles_extractor.infrastructure.logging.qt_log_handler import QtLogBridge


@pytest.fixture(scope="module")
def app() -> QCoreApplication:
    return QCoreApplication.instance() or QCoreApplication([])


def _crash_in_export_dialog() -> None:
    raise ValueError("đường dẫn hỏng")


def _logged_crash(bridge: QtLogBridge, marker: str) -> str:
    try:
        _crash_in_export_dialog()
    except ValueError:
        logger.exception(marker)
    logger.complete()
    return next(e.message for e in reversed(bridge.snapshot()) if marker in e.message)


def test_attached_sink_keeps_the_stack(app: QCoreApplication) -> None:
    bridge = QtLogBridge()
    handler_id = bridge.attach(logger)
    try:
        message = _logged_crash(bridge, "Lỗi thử 613 (attach)")
    finally:
        logger.remove(handler_id)
    assert "Traceback" in message
    assert "_crash_in_export_dialog" in message
    assert "ValueError: đường dẫn hỏng" in message


def test_normal_line_has_no_trailing_text(app: QCoreApplication) -> None:
    bridge = QtLogBridge()
    handler_id = bridge.attach(logger)
    try:
        logger.info("dòng thường 613")
        logger.complete()
    finally:
        logger.remove(handler_id)
    assert [e.message for e in bridge.snapshot() if "613" in e.message] == [
        "dòng thường 613"
    ]


def test_gui_bootstrap_uses_the_attached_sink(app: QCoreApplication) -> None:
    from subtitles_extractor.composition.bootstrap import bootstrap_for_gui

    container = bootstrap_for_gui()
    message = _logged_crash(container.log_bridge, "Lỗi thử 613 (bootstrap)")
    assert "_crash_in_export_dialog" in message
