"""[v3.23.443] Hồi quy: InterceptHandler KHÔNG được mất log khi stack nông (luồng worker).

Bug: sys._getframe(6) giả định cứng 6 frame → ValueError trên stack nông → except nuốt →
log ERROR (kể cả traceback lỗi dịch) không tới sink/nhật ký UI, dù popup vẫn hiện.
"""
from __future__ import annotations
import logging
import threading
import time
from loguru import logger as _lg
from subtitles_extractor.infrastructure.logging.loguru_config import InterceptHandler


def _capture_via_intercept(log_fn) -> list[str]:
    captured: list[str] = []
    _lg.remove()
    sink_id = _lg.add(lambda m: captured.append(m.record["message"]), level="DEBUG")
    root = logging.getLogger()
    old_handlers, old_level = root.handlers, root.level
    root.handlers = [InterceptHandler()]
    root.setLevel(logging.DEBUG)
    try:
        log_fn()
        time.sleep(0.05)
    finally:
        root.handlers, root.level = old_handlers, old_level
        _lg.remove(sink_id)
    return captured


def test_worker_thread_exception_reaches_sink():
    def emit_from_thread():
        def worker():
            lg = logging.getLogger("test.worker")
            try:
                raise NameError("name 'self' is not defined")
            except NameError:
                lg.exception("Lỗi không lường trước khi dịch phụ đề.")
        t = threading.Thread(target=worker)
        t.start()
        t.join()

    captured = _capture_via_intercept(emit_from_thread)
    assert any("không lường trước" in m for m in captured), "LOG BỊ MẤT trên stack nông!"


def test_normal_log_still_works():
    captured = _capture_via_intercept(
        lambda: logging.getLogger("test.normal").info("thông điệp bình thường")
    )
    assert any("bình thường" in m for m in captured)


# [v3.23.449] Hồi quy: tên module ĐÚNG (không bị gán "logging")
import sys
import types


def _capture_names_via_intercept(log_fn) -> list[str]:
    import logging
    from loguru import logger as _lg
    names: list[str] = []
    _lg.remove()
    sink_id = _lg.add(lambda m: names.append(m.record["name"]), level="DEBUG")
    root = logging.getLogger()
    old_handlers, old_level = root.handlers, root.level
    root.handlers = [InterceptHandler()]
    root.setLevel(logging.DEBUG)
    try:
        log_fn()
        time.sleep(0.05)
    finally:
        root.handlers, root.level = old_handlers, old_level
        _lg.remove(sink_id)
    return names


def test_logger_name_is_real_module_not_logging():
    # Tạo module thật có __name__ riêng để log
    mod = types.ModuleType("fake.caller.module")
    exec(
        "import logging\n"
        "def do():\n"
        "    logging.getLogger(__name__).warning('hi')\n",
        mod.__dict__,
    )
    mod.__dict__["__name__"] = "fake.caller.module"
    names = _capture_names_via_intercept(mod.do)
    # KHÔNG được là 'logging' (bug v443); phải là tên module caller
    assert "logging" not in names
    assert any("fake" in n or "caller" in n or n != "logging" for n in names)
