"""[v3.23.747] Giữ máy THỨC + màn hình SÁNG trong lúc chạy việc dài trên GPU (trích xuất, kiểm định, lồng tiếng, xuất video…).

Nguyên nhân gốc treo PaddleOCR-VL (log + Event Viewer máy chủ dự án 18/11): 23:04:17 Kernel-Power 566 — Windows tắt màn
hình vì không dùng (phiên trước 289 s, ``MonitorReason 31``); 23:04:31 VL sinh chữ lần đầu → lệnh GPU không bao giờ xong
(GPU 3 %); 23:12:26 nvlddmkm 153 khi tiến trình treo bị đóng. Ba lần 153 tối đó đều đi sau một lần 566. Lúc người dùng
ngồi máy (phim 86 tập) cùng trình tự chạy tốt. Màn hình tắt → GPU hạ trạng thái nguồn → công việc CUDA đang chờ bị kẹt.

Dùng yêu cầu nguồn của Windows (``PowerCreateRequest`` + ``PowerRequestSystemRequired`` + ``PowerRequestDisplayRequired``,
hiện trong ``powercfg /requests`` kèm lý do) — giống trình phát video; xong việc thì trả lại. Đếm tham chiếu: nhiều việc
chạy chồng nhau giữ một yêu cầu. API lỗi → lùi về ``SetThreadExecutionState``. Hệ điều hành khác: không làm gì.
"""

from __future__ import annotations

import functools
import logging
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

_POWER_REQUEST_DISPLAY_REQUIRED = 0
_POWER_REQUEST_SYSTEM_REQUIRED = 1
_POWER_REQUEST_CONTEXT_SIMPLE_STRING = 0x1
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002

_lock = threading.Lock()
_holders = 0
_request: Any = None
_reason_buffer: Any = None


def _kernel32() -> Any:
    """kernel32 với ``argtypes`` khai báo đủ — mặc định ctypes truyền số nguyên 32 bit → HANDLE 64 bit bị cắt cụt."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
    kernel32.PowerCreateRequest.restype = wintypes.HANDLE
    kernel32.PowerCreateRequest.argtypes = [ctypes.c_void_p]
    kernel32.PowerSetRequest.restype = wintypes.BOOL
    kernel32.PowerSetRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
    kernel32.PowerClearRequest.restype = wintypes.BOOL
    kernel32.PowerClearRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.SetThreadExecutionState.restype = wintypes.DWORD
    kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    return kernel32


def _win_acquire(reason: str) -> Any:
    """Tạo + bật yêu cầu nguồn; trả handle (hoặc ``"es"`` khi lùi về SetThreadExecutionState, ``None`` khi thất bại)."""
    global _reason_buffer  # noqa: PLW0603 — chuỗi lý do phải sống cùng yêu cầu
    import ctypes
    from ctypes import wintypes

    kernel32 = _kernel32()

    class _ReasonContext(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("Flags", wintypes.DWORD), ("SimpleReasonString", wintypes.LPWSTR)]

    _reason_buffer = ctypes.create_unicode_buffer(reason)
    context = _ReasonContext(0, _POWER_REQUEST_CONTEXT_SIMPLE_STRING, ctypes.cast(_reason_buffer, wintypes.LPWSTR))
    handle = kernel32.PowerCreateRequest(ctypes.byref(context))
    invalid = wintypes.HANDLE(-1).value
    if handle and handle != invalid:
        ok = all(kernel32.PowerSetRequest(handle, kind)
                 for kind in (_POWER_REQUEST_SYSTEM_REQUIRED, _POWER_REQUEST_DISPLAY_REQUIRED))
        if ok:
            return handle
        kernel32.CloseHandle(handle)
    if kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED):
        return "es"
    return None


def _win_release(handle: Any) -> None:
    kernel32 = _kernel32()
    if handle == "es":
        kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        return
    for kind in (_POWER_REQUEST_SYSTEM_REQUIRED, _POWER_REQUEST_DISPLAY_REQUIRED):
        kernel32.PowerClearRequest(handle, kind)
    kernel32.CloseHandle(handle)


@contextmanager
def keep_awake(reason: str = "Subtitles Extractor đang chạy việc dài trên GPU") -> Iterator[None]:
    """Giữ máy thức + màn hình sáng trong khối ``with`` (lồng nhau / nhiều luồng: một yêu cầu, đếm tham chiếu)."""
    global _holders, _request  # noqa: PLW0603
    with _lock:
        _holders += 1
        if _holders == 1 and sys.platform == "win32":
            try:
                _request = _win_acquire(reason)
                if _request is not None:
                    logger.info("Giữ máy thức + màn hình sáng trong lúc chạy (%s).", reason)
            except Exception as exc:  # noqa: BLE001 — không giữ được thì vẫn chạy việc
                _request = None
                logger.warning("Không giữ được máy thức trong lúc chạy: %s", exc)
    try:
        yield
    finally:
        with _lock:
            _holders -= 1
            if _holders == 0 and _request is not None:
                try:
                    _win_release(_request)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Trả yêu cầu giữ máy thức lỗi: %s", exc)
                _request = None


_F = TypeVar("_F", bound=Callable[..., Any])


def keeps_awake(reason: str) -> Callable[[_F], _F]:
    """Decorator cho ``run()`` của luồng chạy việc dài."""

    def decorate(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with keep_awake(reason):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


__all__ = ["keep_awake", "keeps_awake"]
