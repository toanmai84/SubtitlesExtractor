"""[v3.23.643] Ghi MỌI ngoại lệ chưa bắt và crash native vào nhật ký.

Bản đóng gói ``console=False`` thay ``sys.stderr`` bằng luồng giả (xem ``main._ensure_std_streams``),
nên traceback của ngoại lệ trong slot Qt / luồng nền và crash native (Paddle, CUDA) không để lại dấu
vết nào — chỉ thấy "app biến mất". Ba lớp:

* ``sys.excepthook``: ngoại lệ chưa bắt ở luồng chính (kể cả trong slot Qt);
* ``threading.excepthook``: ngoại lệ chưa bắt trong ``threading.Thread``;
* ``faulthandler``: SIGSEGV/abort native → ``fault.log`` cạnh nhật ký.
"""

from __future__ import annotations

import contextlib
import faulthandler
import os
import sys
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from loguru import logger

__all__ = [
    "begin_session",
    "end_session",
    "mpv_disabled_by_crash",
    "install_crash_hooks",
    "note_fatal",
    "pop_previous_crash",
    "rearm",
]

_FAULT_FILE: IO[str] | None = None
_INSTALLED = False
_SESSION_MARKER: Path | None = None
_PREVIOUS_CRASH: dict[str, object] | None = None
_KEEP_FAULT_LOGS = 5
_LOG_DIR: Path | None = None
_MPV_FLAG = "mpv_disabled.flag"


def _log_exception(
    context: str,
    exc_type: type[BaseException],
    exc: BaseException | None,
    tb: TracebackType | None,
) -> None:
    """Ghi một ngoại lệ chưa bắt; KHÔNG BAO GIỜ ném lại (hook ném sẽ che lỗi gốc)."""
    try:
        logger.opt(exception=(exc_type, exc, tb)).critical("Ngoại lệ CHƯA BẮT ({}).", context)
    except Exception:  # noqa: BLE001 - bộ ghi lỗi không được phép gây lỗi mới
        pass


def install_crash_hooks(log_dir: Path | None) -> None:
    """Cài các hook (gọi một lần ở bootstrap; gọi lại là no-op).

    Args:
        log_dir: Thư mục nhật ký — ``fault.log`` của faulthandler nằm ở đây. ``None`` → bỏ faulthandler.
    """
    global _FAULT_FILE, _INSTALLED, _LOG_DIR  # noqa: PLW0603
    if _INSTALLED:
        return
    _INSTALLED = True
    _LOG_DIR = log_dir

    previous_hook = sys.excepthook

    def _excepthook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            previous_hook(exc_type, exc, tb)
            return
        _log_exception("luồng chính/slot", exc_type, exc, tb)
        try:
            previous_hook(exc_type, exc, tb)
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = _excepthook

    previous_thread_hook = threading.excepthook

    def _thread_hook(args: Any) -> None:
        if args.exc_type is SystemExit:
            return
        thread_name = getattr(args.thread, "name", "?")
        _log_exception(f"luồng '{thread_name}'", args.exc_type, args.exc_value, args.exc_traceback)
        try:
            previous_thread_hook(args)
        except Exception:  # noqa: BLE001
            pass

    threading.excepthook = _thread_hook

    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            _archive_previous_fault_log(log_dir)
            _FAULT_FILE = (log_dir / "fault.log").open("a", encoding="utf-8")
            faulthandler.enable(file=_FAULT_FILE, all_threads=True)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Không bật được faulthandler: {}.", exc)


def _archive_previous_fault_log(log_dir: Path) -> None:
    """[v3.23.650] ``fault.log`` không rỗng = lần chạy trước có crash native: cất thành ``fault_<thời điểm>.log``.

    Giữ 5 bản mới nhất. Ghi nhớ vào ``_PREVIOUS_CRASH`` để giao diện báo cho người dùng biết (một lần).
    """
    global _PREVIOUS_CRASH  # noqa: PLW0603
    current = log_dir / "fault.log"
    try:
        if not current.exists() or current.stat().st_size == 0:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(current.stat().st_mtime))
        target = log_dir / f"fault_{stamp}.log"
        counter = 1
        while target.exists():
            target = log_dir / f"fault_{stamp}_{counter}.log"
            counter += 1
        # [v3.23.651] Crash NATIVE trong mpv (access violation + khung mpv.py) → lần chạy sau tạm dùng VideoCanvas
        # thay vì lặp lại crash. (Các dòng 0xe24c4a02 là ngoại lệ Lua đã được xử lý, không phải crash.)
        mpv_crash = False
        try:
            text = current.read_text(encoding="utf-8", errors="replace")
            mpv_crash = "access violation" in text and "mpv" in text.lower()
        except OSError:
            pass
        current.replace(target)
        if mpv_crash:
            with contextlib.suppress(OSError):
                (log_dir / _MPV_FLAG).write_text(f"{target.name}\n", encoding="utf-8")
        _PREVIOUS_CRASH = {"fault_log": target, "log_dir": log_dir, "mpv_disabled": mpv_crash}
        for old in sorted(log_dir.glob("fault_*.log"))[:-_KEEP_FAULT_LOGS]:
            with contextlib.suppress(OSError):
                old.unlink()
    except OSError as exc:
        logger.warning("Không cất được fault.log cũ: {}.", exc)


def rearm() -> None:
    """[v3.23.650] Cài lại ``faulthandler`` SAU khi paddle đã nạp.

    Paddle tự cài trình xử lý SIGSEGV/SIGABRT của nó (in "FatalError…" ra fd 2 — không đâu cả khi ``console=False``)
    và đè lên faulthandler: sau đó crash native để ``fault.log`` rỗng. ``enable()`` là no-op khi đang bật nên phải
    ``disable()`` trước.
    """
    if _FAULT_FILE is None:
        return
    try:
        faulthandler.disable()
        faulthandler.enable(file=_FAULT_FILE, all_threads=True)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug("Không cài lại được faulthandler: {}.", exc)


def note_fatal(message: str) -> None:
    """Ghi ĐỒNG BỘ một thông điệp chí tử (Qt fatal) vào ``fault.log`` — loguru xếp hàng nên bị mất khi ``abort()``."""
    if _FAULT_FILE is None:
        return
    try:
        _FAULT_FILE.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        _FAULT_FILE.flush()
    except (OSError, ValueError):
        pass


def _pid_alive(pid: int) -> bool:
    """Tiến trình ``pid`` còn sống? (Windows: OpenProcess — KHÔNG dùng ``os.kill(pid, 0)`` vì ở Windows nó kết thúc tiến trình.)"""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except (OSError, AttributeError):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def begin_session(log_dir: Path) -> None:
    """Đánh dấu phiên đang chạy; nếu dấu còn từ lần trước (không qua ``end_session``) và tiến trình đó đã chết thì lần
    trước kết thúc bất thường. Dấu của một bản ứng dụng KHÁC đang chạy song song không bị coi là crash."""
    global _SESSION_MARKER, _PREVIOUS_CRASH  # noqa: PLW0603
    marker = log_dir / "session.running"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        if marker.exists():
            try:
                other_pid = int(marker.read_text(encoding="utf-8").split()[0])
            except (OSError, ValueError, IndexError):
                other_pid = 0
            if other_pid != os.getpid() and _pid_alive(other_pid):
                return  # bản khác đang chạy: không đè dấu của nó, cũng không báo crash
            if _PREVIOUS_CRASH is None:
                _PREVIOUS_CRASH = {"fault_log": None, "log_dir": log_dir}
        marker.write_text(f"{os.getpid()} {int(time.time())}", encoding="utf-8")
        _SESSION_MARKER = marker
    except OSError as exc:
        logger.debug("Không ghi được dấu phiên: {}.", exc)


def end_session() -> None:
    """Xoá dấu phiên khi thoát bình thường."""
    global _SESSION_MARKER  # noqa: PLW0603
    if _SESSION_MARKER is not None:
        with contextlib.suppress(OSError):
            _SESSION_MARKER.unlink()
        _SESSION_MARKER = None


def pop_previous_crash() -> dict[str, object] | None:
    """Thông tin lần chạy trước kết thúc bất thường (chỉ trả MỘT lần), hoặc None."""
    global _PREVIOUS_CRASH  # noqa: PLW0603
    info, _PREVIOUS_CRASH = _PREVIOUS_CRASH, None
    return info


def mpv_disabled_by_crash() -> bool:
    """mpv bị tắt vì lần chạy trước crash native trong mpv? (``SUBTITLES_EXTRACTOR_FORCE_MPV=1`` bỏ qua lớp bảo vệ.)"""
    if os.environ.get("SUBTITLES_EXTRACTOR_FORCE_MPV"):
        return False
    return _LOG_DIR is not None and (_LOG_DIR / _MPV_FLAG).exists()
