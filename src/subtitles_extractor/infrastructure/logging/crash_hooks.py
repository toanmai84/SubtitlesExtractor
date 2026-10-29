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
import re
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from loguru import logger

__all__ = [
    "begin_session",
    "end_session",
    "mpv_disabled_by_crash",
    "classify_fault_text",
    "install_crash_hooks",
    "mark_clean_exit",
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
            faulthandler.enable(file=_FAULT_FILE, all_threads=_FAULT_ALL_THREADS)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Không bật được faulthandler: {}.", exc)


#: [v3.23.665] Windows: faulthandler chạy cho MỌI ngoại lệ SEH lượt đầu, kể cả loại được bắt ngay (LuaJIT 0xe24c4a02
#: trong mpv, CUDA…). Với ``all_threads=True`` mỗi lần nó đọc ngăn xếp của mọi luồng Python mà KHÔNG giữ GIL trong khi
#: luồng chính đang chạy — kéo thanh seek sinh hàng trăm lần như vậy, một lần đọc trúng khung đang đổi là access
#: violation thật (fault.log người dùng 29/10: "access violation" chen giữa lúc đang in ngăn xếp luồng chính). Chỉ ghi
#: luồng đang lỗi: crash trên luồng chính vẫn có đủ ngăn xếp.
_FAULT_ALL_THREADS: bool = sys.platform != "win32"


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
        try:
            text = current.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        verdict = classify_fault_text(text)
        current.replace(target)
        if verdict.mpv_crash:
            with contextlib.suppress(OSError):
                (log_dir / _MPV_FLAG).write_text(f"{target.name}\n", encoding="utf-8")
        if verdict.shutdown_crash:
            logger.warning("Lần trước có lỗi native SAU khi đã đóng sạch (lúc dọn dẹp): {}.", target.name)
        if not verdict.fatal:
            # [v3.23.652] Chỉ có ngoại lệ đã được xử lý (vd LuaJIT 0xe24c4a02 của mpv) hoặc lỗi sau khi đóng sạch:
            # vẫn cất tệp để chẩn đoán nhưng KHÔNG báo "lần trước bị tắt đột ngột" (từng báo giả mỗi lần đóng app).
            _prune_fault_logs(log_dir)
            return
        _PREVIOUS_CRASH = {"fault_log": target, "log_dir": log_dir, "mpv_disabled": verdict.mpv_crash}
        _prune_fault_logs(log_dir)
    except OSError as exc:
        logger.warning("Không cất được fault.log cũ: {}.", exc)


def _prune_fault_logs(log_dir: Path) -> None:
    for old in sorted(log_dir.glob("fault_*.log"))[:-_KEEP_FAULT_LOGS]:
        with contextlib.suppress(OSError):
            old.unlink()


_CLEAN_EXIT_TAG = "[clean-exit"
# Chữ ký crash THẬT. Windows: faulthandler ghi mọi ngoại lệ SEH ở lượt đầu (kể cả đã được bắt) → không coi dòng
# "Windows fatal exception: code 0x…" là chí tử trừ mã nhóm lỗi 0xC… (vd 0xC0000409 tràn bộ đệm ngăn xếp).
_FATAL_PATTERNS = (
    re.compile(r"access violation", re.I),
    re.compile(r"stack overflow", re.I),
    re.compile(r"Fatal Python error"),
    re.compile(r"Segmentation fault"),
    re.compile(r"\bAborted\b"),
    re.compile(r"illegal instruction", re.I),
    re.compile(r"page error", re.I),
    re.compile(r"divide by zero|float overflow", re.I),
    re.compile(r"\[Qt FATAL\]"),
    re.compile(r"fatal exception: code 0xc[0-9a-f]{7}", re.I),
)


@dataclass(frozen=True)
class FaultVerdict:
    fatal: bool            # crash thật khi đang chạy → báo người dùng
    mpv_crash: bool        # crash thật nằm trong mpv → tạm tắt mpv lần sau
    shutdown_crash: bool   # lỗi native SAU khi đã đóng sạch (chỉ ghi log)


def classify_fault_text(text: str) -> FaultVerdict:
    """[v3.23.652] Phân loại nội dung ``fault.log`` của lần chạy trước."""
    before, marker, after = text.partition(_CLEAN_EXIT_TAG)

    def last_fatal(chunk: str) -> int:
        positions = [m.start() for pattern in _FATAL_PATTERNS for m in pattern.finditer(chunk)]
        return max(positions) if positions else -1

    if marker:
        # Đã đóng sạch: mọi "access violation" trước dấu là ngoại lệ lượt đầu đã được bắt (app vẫn chạy tiếp tới lúc đóng).
        return FaultVerdict(fatal=False, mpv_crash=False, shutdown_crash=last_fatal(after) >= 0)
    position = last_fatal(before)
    if position < 0:
        return FaultVerdict(fatal=False, mpv_crash=False, shutdown_crash=False)
    tail = before[position:]
    # all_threads=True: luồng sự kiện của python-mpv luôn có khung mpv.py — chỉ xét khối "Current thread" (luồng gây lỗi).
    current = re.search(r"Current thread[^\n]*\n(.*?)(?:\n\s*\n|\nThread 0x|\Z)", tail, re.S)
    scope = current.group(1) if current else tail
    return FaultVerdict(fatal=True, mpv_crash="mpv" in scope.lower(), shutdown_crash=False)


def mark_clean_exit() -> None:
    """[v3.23.652] Ghi dấu "đóng sạch" vào ``fault.log`` rồi tắt faulthandler (dọn dẹp sau đó không bị tính là crash)."""
    if _FAULT_FILE is not None:
        with contextlib.suppress(OSError, ValueError):
            _FAULT_FILE.write(f"{_CLEAN_EXIT_TAG} {time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
            _FAULT_FILE.flush()
    with contextlib.suppress(RuntimeError, ValueError):
        faulthandler.disable()
    end_session()


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
        faulthandler.enable(file=_FAULT_FILE, all_threads=_FAULT_ALL_THREADS)
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
