"""Ghi tệp nguyên tử (atomic write) — tránh để lại tệp dở dang.

Cơ chế:
    1. Ghi nội dung vào tệp tạm cùng thư mục với tệp đích.
    2. Sau khi flush, thực hiện ``os.fsync`` trên thread riêng với **timeout**
       để tránh block indefinitely (đặc biệt trên Windows + ổ chậm/NAS/USB).
    3. ``os.replace`` để rename atomic.
    4. Nếu có lỗi giữa chừng, xoá tệp tạm.

v3.6 — Sửa lỗi treo ứng dụng khi xuất SRT:
    ``os.fsync()`` trên Windows có thể block indefinitely khi:
    * Ghi lên ổ HDD chậm / NAS / USB drive
    * Antivirus đang scan file đồng thời
    * Disk gần đầy (Windows giữ write lock lâu hơn)
    Giải pháp: fsync chạy trên daemon thread với timeout 5s (configurable).
    Nếu timeout → WARNING + tiếp tục (best-effort durability, data đã ở OS buffer).

Cùng filesystem là quan trọng: ``os.replace`` chỉ atomic khi tệp tạm và
tệp đích nằm trên cùng một thiết bị.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Timeout mặc định (giây) cho os.fsync().
# 5s đủ rộng cho ổ SSD thông thường, đủ ngắn để không block UI quá lâu.
_DEFAULT_FSYNC_TIMEOUT_SEC: float = 5.0

# Env override để tắt fsync hoàn toàn (ví dụ: CI/CD, testing).
# Đặt SUBTITLES_EXTRACTOR_SKIP_FSYNC=1 để bỏ qua fsync.
_SKIP_FSYNC: bool = os.environ.get("SUBTITLES_EXTRACTOR_SKIP_FSYNC", "") == "1"


def _fsync_with_timeout(fileno: int, timeout_sec: float) -> bool:
    """Thực hiện ``os.fsync()`` trên daemon thread với timeout.

    Chạy fsync trên thread riêng để tránh block caller indefinitely.
    Thread là daemon → tự thoát khi main process kết thúc.

    Args:
        fileno:      File descriptor số nguyên.
        timeout_sec: Thời gian chờ tối đa (giây).

    Returns:
        ``True`` nếu fsync hoàn thành trong thời gian timeout.
        ``False`` nếu timeout (thread vẫn đang chạy).

    Raises:
        OSError: Nếu fsync hoàn thành nhưng trả lỗi hệ thống.
    """
    _completed = [False]
    _error: list[OSError | None] = [None]

    def _do_fsync() -> None:
        try:
            os.fsync(fileno)
            _completed[0] = True
        except OSError as exc:
            _error[0] = exc

    fsync_thread = threading.Thread(
        target=_do_fsync, daemon=True, name="atomic-fsync"
    )
    fsync_thread.start()
    fsync_thread.join(timeout=timeout_sec)

    if fsync_thread.is_alive():
        # Thread vẫn đang block → timeout, trả False để caller xử lý.
        return False

    if _error[0] is not None:
        raise _error[0]

    return _completed[0]


def _replace_with_retry(
    source: str,
    destination: str,
    *,
    max_attempts: int = 5,
    initial_delay_sec: float = 0.2,
) -> None:
    """``os.replace`` với Exponential Backoff Retry cho ``PermissionError``.

    Trên Windows, ``os.replace`` có thể ném ``PermissionError`` (WinError 5/32)
    khi Windows Defender đang quét tệp tạm hoặc ổ HDD/NAS chưa kịp nhả khoá. Ta
    đợi rồi thử lại, nhân đôi thời gian chờ mỗi lần (0.2s → 0.4s → 0.8s …), tối đa
    ``max_attempts`` lần trước khi chịu thua.

    Args:
        source:        Đường dẫn tệp tạm.
        destination:   Đường dẫn tệp đích.
        max_attempts:  Số lần thử tối đa.
        initial_delay_sec: Thời gian chờ ban đầu (giây).

    Raises:
        PermissionError: Nếu vẫn thất bại sau ``max_attempts`` lần.
        OSError: Các lỗi I/O khác (không retry).
    """
    delay = initial_delay_sec
    # [v3.20.3 #2] Ép Python giải phóng file descriptor còn lơ lửng (file tạm vừa
    # ghi) NGAY trước khi rename — giúp lách Windows Defender đang quét file tạm,
    # giảm xác suất PermissionError (WinError 5) ở lần thử đầu.
    gc.collect()
    for attempt in range(1, max_attempts + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if attempt >= max_attempts:
                logger.error(
                    "atomic_write: os.replace thất bại sau %d lần (PermissionError) "
                    "→ %s. Có thể do antivirus/khoá file Windows.",
                    max_attempts, Path(destination).name,
                )
                raise
            logger.warning(
                "atomic_write: os.replace bị khoá (lần %d/%d): %s. Chờ %.2fs rồi thử lại.",
                attempt, max_attempts, exc, delay,
            )
            gc.collect()  # thử nhả lại FD trước mỗi lần retry
            time.sleep(delay)
            delay *= 2.0


def atomic_write_text(
    target: Path,
    content: str,
    *,
    encoding: str = "utf-8",
    fsync_timeout_sec: float = _DEFAULT_FSYNC_TIMEOUT_SEC,
) -> None:
    """Ghi ``content`` vào ``target`` theo cơ chế atomic với fsync timeout.

    Cải tiến v3.6:
        * ``os.fsync()`` chạy trên thread riêng với timeout ``fsync_timeout_sec``
          → không bao giờ block caller indefinitely.
        * Nếu fsync timeout/lỗi → ``WARNING`` (best-effort durability).
          File vẫn được ghi hoàn toàn vào OS buffer (``flush()`` đã chạy).
        * Log chi tiết từng bước với thời gian.
        * Env flag ``SUBTITLES_EXTRACTOR_SKIP_FSYNC=1`` để bỏ qua fsync
          hoàn toàn (hữu ích cho testing/CI).

    Args:
        target:            Đường dẫn tệp đích.
        content:           Nội dung văn bản.
        encoding:          Mã hoá ký tự (mặc định ``"utf-8"``).
        fsync_timeout_sec: Timeout tối đa cho ``os.fsync()``.
                           Mặc định :data:`_DEFAULT_FSYNC_TIMEOUT_SEC` (5s).
                           Đặt ``0`` để bỏ qua fsync hoàn toàn.

    Raises:
        OSError: Khi không tạo được tệp tạm hoặc ``os.replace`` thất bại.
                 Tệp tạm được dọn dẹp tự động trước khi raise.
    """
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    _content_size_kb = len(content.encode(encoding, errors="replace")) // 1024
    logger.debug(
        "atomic_write: bắt đầu → %s  (~%d KB, platform=%s, fsync_timeout=%.1fs)",
        target.name,
        _content_size_kb,
        sys.platform,
        fsync_timeout_sec,
    )
    _t0 = time.perf_counter()

    fd, tmp_path = tempfile.mkstemp(
        prefix=target.stem + "_",
        suffix=target.suffix + ".tmp",
        dir=str(target.parent),
    )
    logger.debug("atomic_write: tạo tệp tạm → %s", Path(tmp_path).name)

    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(content)
            fh.flush()
            logger.debug(
                "atomic_write: write+flush xong (%.3fs)",
                time.perf_counter() - _t0,
            )

            # ── fsync: best-effort, không raise nếu timeout/fail ──────────
            _should_fsync = (
                not _SKIP_FSYNC
                and fsync_timeout_sec > 0
            )
            if _should_fsync:
                try:
                    _fsync_ok = _fsync_with_timeout(
                        fh.fileno(), timeout_sec=fsync_timeout_sec
                    )
                    if _fsync_ok:
                        logger.debug(
                            "atomic_write: fsync() OK (%.3fs)",
                            time.perf_counter() - _t0,
                        )
                    else:
                        # Timeout — data đã ở OS buffer, ghi vẫn sẽ hoàn tất
                        # trừ khi mất điện đúng thời điểm này (cực hiếm với SRT nhỏ).
                        logger.warning(
                            "atomic_write: fsync() TIMEOUT sau %.1fs trên '%s'. "
                            "Tiếp tục best-effort — data đã flush vào OS buffer. "
                            "Nguyên nhân thường gặp: ổ chậm, NAS, USB, antivirus scan. "
                            "(platform=%s)",
                            fsync_timeout_sec,
                            target.name,
                            sys.platform,
                        )
                except OSError as fsync_exc:
                    # fsync lỗi hệ thống (EROFS, EINVAL, …) — không ảnh hưởng
                    # đến tính toàn vẹn file vì flush() đã xong.
                    logger.warning(
                        "atomic_write: fsync() lỗi OS (%s) trên '%s' — bỏ qua, "
                        "data đã flush vào OS buffer.",
                        fsync_exc,
                        target.name,
                    )
            else:
                logger.debug("atomic_write: fsync() bỏ qua (skip_fsync=%s, timeout=%.1f)", _SKIP_FSYNC, fsync_timeout_sec)

        # ── atomic rename: temp → target ──────────────────────────────────
        # Đây là bước quan trọng nhất: nếu thành công, file đích sẽ luôn
        # hoàn chỉnh (không bao giờ partial). os.replace là atomic trên POSIX
        # và best-effort trên Windows (NTFS hỗ trợ atomic rename).
        _replace_with_retry(tmp_path, str(target))
        logger.debug(
            "atomic_write: hoàn tất → %s (tổng %.3fs, %d KB)",
            target.name,
            time.perf_counter() - _t0,
            _content_size_kb,
        )

    except OSError:
        # Dọn dẹp tệp tạm khi gặp lỗi I/O.
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


__all__ = ["atomic_write_text"]
