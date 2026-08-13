"""Cấu hình Loguru tập trung — đỉnh cao Plug & Play (v2.29).

Module này thay thế hoàn toàn Python ``logging`` mặc định bằng Loguru, đồng
thời cài ``InterceptHandler`` để **bắt và chuyển hướng** mọi log call qua
stdlib ``logging`` (từ 60+ modules hiện có, từ thư viện 3rd-party) sang
pipeline Loguru. Lợi ích:

    * **Plug & Play**: import 1 lần, dùng được ngay (không setup handler /
      formatter rườm rà).
    * **Tô màu tự động**: console output có màu theo level (DEBUG xám, INFO
      trắng, WARNING vàng, ERROR đỏ, CRITICAL nền đỏ).
    * **Thread-safe**: an toàn khi chạy đa luồng (workers Qt, OCR).
    * **Format đẹp**: bao gồm timestamp ms, level, file:line, message.
    * **File rotation**: tự động xoay file 5MB, giữ 3 backup, nén gz.
    * **Backward compatible**: 60+ modules cũ dùng ``logging.getLogger`` vẫn
      work qua InterceptHandler — không cần migrate hết.

Cách dùng:
    Tại bootstrap:
        >>> from subtitles_extractor.infrastructure.logging.loguru_config import setup_loguru
        >>> setup_loguru(level="INFO", log_dir=Path("./logs"))

    Tại module mới (preferred):
        >>> from loguru import logger
        >>> logger.info("Subtitle built: {} events", len(events))

    Tại module cũ (vẫn work qua bridge):
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Done")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

from loguru import logger

# ============================================================================
# Format strings
# ============================================================================
# Console: ngắn gọn, có màu. Loguru tự động colorize qua {level: ...} tags.
_CONSOLE_FORMAT_TEMPLATE: Final[str] = (
    "<green>{time:HH:mm:ss.SSS}</green> "
    "<level>{level: <7}</level> "
    "<cyan>{name}</cyan>: "
    "<level>{message}</level>"
)

# File: đầy đủ, không màu, có file:line cho debug trace-back.
_FILE_FORMAT_TEMPLATE: Final[str] = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <7} | "
    "{name}:{function}:{line} | "
    "{message}"
)


class InterceptHandler(logging.Handler):
    """Bridge: chuyển log từ stdlib ``logging`` sang Loguru pipeline.

    Cài đặt root logger handler là instance này → mọi ``logger.info(...)`` ở
    các modules cũ (kể cả thư viện 3rd-party) đều được redirect sang Loguru.

    Lý do tồn tại:
        - App có 60+ modules dùng ``logging.getLogger(__name__)``.
        - Migrate hết sang ``from loguru import logger`` tốn công và rủi ro
          (loguru dùng ``{}`` format, stdlib dùng ``%s`` — không tương thích
          trực tiếp).
        - InterceptHandler bắt log ở root, format message bằng stdlib trước
          khi đẩy sang Loguru — đảm bảo tương thích 100%.

    Tham khảo: https://loguru.readthedocs.io/en/stable/overview.html
    """

    # Logger 3rd-party ồn ào: log DEBUG kèm exception/traceback nội bộ (vd torio
    # thử nạp FFmpeg extension thất bại rồi tự fallback) — KHÔNG forward traceback
    # sang Loguru để tránh "Logging error in Handler" khi format traceback C-ext.
    _NOISY_DEBUG_PREFIXES = ("torio", "torchaudio", "torch", "numba", "speechbrain")

    def emit(self, record: logging.LogRecord) -> None:
        # Handler logging KHÔNG ĐƯỢC phép ném — bọc toàn bộ để mọi lỗi bridge
        # không bao giờ làm gián đoạn luồng nghiệp vụ (vd phiên âm WhisperX).
        try:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            # Bỏ traceback cho DEBUG nội bộ của thư viện ồn ào (chỉ giữ message).
            exc_info = record.exc_info
            if exc_info is not None and record.levelno <= logging.DEBUG:
                if record.name.split(".")[0] in self._NOISY_DEBUG_PREFIXES:
                    exc_info = None

            logger.opt(depth=depth, exception=exc_info).log(
                level, record.getMessage()
            )
        except Exception:
            pass


def setup_loguru(
    *,
    level: str = "INFO",
    log_dir: Path | None = None,
    file_rotation: str = "5 MB",
    file_retention: int = 3,
    file_compression: str = "gz",
    enable_diagnose: bool = False,
) -> None:
    """Cấu hình Loguru cho toàn ứng dụng — gọi 1 lần duy nhất ở bootstrap.

    Args:
        level: Cấp tối thiểu (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ...).
        log_dir: Thư mục lưu file log xoay vòng. Nếu ``None`` → chỉ log
            console.
        file_rotation: Kích thước hoặc thời gian xoay file. Mặc định
            ``"5 MB"``. Có thể dùng ``"00:00"`` (xoay nửa đêm) hoặc
            ``"1 week"``.
        file_retention: Số file backup giữ lại sau khi xoay (mặc định 3).
        file_compression: Loại nén cho file backup (``"gz"``, ``"zip"``, ...
            hoặc ``None`` để không nén).
        enable_diagnose: Bật ``diagnose=True`` của Loguru — hiển thị giá trị
            biến trong exception traceback. Cảnh báo: có thể leak data nhạy
            cảm; chỉ bật khi debug local.

    Tác dụng phụ:
        - Xoá toàn bộ handler hiện tại của root stdlib logging.
        - Cài ``InterceptHandler`` lên root để bridge stdlib → Loguru.
        - Hạ level các thư viện 3rd-party ồn ào (PIL, matplotlib, urllib3,
          paddle, ppocr) xuống ``WARNING``.
    """
    # ── Bước 1: Reset Loguru handlers (tránh tích tụ khi gọi lặp).
    logger.remove()

    # ── Bước 2: Console sink với màu sắc theo level.
    # [v3.23.278] Khi đóng gói PyInstaller với console=False (GUI không cửa sổ console),
    # ``sys.stderr``/``sys.stdout`` là ``None`` → loguru.add(None) báo TypeError. Chỉ thêm
    # console sink khi thực sự có stderr; nếu không, chỉ ghi file (bước 3).
    if sys.stderr is not None:
        logger.add(
            sys.stderr,
            format=_CONSOLE_FORMAT_TEMPLATE,
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=enable_diagnose,
            enqueue=False,  # Console nên synchronous để output realtime.
        )

    # ── Bước 3: File sink (nếu có log_dir).
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "subtitles_extractor.log",
            format=_FILE_FORMAT_TEMPLATE,
            level=level,
            rotation=file_rotation,
            retention=file_retention,
            compression=file_compression,
            encoding="utf-8",
            backtrace=True,
            diagnose=enable_diagnose,
            # Thread-safe + async — không block main thread khi I/O.
            enqueue=True,
        )

    # ── Bước 4: Bridge stdlib logging → Loguru qua InterceptHandler.
    intercept_handler = InterceptHandler()
    intercept_handler.setLevel(logging.NOTSET)  # Bắt mọi level, lọc ở Loguru.

    root_logger = logging.getLogger()
    # Xoá tất cả handler cũ (tránh duplicate output).
    for existing_handler in list(root_logger.handlers):
        root_logger.removeHandler(existing_handler)
    root_logger.addHandler(intercept_handler)
    root_logger.setLevel(logging.NOTSET)

    # ── Bước 5: Hạ thấp cấp các thư viện ồn ào.
    for noisy_logger_name in (
        "PIL", "matplotlib", "urllib3", "paddle", "ppocr",
        "PaddleX", "paddlex", "PaddleOCR",
        # [v3.22.4] torio/torchaudio log DEBUG kèm traceback khi thử nạp FFmpeg
        # extension thất bại (tự fallback) → gây "Logging error in Handler" khi
        # format traceback C-ext. Chỉ giữ log từ WARNING trở lên.
        "torio", "torchaudio", "torch", "numba", "speechbrain", "whisperx",
        # [v3.23.18] httpcore/httpx phát log DEBUG dày đặc TRONG thread nền (upload
        # video, gọi API) → InterceptHandler đẩy vào loguru gây re-entrant lock
        # ("Could not acquire internal lock (deadlock avoided)"). Hạ xuống WARNING:
        # CHỈ ẩn DEBUG/INFO ồn ào, VẪN GIỮ WARNING/ERROR để debug khi có sự cố.
        # [v3.23.20] KHÔNG hạ cả "google"/"google_genai" — chúng có thể chứa cảnh báo
        # hữu ích (AFC, quota). Chỉ chặn đúng 2 logger trace gây deadlock.
        "httpcore", "httpx",
    ):
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)


# ============================================================================
# Backward-compatible API — module cũ vẫn import được ``setup_logging``.
# ============================================================================

def setup_logging(
    *,
    level: int = logging.INFO,
    log_dir: Path | None = None,
    file_max_bytes: int = 5 * 1024 * 1024,
    file_backup_count: int = 3,
) -> None:
    """[BACKWARD-COMPAT] Wrapper cho ``setup_loguru`` với API cũ.

    Module cũ và bootstrap đang dùng ``setup_logging(level=logging.INFO, ...)``.
    Wrapper này delegate sang Loguru-based setup mới — không phá vỡ
    import cũ.

    Args:
        level: Cấp logging (số nguyên stdlib, vd ``logging.INFO`` = 20).
        log_dir: Thư mục log (tương đương ``setup_loguru.log_dir``).
        file_max_bytes: KHÔNG dùng (Loguru dùng ``"5 MB"`` text). Giữ tham
            số để không phá vỡ API.
        file_backup_count: Số file backup giữ lại.
    """
    # Chuyển int level sang tên string của Loguru.
    int_to_name_map = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }
    level_name = int_to_name_map.get(level, "INFO")

    # Chuyển file_max_bytes (int) sang "N MB" hoặc "N KB" cho Loguru rotation.
    if file_max_bytes >= 1024 * 1024:
        rotation_spec = f"{file_max_bytes // (1024 * 1024)} MB"
    elif file_max_bytes >= 1024:
        rotation_spec = f"{file_max_bytes // 1024} KB"
    else:
        rotation_spec = f"{file_max_bytes} B"

    setup_loguru(
        level=level_name,
        log_dir=log_dir,
        file_rotation=rotation_spec,
        file_retention=file_backup_count,
    )


__all__ = ["InterceptHandler", "setup_logging", "setup_loguru"]
