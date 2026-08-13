"""Điểm khởi chạy ứng dụng GUI.

Quy trình:
    1. Inject ``src/`` vào ``sys.path`` để chạy mà không cần ``pip install``.
    2. Tạo :class:`QApplication` TRƯỚC khi import fluent_compat/MainWindow
       (fluent_compat/Qt tự khởi tạo widget toàn cục lúc import → cần
       QApplication đã sẵn sàng).
    3. Cài Qt message handler để redirect Qt C++ warnings vào Python logging
       thay vì xuất ra stderr (giúp dễ debug và phân loại severity).
    4. Bootstrap container.
    5. Hiện :class:`MainWindow`.
    6. Trả mã thoát của Qt.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

# CuPy/CUDA in UserWarning "CUDA path could not be detected" khi máy không cấu
# hình CUDA — vô hại (ứng dụng tự fallback CPU). Chặn sớm để console sạch.
warnings.filterwarnings(
    "ignore", message=r".*CUDA path could not be detected.*", category=UserWarning
)

logger = logging.getLogger(__name__)

# ── Map QtMsgType → Python log level ─────────────────────────────────────
_QT_MSG_LEVEL_MAP: dict[int, int] = {}  # Điền sau khi import PyQt6


def _qt_message_handler(msg_type: int, context: object, message: str) -> None:
    """Chuyển hướng Qt C++ warnings vào Python ``logging`` thay vì stderr.

    Ưu điểm:
        * Logs xuất hiện trong log file (``logs/app.log``) với timestamp.
        * Severity đúng chuẩn: QtCriticalMsg → ERROR, QtWarningMsg → WARNING.
        * Dễ phân tích hơn so với đọc stderr raw.

    Known Qt warnings được hạ xuống DEBUG để tránh spam:
        * ``QFont::setPointSize: Point size <= 0`` — thường từ widget Qt
          khi theme apply pixel-size font; không ảnh hưởng chức năng.
        * ``QWidgetWindow * must be a top level window`` — đã được sửa từ
          nguồn (ScrollArea parent), log lại để xác nhận đã hết.
    """
    try:
        from PySide6.QtCore import QtMsgType
        level_map = {
            QtMsgType.QtDebugMsg:    logging.DEBUG,
            QtMsgType.QtInfoMsg:     logging.INFO,
            QtMsgType.QtWarningMsg:  logging.WARNING,
            QtMsgType.QtCriticalMsg: logging.ERROR,
            QtMsgType.QtFatalMsg:    logging.CRITICAL,
        }
        level = level_map.get(msg_type, logging.WARNING)
    except (ImportError, AttributeError):
        level = logging.WARNING

    # Hạ severity cho các Qt internal warnings đã biết — không cần làm ô nhiễm log.
    _KNOWN_NOISY_PATTERNS: tuple[str, ...] = (
        "QFont::setPointSize: Point size <= 0",
        "must be a top level window",
        "QObject::connect: Cannot queue arguments",
        "libpng warning",
        "iCCP: known incorrect sRGB profile",
    )
    msg_stripped = message.strip() if message else ""
    if any(pattern in msg_stripped for pattern in _KNOWN_NOISY_PATTERNS):
        level = logging.DEBUG

    logger.log(level, "[Qt] %s", msg_stripped)


def _setup_python_path() -> None:
    """Thêm ``src/`` vào ``sys.path`` — bền vững trên Windows/Linux/macOS.

    Vấn đề Windows: so sánh string path có thể thất bại do case/separator khác
    nhau (``C:/src`` vs ``C:\\src``). Giải pháp: so sánh bằng ``Path.resolve()``
    thay vì string thuần.

    Quan trọng: gọi hàm này TRƯỚC mọi import của ``subtitles_extractor`` để
    đảm bảo package luôn tìm thấy được, kể cả khi môi trường thay đổi sau
    khi Qt platform plugin khởi tạo.
    """
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"

    # Chuẩn hoá tất cả path trong sys.path để so sánh đúng trên Windows
    resolved_paths = set()
    for p in sys.path:
        try:
            resolved_paths.add(Path(p).resolve())
        except (OSError, ValueError):
            resolved_paths.add(Path(p))

    if src_dir not in resolved_paths:
        sys.path.insert(0, str(src_dir))

    # Đảm bảo package subtitles_extractor tìm thấy được ngay lập tức
    # bằng cách pre-load __init__ nếu cần (tránh Windows import hook issue)
    try:
        import importlib
        importlib.import_module("subtitles_extractor")
    except ImportError:
        pass  # Sẽ báo lỗi rõ ràng hơn khi import thực sự xảy ra


def _ensure_std_streams() -> None:
    """Gắn stdout/stderr giả khi chúng là ``None`` (đóng gói console=False).

    [v3.23.278] PyInstaller với ``console=False`` đặt ``sys.stdout``/``sys.stderr`` là
    ``None``. Thư viện bên thứ ba (paddle, tqdm, ...) gọi ``print()`` sẽ crash
    ``AttributeError: 'NoneType' object has no attribute 'write'``. Gắn stream giả nuốt
    output để mọi print đi vào hư không thay vì crash. Log thật của app đi qua loguru
    (file sink) nên không mất gì.
    """
    import io

    class _NullStream(io.TextIOBase):
        def write(self, _s: str) -> int:
            return len(_s)

        def flush(self) -> None:
            return

    if sys.stdout is None:
        sys.stdout = _NullStream()
    if sys.stderr is None:
        sys.stderr = _NullStream()


def main() -> int:
    """Chạy ứng dụng và trả về exit code của Qt."""
    _ensure_std_streams()
    _setup_python_path()

    # [v3.23.301] KHO MODEL TẬP TRUNG: trỏ PaddleOCR (models/paddle) và HuggingFace
    # (models/huggingface — VieNeu-TTS, fastembed) về thư mục models/ nếu đã prefetch.
    # PHẢI chạy TRƯỚC mọi import kéo paddle/paddlex/huggingface_hub vì các thư viện
    # này đọc biến môi trường cache ngay lúc import. Chạy nguồn lẫn đóng gói đều dùng
    # được. Xem infrastructure/model_store.py.
    from subtitles_extractor.infrastructure.model_store import (
        configure_all_model_stores,
    )

    configure_all_model_stores()

    # [v3.23.271] Cài shim chardet (charset-normalizer MIT thay chardet LGPL) TRƯỚC
    # khi import nào kéo paddle/paddlex — bản đóng gói thương mại không nhúng chardet
    # LGPL. Xem docs/LICENSE_ANALYSIS.md.
    from subtitles_extractor.infrastructure.compat import install_chardet_shim

    install_chardet_shim()

    # QApplication TRƯỚC khi import bất cứ thứ gì liên quan đến giao diện.
    from PySide6.QtCore import qInstallMessageHandler
    from PySide6.QtWidgets import QApplication

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("SubtitlesExtractor")
    qt_app.setOrganizationName("SubtitlesExtractor")

    # [v3.23.370] Icon ứng dụng + AppUserModelID — PHẢI đặt TRƯỚC khi tạo cửa sổ để cả
    # thanh tiêu đề LẪN thanh tác vụ Windows hiện đúng icon (nếu không Windows gộp theo
    # icon của python/pythonw.exe). Dùng _resolve_data_dir() để tìm app.ico đáng tin cả
    # khi chạy nguồn lẫn bản đóng gói PyInstaller.
    try:
        from subtitles_extractor.composition.bootstrap import _resolve_data_dir

        _icon_path = _resolve_data_dir() / "app.ico"
        if _icon_path.is_file():
            from PySide6.QtGui import QIcon

            qt_app.setWindowIcon(QIcon(str(_icon_path)))
        if sys.platform == "win32":
            import ctypes

            # AppUserModelID riêng → Windows tách khỏi python.exe và dùng icon của app.
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "SubtitleStudio.App"
            )
    except (ImportError, OSError, AttributeError) as icon_exc:
        logging.getLogger(__name__).warning("Không đặt được icon app: %s", icon_exc)

    # Cài message handler ngay sau khi QApplication tồn tại — TRƯỚC bootstrap.
    qInstallMessageHandler(_qt_message_handler)

    # Lazy import sau khi QApplication tồn tại.
    from subtitles_extractor.composition.bootstrap import bootstrap_for_gui
    from subtitles_extractor.presentation.main_window import MainWindow

    container = bootstrap_for_gui()
    try:
        window = MainWindow(container)
        window.show()
        exit_code = qt_app.exec()
    except (ImportError, RuntimeError, OSError):
        logger.exception("Ứng dụng kết thúc do lỗi không bắt được.")
        try:
            container.shutdown()
        except (RuntimeError, OSError) as shutdown_exc:
            logger.warning("Không shutdown sạch container: %s.", shutdown_exc)
        return 1
    return exit_code


if __name__ == "__main__":
    # [Windows] freeze_support() bắt buộc khi đóng gói thành .exe (PyInstaller,
    # cx_Freeze) để multiprocessing.spawn hoạt động đúng. Gọi sớm nhất có thể.
    import multiprocessing as _mp
    _mp.freeze_support()
    # [Windows] Set torch multiprocessing sharing strategy trước khi import torch
    # để tránh shm.dll WinError 127 (Unix shared memory không tồn tại trên Windows).
    import os as _os
    if _os.name == "nt":
        _os.environ.setdefault("PYTORCH_DISABLE_SHAREDMEM", "1")

    sys.exit(main())
