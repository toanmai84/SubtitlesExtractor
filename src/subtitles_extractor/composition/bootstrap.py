"""Bootstrap — khởi tạo :class:`ApplicationContainer` từ môi trường thực."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.infrastructure.logging.setup import setup_logging
from subtitles_extractor.infrastructure.settings.json_file_repository import (
    JsonFileRepository,
)

logger = logging.getLogger(__name__)

def _resolve_data_dir() -> Path:
    # [v3.23.265] Hỗ trợ chạy đóng gói (PyInstaller): khi frozen, data files nằm trong
    # thư mục giải nén ``sys._MEIPASS`` theo cấu trúc ``subtitles_extractor/data``. Khi chạy
    # từ source, dùng đường dẫn tương đối như cũ (``__file__`` trỏ vào src/).
    import sys

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "subtitles_extractor" / "data"
    return Path(__file__).resolve().parent.parent / "data"

def _resolve_user_data_dir(app_name: str = "SubtitlesExtractor") -> Path:
    import os
    import sys
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    return base / app_name

def _apply_paddle_env_vars(settings) -> None:
    if settings.advanced.disable_paddle_network_check:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        logger.debug("Đã thiết lập biến môi trường tắt Paddle Network Check.")
    else:
        os.environ.pop("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", None)
        logger.debug("Đã gỡ biến môi trường tắt Paddle Network Check.")

def _prepare_process_for_paddle() -> None:
    """[v3.23.394] Chuẩn bị tiến trình để PaddleOCR chạy được — DÙNG CHUNG cho GUI lẫn CLI.

    Gộp 3 việc phải làm TRƯỚC khi bất kỳ đâu import paddle, để hai đường khởi động (GUI/CLI)
    không lệch nhau (trước đây ``bootstrap_for_cli`` thiếu các bước này → OCR sẽ hỏng nếu chạy
    headless):

    1. Thêm lõi paddle TẢI-LÚC-CHẠY (``models/paddle_runtime``) vào ``sys.path`` — bản build nhỏ
       không nhúng paddlepaddle-gpu. Phải chạy TRƯỚC mọi thao tác dò/import paddle (kể cả
       ``_prioritize_paddle_cudnn`` dùng ``find_spec``).
    2. (Windows) Ưu tiên cuDNN của paddle để tránh nạp nhầm cuDNN của torch CUDA (WinError 127).
    3. Chặn import torch ở tiến trình chính — PaddleX/transformers tự dò & import torch nếu thấy
       → torch chiếm cuDNN → paddle lỗi. WhisperX chạy ở tiến trình con (không dính hook này).
    """
    import os as _os

    _ensure_paddle_runtime_on_syspath()

    if _os.name == "nt":
        _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        _prioritize_paddle_cudnn()

    from subtitles_extractor.infrastructure.torch_import_blocker import (
        install_torch_import_blocker,
    )

    install_torch_import_blocker()


def bootstrap_for_gui(*, organization: str = "SubtitlesExtractor", application_name: str = "PaddleOCR", log_level: int = logging.INFO) -> ApplicationContainer:
    user_dir = _resolve_user_data_dir(application_name)
    setup_logging(level=log_level, log_dir=user_dir / "logs")
    _inject_mpv_dll_path_if_available(user_dir)

    # [v3.23] KHÔNG import torch ở tiến trình chính nữa. WhisperX chạy ở TIẾN TRÌNH
    # CON riêng (process isolation), nên không cần set torch sharing strategy ở đây.
    # Quan trọng: import torch vào process chính sẽ nạp cuDNN của torch → khiến
    # PaddleOCR lỗi "WinError 127 cudnn_engines_precompiled64_9.dll" khi cài torch
    # CUDA. Giữ process chính SẠCH torch để paddle nạp cuDNN của riêng nó.
    _prepare_process_for_paddle()

    from subtitles_extractor.infrastructure.settings.qsettings_repository import (
        QSettingsRepository,
    )
    repository = QSettingsRepository(organization=organization, application=application_name)
    container = ApplicationContainer(
        settings_repository=repository, i18n_data_dir=_resolve_data_dir(), user_data_dir=user_dir,
    )

    _apply_saved_ui_settings(container)
    _apply_paddle_env_vars(container.settings_service.current)

    # Gắn cầu nối log → trang Nhật ký (hiển thị realtime trong ứng dụng).
    try:
        from loguru import logger as _loguru_logger

        _loguru_logger.add(container.log_bridge.loguru_sink, level="DEBUG", enqueue=True)
        logger.debug("Đã gắn cầu nối log Loguru → trang Nhật ký.")
    except (ImportError, ValueError) as exc:
        logger.warning("Không gắn được cầu nối log cho trang Nhật ký: %s.", exc)

    logger.info("Bootstrap GUI hoàn tất — user_dir=%s, data_dir=%s.", user_dir, _resolve_data_dir())
    return container

def _apply_saved_ui_settings(container: ApplicationContainer) -> None:
    """Áp dụng cấu hình UI đã lưu (font size, theme). Lỗi phụ → log + bỏ qua."""
    try:
        settings = container.settings_service.current
        level_str = settings.advanced.log_level
        numeric = getattr(logging, level_str.upper(), logging.INFO)
        logging.getLogger().setLevel(numeric)

        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            font = app.font()
            size_pt = settings.ui.safe_font_size
            # Guard: chỉ set khi là số nguyên dương hợp lệ.
            # settings.ui.font_size có thể là None, 0, -1 (chưa cấu hình).
            if isinstance(size_pt, int) and size_pt > 0 and font.pointSize() != size_pt:
                clamped = max(6, min(24, size_pt))
                font.setPointSize(clamped)
                app.setFont(font)

        # Theme từ QFluentWidgets (optional dependency).
        try:
            from subtitles_extractor.presentation.fluent_compat import Theme, setTheme

            theme_str = settings.ui.theme
            if theme_str == "dark":
                setTheme(Theme.DARK)
            elif theme_str == "light":
                setTheme(Theme.LIGHT)
            else:
                setTheme(Theme.AUTO)
        except ImportError:
            logger.debug("qfluentwidgets không khả dụng — bỏ qua theme.")

    except (AttributeError, ImportError, OSError) as exc:
        logger.warning("Không áp dụng được UI settings lúc startup: %s.", exc)

def bootstrap_for_cli(*, config_file: Path | None = None, log_level: int = logging.INFO) -> ApplicationContainer:
    user_dir = _resolve_user_data_dir()
    setup_logging(level=log_level, log_dir=user_dir / "logs")
    _inject_mpv_dll_path_if_available(user_dir)
    # [v3.23.394] Chuẩn bị paddle GIỐNG bootstrap_for_gui (sys.path lõi paddle tải-lúc-chạy +
    # cuDNN + chặn torch) — nếu không, đường CLI/headless sẽ không import được paddle.
    _prepare_process_for_paddle()
    config_path = config_file or (user_dir / "config.json")
    repository = JsonFileRepository(file_path=config_path)
    container = ApplicationContainer(
        settings_repository=repository, i18n_data_dir=_resolve_data_dir(), user_data_dir=user_dir,
    )
    _apply_paddle_env_vars(container.settings_service.current)
    logger.info("Bootstrap CLI hoàn tất — config=%s, data_dir=%s.", config_path, _resolve_data_dir())
    return container

def _inject_mpv_dll_path_if_available(user_dir: Path) -> None:
    """Bơm thư mục libmpv DLL vào ``PATH`` (Windows). Bỏ qua nếu có lỗi."""
    try:
        from subtitles_extractor.infrastructure.video.mpv_dll_manager import (
            MpvDllManager,
        )

        manager = MpvDllManager(app_data_dir=user_dir)
        manager.ensure_available()
    except (ImportError, OSError, RuntimeError) as exc:
        logger.warning("Kiểm tra libmpv DLL thất bại: %s.", exc)


def _ensure_paddle_runtime_on_syspath() -> None:
    """Thêm lõi ``paddlepaddle-gpu`` TẢI-LÚC-CHẠY vào CUỐI ``sys.path`` (nếu có).

    Bản build nhỏ (one-file / minimal) KHÔNG nhúng lõi paddle (~810MB). Người dùng tải riêng
    vào ``models/paddle_runtime/`` (kho runtime CỐ ĐỊNH cạnh exe). Hàm này thêm thư mục đó vào
    CUỐI ``sys.path`` để ``import paddle`` thấy lõi vừa tải, trong khi các phụ thuộc dùng chung
    (numpy, opencv…) vẫn ưu tiên bản NHÚNG đã kiểm thử.

    An toàn: nếu paddle đã nhúng sẵn (bản đầy đủ) hoặc chưa tải, hàm không làm gì (không thêm
    đường dẫn vô ích, không ghi đè). Chỉ thêm khi thư mục lõi paddle THỰC SỰ tồn tại.
    """
    import sys

    try:
        from subtitles_extractor.infrastructure.model_store import (
            ensure_model_store_root,
            model_store_root,
        )
        from subtitles_extractor.infrastructure.ocr.paddle_runtime_plan import (
            PADDLE_RUNTIME_DIRNAME,
            paddle_runtime_sys_path,
        )
    except ImportError:
        return

    seen: set[str] = set()
    for root in (model_store_root(), ensure_model_store_root()):
        if root is None:
            continue
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        for path_dir in paddle_runtime_sys_path(root / PADDLE_RUNTIME_DIRNAME):
            path_str = str(path_dir)
            if path_str not in sys.path:
                sys.path.append(path_str)  # CUỐI: bản nhúng thắng cho deps dùng chung
                logger.info("Đã thêm lõi paddle tải-lúc-chạy vào sys.path: %s", path_str)


def _prioritize_paddle_cudnn() -> None:
    """[v3.23] Đảm bảo PaddlePaddle nạp ĐÚNG cuDNN/cuBLAS của nó, không nhầm torch.

    Khi cài torch CUDA cùng môi trường, paddle có thể nạp nhầm cuDNN từ ``torch\\lib``
    (sai phiên bản) → ``WinError 127``. Hai biện pháp (Windows):
      1. Thêm thư mục cuDNN/cuBLAS RIÊNG của paddle vào ĐẦU DLL search path.
      2. LOẠI ``torch\\lib`` khỏi biến PATH của tiến trình chính, để paddle không
         "nhìn thấy" cuDNN của torch. (WhisperX chạy ở tiến trình con với PATH riêng,
         không ảnh hưởng.)
    Bỏ qua mọi lỗi (no-op nếu không phải Windows / không có paddle).
    """
    import os

    if os.name != "nt":
        return
    try:
        import importlib.util

        # (2) Loại torch\lib khỏi PATH của process chính.
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is not None and torch_spec.submodule_search_locations:
            torch_lib = str(Path(list(torch_spec.submodule_search_locations)[0]) / "lib")
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            filtered = [p for p in path_entries if os.path.normcase(p.rstrip("\\/")) !=
                        os.path.normcase(torch_lib.rstrip("\\/"))]
            if len(filtered) != len(path_entries):
                os.environ["PATH"] = os.pathsep.join(filtered)
                logger.debug("Đã loại torch\\lib khỏi PATH (tránh paddle nạp nhầm cuDNN).")

        # (1) Ưu tiên cuDNN/cuBLAS của paddle.
        spec = importlib.util.find_spec("paddle")
        if spec is None or not spec.submodule_search_locations:
            return
        paddle_root = Path(list(spec.submodule_search_locations)[0])
        candidate_dirs = [
            paddle_root / "libs",
            paddle_root.parent / "nvidia" / "cudnn" / "bin",
            paddle_root.parent / "nvidia" / "cublas" / "bin",
        ]

        # [v3.23.375] CUDA runtime TẢI LÚC CHẠY (bản build nhỏ không nhúng CUDA): thêm
        # models/cuda_runtime/nvidia/<component>/bin nếu người dùng đã bấm "Bật tăng tốc GPU".
        try:
            from subtitles_extractor.infrastructure.model_store import model_store_root
            from subtitles_extractor.infrastructure.ocr.cuda_runtime_plan import (
                CUDA_RUNTIME_DIRNAME,
                cuda_runtime_dll_dirs,
            )

            store = model_store_root()
            # [v3.23.384] Tìm CUDA runtime ở CẢ hai nơi (dedup): (1) model_store_root()
            # — có thể là model NHÚNG ở _MEIPASS với bản onedir; (2) kho runtime CỐ ĐỊNH
            # cạnh exe — nơi tính năng "Bật tăng tốc GPU" TẢI CUDA về (bền cho cả one-file).
            from subtitles_extractor.infrastructure.model_store import (
                ensure_model_store_root,
            )

            runtime_root = ensure_model_store_root()
            seen_roots: set[str] = set()
            for root in (store, runtime_root):
                if root is None:
                    continue
                root_key = str(root)
                if root_key in seen_roots:
                    continue
                seen_roots.add(root_key)
                candidate_dirs.extend(
                    cuda_runtime_dll_dirs(root / CUDA_RUNTIME_DIRNAME)
                )
        except (ImportError, OSError, ValueError):
            pass

        for dll_dir in candidate_dirs:
            if dll_dir.is_dir():
                os.add_dll_directory(str(dll_dir))
                os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
        logger.debug("Đã ưu tiên cuDNN/cuBLAS của paddle trong DLL search path.")
    except (ImportError, OSError, ValueError) as exc:
        logger.debug("Không cấu hình được cuDNN cho paddle (%s) — bỏ qua.", exc)

__all__ = ["bootstrap_for_cli", "bootstrap_for_gui"]
