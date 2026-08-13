"""[v3.23.386] Worker TẢI lõi ``paddlepaddle-gpu`` trên QThread.

Chạy ``pip install --target models/paddle_runtime paddlepaddle-gpu==3.3.1 --no-deps`` bằng
Python HỆ THỐNG (bản đóng gói không có pip bên trong). Sau khi xong, người dùng khởi động lại
app để bootstrap thêm thư mục vừa tải vào ``sys.path`` và OCR dùng được.

Song song mẫu ``InstallCudaRuntimeWorker`` để bảo trì nhất quán.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.infrastructure.ocr.paddle_runtime_plan import (
    PADDLE_RUNTIME_DOWNLOAD_GB,
    build_paddle_install_command,
)
from subtitles_extractor.infrastructure.process.hidden_process import no_window_kwargs

logger = logging.getLogger(__name__)


class InstallPaddleRuntimeWorker(QObject):
    """Tải lõi ``paddlepaddle-gpu`` (~810MB) vào ``paddle_dir`` để chạy OCR.

    Signals:
        progress: ``(phần trăm 0–100, mô tả bước)``.
        finished: Đường dẫn thư mục paddle đã tải.
        failed: Thông điệp lỗi thân thiện.
        done: Luôn phát ở cuối để dọn luồng.
    """

    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)
    done = Signal()

    def __init__(self, python_exe: str, paddle_dir: Path) -> None:
        super().__init__()
        self._python_exe = python_exe
        self._paddle_dir = paddle_dir
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        """Thực hiện tải. Gọi trong QThread, KHÔNG gọi trực tiếp."""
        try:
            self._paddle_dir.mkdir(parents=True, exist_ok=True)
            # Phiên bản Python BUNDLED (app đang chạy trên nó) — để pip tải đúng wheel paddle,
            # bất kể Python hệ thống chạy pip là phiên bản nào. Thiết yếu khi .exe được chia sẻ
            # sang máy có Python hệ thống khác phiên bản.
            target_py = f"{sys.version_info.major}.{sys.version_info.minor}"
            command = build_paddle_install_command(
                self._python_exe, self._paddle_dir, target_py
            )
            self.progress.emit(
                5, f"Đang tải lõi paddlepaddle-gpu (~{PADDLE_RUNTIME_DOWNLOAD_GB:.1f}GB)…"
            )
            logger.info("Tải paddle runtime: %s", " ".join(command))
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3600,
                check=False,
                **no_window_kwargs(),
            )
            if self._cancelled:
                self.failed.emit("Đã huỷ tải paddle runtime theo yêu cầu.")
                return
            if result.returncode != 0:
                tail = (result.stderr or result.stdout or "").strip()
                self.failed.emit(
                    "Tải paddle runtime thất bại:\n"
                    + "\n".join(tail.splitlines()[-4:])
                )
                return
            self.progress.emit(100, "Hoàn tất.")
            self.finished.emit(self._paddle_dir)
        except subprocess.TimeoutExpired:
            self.failed.emit("Tải paddle runtime quá thời gian cho phép (mạng chậm?).")
        except (OSError, ValueError) as exc:
            logger.exception("Lỗi hệ thống khi tải paddle runtime.")
            self.failed.emit(f"Lỗi hệ thống: {exc}")
        finally:
            self.done.emit()


__all__ = ["InstallPaddleRuntimeWorker"]
