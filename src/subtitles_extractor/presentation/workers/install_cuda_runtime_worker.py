"""[v3.23.375] Worker TẢI CUDA runtime cho GPU OCR trên QThread.

Chạy ``pip install --target models/cuda_runtime nvidia-*-cu12`` bằng Python HỆ THỐNG (bản
đóng gói không có pip bên trong). Sau khi xong, người dùng khởi động lại app để bootstrap
nạp CUDA từ thư mục vừa tải và paddle dùng được GPU.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.infrastructure.ocr.cuda_runtime_plan import (
    CUDA_RUNTIME_PACKAGES,
    build_cuda_install_command,
)
from subtitles_extractor.infrastructure.process.hidden_process import no_window_kwargs

logger = logging.getLogger(__name__)


class InstallCudaRuntimeWorker(QObject):
    """Tải CUDA runtime (~2.3GB) vào ``cuda_dir`` để bật GPU OCR.

    Signals:
        progress: ``(phần trăm 0–100, mô tả bước)``.
        finished: Đường dẫn thư mục CUDA đã tải.
        failed: Thông điệp lỗi thân thiện.
        done: Luôn phát ở cuối để dọn luồng.
    """

    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)
    done = Signal()

    def __init__(self, python_exe: str, cuda_dir: Path) -> None:
        super().__init__()
        self._python_exe = python_exe
        self._cuda_dir = cuda_dir
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        """Thực hiện tải. Gọi trong QThread, KHÔNG gọi trực tiếp."""
        try:
            self._cuda_dir.mkdir(parents=True, exist_ok=True)
            command = build_cuda_install_command(self._python_exe, self._cuda_dir)
            self.progress.emit(
                5, f"Đang tải {len(CUDA_RUNTIME_PACKAGES)} gói CUDA (~2.3GB)…"
            )
            logger.info("Tải CUDA runtime: %s", " ".join(command))
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
                self.failed.emit("Đã huỷ tải CUDA runtime theo yêu cầu.")
                return
            if result.returncode != 0:
                tail = (result.stderr or result.stdout or "").strip()
                self.failed.emit(
                    "Tải CUDA runtime thất bại:\n"
                    + "\n".join(tail.splitlines()[-4:])
                )
                return
            self.progress.emit(100, "Hoàn tất.")
            self.finished.emit(self._cuda_dir)
        except subprocess.TimeoutExpired:
            self.failed.emit("Tải CUDA runtime quá thời gian cho phép (mạng chậm?).")
        except (OSError, ValueError) as exc:
            logger.exception("Lỗi hệ thống khi tải CUDA runtime.")
            self.failed.emit(f"Lỗi hệ thống: {exc}")
        finally:
            self.done.emit()


__all__ = ["InstallCudaRuntimeWorker"]
