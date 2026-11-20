"""[v3.23.755] Worker TẢI TensorRT (~1,6 GB) cho OCR GPU trên QThread — cùng mẫu tải CUDA runtime.

``pip install --target models/tensorrt_runtime tensorrt-cu12-libs tensorrt-cu12-bindings`` (kho NVIDIA) bằng Python
nhúng / hệ thống, rồi ghi mô-đun ``tensorrt``. Xong thì khởi động lại ứng dụng để bootstrap đưa TensorRT vào sys.path +
đường dẫn DLL.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.infrastructure.ocr.tensorrt_runtime_plan import (
    TENSORRT_DOWNLOAD_GB,
    TENSORRT_VERSION,
    build_tensorrt_install_command,
    is_tensorrt_installed,
    write_tensorrt_shim,
)
from subtitles_extractor.infrastructure.process.hidden_process import no_window_kwargs

logger = logging.getLogger(__name__)


class InstallTensorRtRuntimeWorker(QObject):
    """Signals như ``InstallCudaRuntimeWorker``: progress(%, mô tả), finished(thư mục), failed(lỗi), done()."""

    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)
    done = Signal()

    def __init__(self, python_exe: str, trt_dir: Path) -> None:
        super().__init__()
        self._python_exe = python_exe
        self._trt_dir = trt_dir

    def run(self) -> None:
        """Thực hiện tải. Gọi trong QThread, KHÔNG gọi trực tiếp."""
        try:
            self._trt_dir.mkdir(parents=True, exist_ok=True)
            command = build_tensorrt_install_command(self._python_exe, self._trt_dir)
            self.progress.emit(5, f"Đang tải TensorRT {TENSORRT_VERSION} (~{TENSORRT_DOWNLOAD_GB:.1f} GB)…")
            logger.info("Tải TensorRT: %s", " ".join(command))
            result = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600,
                check=False, **no_window_kwargs(),
            )
            if result.returncode != 0:
                tail = (result.stderr or result.stdout or "").strip()
                logger.error("Tải TensorRT thất bại (mã thoát %s). Đuôi log:\n%s", result.returncode, tail[-4000:])
                self.failed.emit("Tải TensorRT thất bại:\n" + "\n".join(tail.splitlines()[-4:]))
                return
            write_tensorrt_shim(self._trt_dir)
            if not is_tensorrt_installed(self._trt_dir):
                self.failed.emit("Tải xong nhưng thiếu tệp TensorRT (nvinfer_10.dll / tensorrt_bindings).")
                return
            self.progress.emit(100, "Hoàn tất.")
            self.finished.emit(self._trt_dir)
        except subprocess.TimeoutExpired:
            logger.error("Tải TensorRT quá thời gian cho phép (giới hạn 3600 s).")
            self.failed.emit("Tải TensorRT quá thời gian cho phép (mạng chậm?).")
        except (OSError, ValueError) as exc:
            logger.exception("Lỗi hệ thống khi tải TensorRT.")
            self.failed.emit(f"Lỗi hệ thống: {exc}")
        finally:
            self.done.emit()


__all__ = ["InstallTensorRtRuntimeWorker"]
