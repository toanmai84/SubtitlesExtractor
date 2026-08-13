"""Worker cài VieNeu vào môi trường riêng để chạy TTS trên GPU.

Chạy trên luồng riêng vì thao tác cài có thể mất vài phút. Tải nhẹ hơn hẳn WhisperX
(~51 MB thay vì ~3GB) vì ``torch`` đã có sẵn trong ``whisperx_env``.
"""

from __future__ import annotations

import logging
import subprocess
import sys

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.infrastructure.tts.vieneu_gpu_plan import (
    build_gpu_tts_plan,
)

logger = logging.getLogger(__name__)


class InstallVieneuGpuWorker(QObject):
    """Cài các gói còn thiếu để VieNeu chạy được trên GPU.

    Signals:
        progress: ``(phần trăm 0–100, mô tả bước)``.
        finished: Đường dẫn Python của môi trường đã sẵn sàng.
        failed: Thông điệp lỗi thân thiện.
        done: Luôn phát ở cuối, kể cả khi lỗi.
    """

    progress = Signal(int, str)
    finished = Signal(str)
    failed = Signal(str)
    done = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False

    def cancel(self) -> None:
        """Yêu cầu dừng giữa các bước."""
        self._cancelled = True

    def run(self) -> None:
        """Thực hiện cài. Gọi trong QThread, KHÔNG gọi trực tiếp."""
        try:
            plan = build_gpu_tts_plan()
            if plan.is_ready:
                # [v3.23.346] "Đã cài" chưa chắc "chạy được": bản trước cài `vieneu`
                # TRẦN nên thiếu transformers, worker hỏng mọi câu. Chạy lệnh nâng cấp
                # để sửa môi trường cũ mà không phải xoá đi làm lại.
                from subtitles_extractor.infrastructure.tts.vieneu_gpu_plan import (
                    repair_command,
                )

                self.progress.emit(10, "Kiểm và bổ sung gói còn thiếu…")
                repair = repair_command(plan.python_exe or "")
                result = subprocess.run(
                    repair, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=1800, check=False,
                    **_no_window_kwargs(),
                )
                if result.returncode != 0:
                    tail = (result.stderr or result.stdout or "").strip()
                    self.failed.emit(
                        "Bổ sung gói thất bại:\n" + "\n".join(tail.splitlines()[-3:])
                    )
                    return
                self.progress.emit(100, "Hoàn tất.")
                self.finished.emit(plan.python_exe or "")
                return
            if plan.python_exe is None:
                self.failed.emit(
                    "Chưa có môi trường riêng. Hãy bấm “⬇️ Cài WhisperX tự động” trong "
                    "nhóm Giọng nói (STT) trước — bước đó cài torch bản CUDA, thứ mà "
                    "TTS trên GPU cũng cần."
                )
                return

            total = len(plan.install_commands) or 1
            for index, command in enumerate(plan.install_commands):
                if self._cancelled:
                    self.failed.emit("Đã huỷ cài đặt theo yêu cầu.")
                    return

                # An toàn: mọi lệnh phải chạy bằng Python của môi trường RIÊNG.
                if command[0] != plan.python_exe:
                    self.failed.emit(
                        "Từ chối lệnh cài không trỏ vào môi trường riêng."
                    )
                    return

                self.progress.emit(
                    int(index / total * 100),
                    f"Đang cài {' '.join(command[4:])}…",
                )
                logger.info("Cài VieNeu GPU: %s", " ".join(command))
                result = subprocess.run(
                    command, capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                    timeout=1800, check=False, **_no_window_kwargs(),
                )
                if result.returncode != 0:
                    tail = (result.stderr or result.stdout or "").strip()
                    self.failed.emit(
                        "Cài thất bại:\n" + "\n".join(tail.splitlines()[-3:])
                    )
                    return

            # Xác nhận lại bằng cách lập kế hoạch mới — không tin suông vào mã thoát.
            verified = build_gpu_tts_plan()
            if not verified.is_ready:
                self.failed.emit(
                    "Cài xong nhưng kiểm lại vẫn thiếu: "
                    f"{', '.join(verified.missing_packages)}"
                )
                return

            self.progress.emit(100, "Hoàn tất.")
            self.finished.emit(verified.python_exe or "")
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.exception("Lỗi hệ thống khi cài VieNeu GPU.")
            self.failed.emit(f"Lỗi hệ thống: {exc}")
        finally:
            self.done.emit()


def _no_window_kwargs() -> dict[str, object]:
    """Ẩn cửa sổ console trên Windows."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


__all__ = ["InstallVieneuGpuWorker"]
