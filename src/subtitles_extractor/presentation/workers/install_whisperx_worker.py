"""Worker cài WhisperX trên luồng riêng (không khoá giao diện).

Việc cài tải khoảng 3GB nên có thể mất nhiều phút; bắt buộc chạy ngoài luồng giao diện.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.infrastructure.stt.whisperx_installer import (
    WhisperXInstallError,
    install_whisperx,
)

logger = logging.getLogger(__name__)


class InstallWhisperXWorker(QObject):
    """Cài WhisperX vào môi trường riêng và báo tiến độ.

    Signals:
        progress: ``(phần trăm 0–100, mô tả bước)``.
        finished: Đường dẫn Python của môi trường vừa tạo.
        failed: Thông điệp lỗi đã thân thiện với người dùng.
        done: Luôn phát ở cuối, kể cả khi lỗi — để dọn luồng.
    """

    progress = Signal(int, str)
    finished = Signal(str)
    failed = Signal(str)
    done = Signal()

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self._project_root = project_root
        self._cancelled = False

    def cancel(self) -> None:
        """Yêu cầu dừng. An toàn khi gọi từ luồng giao diện.

        Chỉ dừng được GIỮA các bước, không cắt ngang một lệnh pip đang tải.
        """
        self._cancelled = True

    def run(self) -> None:
        """Thực hiện cài. Gọi trong QThread, KHÔNG gọi trực tiếp."""
        try:
            env_python = install_whisperx(
                self._project_root,
                on_progress=lambda ratio, label: self.progress.emit(
                    int(ratio * 100), label
                ),
                is_cancelled=lambda: self._cancelled,
            )
            self.finished.emit(str(env_python))
        except WhisperXInstallError as exc:
            logger.warning("Cài WhisperX thất bại: %s", exc)
            self.failed.emit(str(exc))
        except (OSError, RuntimeError) as exc:
            logger.exception("Lỗi hệ thống khi cài WhisperX.")
            self.failed.emit(f"Lỗi hệ thống: {exc}")
        finally:
            self.done.emit()


__all__ = ["InstallWhisperXWorker"]
