"""Worker xuất video hoàn chỉnh trên QThread (không khoá giao diện).

Việc mã hoá lại video có thể mất hàng chục phút, nên bắt buộc chạy ngoài luồng giao
diện. Worker này bọc :func:`render_video` và phát tín hiệu tiến độ để trang Xuất bản
cập nhật thanh trạng thái.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.infrastructure.video.ffmpeg_video_renderer import render_video
from subtitles_extractor.infrastructure.video.video_render_command import (
    RenderRequest,
    VideoRenderError,
)

logger = logging.getLogger(__name__)


class RenderVideoWorker(QObject):
    """Chạy một yêu cầu xuất video và báo tiến độ.

    Signals:
        progress: Tỉ lệ hoàn thành 0–100 (phần trăm, số nguyên cho thanh tiến độ).
        finished: Phát ra đường dẫn tệp đã xuất khi thành công.
        failed: Phát ra thông điệp lỗi (đã thân thiện với người dùng).
        done: Luôn phát ra ở cuối, kể cả khi lỗi — để dọn dẹp luồng.
    """

    progress = Signal(int)
    finished = Signal(object)
    failed = Signal(str)
    done = Signal()

    def __init__(self, request: RenderRequest, total_duration_sec: float) -> None:
        super().__init__()
        self._request = request
        self._total_duration_sec = total_duration_sec
        self._cancelled = False

    def cancel(self) -> None:
        """Yêu cầu dừng. An toàn khi gọi từ luồng giao diện."""
        self._cancelled = True

    def run(self) -> None:
        """Thực hiện xuất video. Gọi trong QThread, KHÔNG gọi trực tiếp."""
        try:
            output = render_video(
                self._request,
                total_duration_sec=self._total_duration_sec,
                on_progress=lambda ratio: self.progress.emit(int(ratio * 100)),
                is_cancelled=lambda: self._cancelled,
            )
            self.finished.emit(output)
        except VideoRenderError as exc:
            logger.warning("Xuất video thất bại: %s", exc)
            self.failed.emit(str(exc))
        except (OSError, RuntimeError) as exc:
            logger.exception("Lỗi hệ thống khi xuất video.")
            self.failed.emit(f"Lỗi hệ thống: {exc}")
        finally:
            self.done.emit()


__all__ = ["RenderVideoWorker"]
