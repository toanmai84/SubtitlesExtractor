"""[v3.23.366] Worker nối các tập thành một video trọn bộ trên QThread.

Nối video có thể mất từ vài giây (sao chép luồng) tới hàng chục phút (nén lại), nên chạy
ngoài luồng giao diện và phát tín hiệu tiến độ cho trang Xuất bản.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.infrastructure.video.ffmpeg_concat_adapter import (
    VideoConcatError,
    concatenate_videos,
)

logger = logging.getLogger(__name__)


class ConcatVideoWorker(QObject):
    """Nối một danh sách video (đã sắp thứ tự) thành một tệp trọn bộ.

    Signals:
        progress: Tỉ lệ hoàn thành 0–100.
        finished: Đường dẫn tệp trọn bộ khi thành công.
        failed: Thông điệp lỗi thân thiện.
        done: Luôn phát ở cuối để dọn luồng.
    """

    progress = Signal(int)
    finished = Signal(object)
    failed = Signal(str)
    done = Signal()

    def __init__(
        self,
        videos: list[Path],
        output_path: Path,
        total_duration_sec: float,
        *,
        reencode: bool = False,
    ) -> None:
        super().__init__()
        self._videos = videos
        self._output_path = output_path
        self._total_duration_sec = total_duration_sec
        self._reencode = reencode
        self._cancelled = False

    def cancel(self) -> None:
        """Yêu cầu dừng (an toàn từ luồng giao diện)."""
        self._cancelled = True

    def run(self) -> None:
        """Thực hiện nối video. Gọi trong QThread, KHÔNG gọi trực tiếp."""
        try:
            output = concatenate_videos(
                self._videos,
                self._output_path,
                reencode=self._reencode,
                total_duration_sec=self._total_duration_sec,
                on_progress=lambda ratio: self.progress.emit(int(ratio * 100)),
                cancel_check=lambda: self._cancelled,
            )
            self.finished.emit(output)
        except VideoConcatError as exc:
            logger.warning("Nối video thất bại: %s", exc)
            self.failed.emit(str(exc))
        except (OSError, RuntimeError) as exc:
            logger.exception("Lỗi hệ thống khi nối video.")
            self.failed.emit(f"Lỗi hệ thống: {exc}")
        finally:
            self.done.emit()


__all__ = ["ConcatVideoWorker"]
