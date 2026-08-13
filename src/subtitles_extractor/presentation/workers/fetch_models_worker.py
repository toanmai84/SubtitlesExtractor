"""Worker QThread lấy danh sách model Gemini không chặn UI."""

from __future__ import annotations

import logging
from threading import Event

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class FetchModelsWorker(QThread):
    """Lấy danh sách model Gemini trên thread nền.

    **Vòng đời an toàn**:
    ViewModel lưu instance này vào ``self._fetch_models_worker`` (strong reference)
    để tránh Python GC huỷ object khi thread vẫn chạy → crash
    ``QThread: Destroyed while thread is still running``.

    Signals:
        models_ready: Phát ``list[str]`` khi lấy thành công.
        failed:       Phát thông điệp lỗi khi thất bại.
    """

    models_ready = Signal(list)   # list[str]
    failed = Signal(str)

    def __init__(self, translator, parent=None) -> None:
        super().__init__(parent)
        self._translator = translator
        self._cancel_event = Event()

    def request_cancel(self) -> None:
        """Yêu cầu dừng (đặt cờ — thread kết thúc tự nhiên sau API call)."""
        self._cancel_event.set()

    def run(self) -> None:
        try:
            if self._cancel_event.is_set():
                return
            models: list[str] = self._translator.list_available_models()
            if self._cancel_event.is_set():
                return
            if models:
                logger.info("FetchModelsWorker: tìm được %d model.", len(models))
                self.models_ready.emit(models)
            else:
                # Trả về rỗng: có thể API key chưa có quyền hoặc phiên bản SDK cũ
                logger.warning(
                    "FetchModelsWorker: list_available_models trả về rỗng. "
                    "Kiểm tra log DEBUG để biết chi tiết."
                )
                self.failed.emit(
                    "Không tìm được model nào từ API. "
                    "Có thể API Key không có quyền truy cập Gemini, "
                    "hoặc mạng bị tường lửa chặn. "
                    "Xem log để biết thêm chi tiết."
                )
        except Exception as exc:  # noqa: BLE001
            # [v3.23.340] Dùng exception để ghi kèm traceback — `except Exception` bắt
            # cả lỗi không lường trước, thiếu traceback thì gần như không chẩn đoán được.
            logger.exception("FetchModelsWorker lỗi.")
            if not self._cancel_event.is_set():
                self.failed.emit(f"Lỗi khi lấy danh sách model: {exc}")


__all__ = ["FetchModelsWorker"]
