"""[v3.23.695] Nạp danh sách giọng VieNeu ở luồng nền cho trang Lồng tiếng.

Tách nguyên văn khỏi ``tts_page.py`` (U6) — hành vi không đổi.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal

#: Giữ tên logger của trang Lồng tiếng (lọc log theo trang vẫn thấy).
logger = logging.getLogger("subtitles_extractor.presentation.pages.tts_page")


class _VieNeuVoiceLoader(QThread):
    """[v3.23.200] Nạp danh sách giọng VieNeu ở BACKGROUND thread (hết khựng UI ~9s).

    ``list_speakers`` phải nạp model VieNeu (llama+codec, ~9s lần đầu) — chạy trên UI
    thread làm ứng dụng đứng hình khi người dùng đổi engine/chế độ (log thực tế
    11:35:55 -> 11:36:03). Worker này nạp ở thread nền rồi phát tín hiệu về UI thread;
    kết quả kèm ``mode`` để UI bỏ qua kết quả STALE khi người dùng đã đổi chế độ giữa
    chừng. Model được cache thread-safe (``_ENGINE_LOCK``) nên Generate chạy song song
    không nạp trùng.
    """

    voices_loaded = Signal(str, list)  # (mode, danh sách tên giọng)
    load_failed = Signal(str)          # (mode)
    # [v3.23.593] (mode, list[VoiceProfile]) — giọng kèm giới tính cho bảng phân vai.
    catalog_loaded = Signal(str, list)

    def __init__(self, container, mode: str, parent=None, precision: str = "") -> None:
        super().__init__(parent)
        self._container = container
        self._mode = mode
        # [v3.23.595] Cùng độ chính xác với lần tổng hợp → dùng chung engine đã nạp (khoá
        # cache gồm độ chính xác), không nạp model hai lần.
        self._precision = precision

    def run(self) -> None:  # noqa: D102 — QThread API
        try:
            adapter = self._container.make_vieneu_tts_adapter(
                mode=self._mode, precision=self._precision or None
            )
            if not adapter.is_available():
                self.load_failed.emit(self._mode)
                return
            voice_ids = adapter.list_speakers("vi-VN") or []
            catalog = adapter.list_voice_catalog("vi-VN") or []
        except Exception as exc:  # noqa: BLE001
            # [v3.23.650] MỌI lỗi đều phải phát load_failed: ngoại lệ lạ (TypeError/KeyError khi vieneu đổi API…) từng
            # thoát khỏi run() không phát tín hiệu → combo giọng kẹt "Đang nạp…" và nút Tổng hợp tắt tới khi khởi động lại.
            logger.warning("Nạp giọng VieNeu (nền) lỗi: %s", exc)
            self.load_failed.emit(self._mode)
            return
        self.catalog_loaded.emit(self._mode, list(catalog))
        self.voices_loaded.emit(self._mode, list(voice_ids))


__all__ = ['_VieNeuVoiceLoader']
