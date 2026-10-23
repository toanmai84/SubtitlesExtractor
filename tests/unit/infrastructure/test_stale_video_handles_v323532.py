"""[v3.23.532] Sau khi xoay API key, KHÔNG thử lại handle video cũ.

File tải lên qua Files API gắn với project của API key. Sau khi xoay key, mọi handle cũ chắc chắn
trả 403. Trước đây ứng dụng vẫn thử dùng chúng rồi mới nhận lỗi — mà MỖI LẦN THỬ HỎNG VẪN TÍNH MỘT
REQUEST vào hạn mức key mới. Nhật ký thật cho thấy chuỗi 403 liên tiếp cho từng đoạn; với 23 đoạn và
hạn mức 20 request/ngày, riêng các lần thử vô ích đã đủ làm cạn quota ngay sau khi xoay key.
"""
from __future__ import annotations

import inspect

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


class TestStaleFlag:
    def test_flag_initialised(self) -> None:
        source = inspect.getsource(GeminiSubtitleTranslator.__init__)
        assert "_video_handles_stale = False" in source

    def test_key_switch_marks_handles_stale(self) -> None:
        source = inspect.getsource(GeminiSubtitleTranslator._switch_to_key)
        assert "_video_handles_stale = True" in source

    def test_switch_also_clears_memory_cache(self) -> None:
        source = inspect.getsource(GeminiSubtitleTranslator._switch_to_key)
        assert "_video_handle_cache.clear()" in source


class TestSkipDoomedResolve:
    @staticmethod
    def _heal_source() -> str:
        for name in dir(GeminiSubtitleTranslator):
            member = getattr(GeminiSubtitleTranslator, name, None)
            if not callable(member):
                continue
            try:
                text = inspect.getsource(member)
            except (OSError, TypeError):
                continue
            if "_video_reupload_cb(self._api_key)" in text:
                return text
        raise AssertionError("không tìm thấy hàm heal video")

    def test_skips_resolve_when_stale(self) -> None:
        source = self._heal_source()
        assert "_video_handles_stale" in source
        assert "bỏ qua lần thử hỏng" in source

    def test_still_resolves_when_not_stale(self) -> None:
        """Chưa đổi key thì vẫn thử dùng handle sẵn có (tái dùng được thì tiết kiệm nhất)."""
        source = self._heal_source()
        assert "else:" in source
        assert "_resolve_video_handles(video_refs)" in source

    def test_clears_flag_after_successful_reupload(self) -> None:
        source = self._heal_source()
        assert "_video_handles_stale = False" in source
