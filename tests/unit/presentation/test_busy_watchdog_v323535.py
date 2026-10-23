"""[v3.23.535] Hai lớp chống lỗi "treo, không bấm được gì, phải End Task".

Chuỗi nhân quả tìm được: giao diện chỉ mở khoá trong handler ``finished`` của luồng — tức khi
``run()`` trả về. Bộ cung cấp video lại dựng client Gemini KHÔNG có thời hạn (khác client của
adapter dịch), nên một lời gọi Files API treo là treo vĩnh viễn: ``run()`` không trả về → tín hiệu
``finished`` không phát → giao diện khoá mãi. Nhật ký thật dừng đột ngột giữa chuỗi kiểm file trên
cloud, đúng kịch bản này.
"""
from __future__ import annotations

from pathlib import Path

VIDEO_CONTEXT = Path(
    "src/subtitles_extractor/infrastructure/translation/gemini_video_context.py"
)
VIEW_MODEL = Path(
    "src/subtitles_extractor/presentation/view_models/translate_page_view_model.py"
)


class TestClientTimeout:
    def test_video_client_sets_timeout(self) -> None:
        source = VIDEO_CONTEXT.read_text(encoding="utf-8")
        assert "http_options={\"timeout\": timeout_ms}" in source

    def test_timeout_constant_is_finite_and_generous(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            CLIENT_TIMEOUT_SEC,
        )

        assert 60 <= CLIENT_TIMEOUT_SEC <= 600

    def test_old_sdk_falls_back_without_crash(self) -> None:
        """SDK cũ không nhận http_options → vẫn chạy, chỉ cảnh báo."""
        source = VIDEO_CONTEXT.read_text(encoding="utf-8")
        assert "except TypeError:" in source
        # [v3.23.609] Client tạo theo từng key (xoá file cloud thử mọi key).
        assert "genai.Client(api_key=api_key)" in source


class TestBusyWatchdog:
    def test_watchdog_exists(self) -> None:
        source = VIEW_MODEL.read_text(encoding="utf-8")
        assert "_busy_watchdog" in source
        assert "_check_busy_watchdog" in source

    def test_watchdog_started_when_busy(self) -> None:
        source = VIEW_MODEL.read_text(encoding="utf-8")
        index = source.index("def _set_busy")
        block = source[index : index + 900]
        assert "_busy_watchdog.start" in block
        assert "_busy_watchdog.stop" in block

    def test_unlocks_only_when_no_thread_alive(self) -> None:
        """Không đoán 'treo' theo thời gian — chỉ mở khoá khi chắc chắn sai."""
        source = VIEW_MODEL.read_text(encoding="utf-8")
        index = source.index("def _check_busy_watchdog")
        block = source[index : index + 1200]
        assert "isRunning" in block
        assert "if alive:" in block

    def test_interval_is_reasonable(self) -> None:
        import re

        source = VIEW_MODEL.read_text(encoding="utf-8")
        value = int(re.search(r"BUSY_WATCHDOG_INTERVAL_MS = (\d+)", source).group(1))
        assert 1000 <= value <= 15000
