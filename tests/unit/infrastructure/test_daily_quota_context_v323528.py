"""[v3.23.528] Ghi ngữ cảnh khi server báo hết hạn mức NGÀY, để đối chiếu với bảng Google.

Quan sát thật: bảng điều khiển ghi Gemini 3.8 Flash mới dùng 0/20 và 1/20 ở hai dự án, nhưng server
vẫn từ chối cả ba API key. Dấu hiệu điển hình của việc các key dùng CHUNG một dự án — hạn mức ngày
tính theo DỰ ÁN chứ không theo key, trong khi ứng dụng đếm riêng từng key nên tưởng còn nhiều.
"""
from __future__ import annotations

import logging

import pytest

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    RateLimit,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

MODEL = "gemini-3.8-flash"
ERROR = Exception(
    "429 RESOURCE_EXHAUSTED {'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '20'}"
)


@pytest.fixture
def translator(tmp_path) -> GeminiSubtitleTranslator:
    manager = GeminiQuotaManager(
        rate_limits={MODEL: RateLimit(rpm=100, tpm=9_000_000, rpd=70)},
        state_path=tmp_path / "quota.json",
    )
    for _ in range(3):
        manager.acquire(MODEL, estimated_tokens=1, key_id="AIzaSyXXXX1234")
    instance = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    instance._quota_manager = manager
    instance._current_key_id = lambda: "AIzaSyXXXX1234"
    return instance


class TestContextLogging:
    def test_reports_server_limit_and_local_count(self, translator, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            translator._log_daily_quota_context(ERROR, MODEL)
        message = caplog.text
        assert "20" in message  # hạn mức server nêu
        assert "3" in message  # số ứng dụng đếm được

    def test_masks_key_to_last_four_chars(self, translator, caplog) -> None:
        """Không in nguyên API key ra nhật ký."""
        with caplog.at_level(logging.WARNING):
            translator._log_daily_quota_context(ERROR, MODEL)
        assert "1234" in caplog.text
        assert "AIzaSyXXXX1234" not in caplog.text

    def test_explains_project_level_quota(self, translator, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            translator._log_daily_quota_context(ERROR, MODEL)
        assert "CÙNG MỘT DỰ ÁN" in caplog.text

    def test_missing_quota_value_is_safe(self, translator, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            translator._log_daily_quota_context(Exception("429 RESOURCE_EXHAUSTED"), MODEL)
        assert "không rõ" in caplog.text

    def test_never_raises(self, caplog) -> None:
        """Ghi chẩn đoán không được làm hỏng luồng xử lý lỗi."""
        broken = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
        broken._quota_manager = None
        broken._current_key_id = lambda: ""
        broken._log_daily_quota_context(ERROR, MODEL)
