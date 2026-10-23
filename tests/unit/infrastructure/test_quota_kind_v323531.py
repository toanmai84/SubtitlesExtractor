"""[v3.23.531] Phân biệt hạn mức theo SỐ REQUEST/ngày với hạn mức theo SỐ TOKEN/ngày.

Trước đây gộp chung thành một nhãn "quota token/NGÀY", nên nhật ký nói sai loại: bảng điều khiển
hiện RPD mới 0/20 mà ứng dụng lại báo hết request/ngày — không cách nào truy ra sự thật. Tệ hơn,
v3.23.530 học hạn mức từ ``quotaValue`` mà không xét loại: gặp quota TOKEN/ngày (giá trị hàng triệu)
sẽ đặt hạn mức request thành 1.000.000 và vô hiệu hoá lớp bảo vệ.
"""
from __future__ import annotations

import pathlib
import tempfile

import pytest

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    RateLimit,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

REQUEST_QUOTA = (
    "429 RESOURCE_EXHAUSTED {'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '20'}"
)
TOKEN_DAY_QUOTA = (
    "429 RESOURCE_EXHAUSTED {'quotaId': "
    "'GenerateContentInputTokensPerModelPerDay-FreeTier', 'quotaValue': '1000000'}"
)
TOKEN_MINUTE_QUOTA = (
    "429 RESOURCE_EXHAUSTED {'quotaId': "
    "'GenerateContentInputTokensPerModelPerMinute-FreeTier'}"
)


class TestConciseLabel:
    def test_request_per_day_labelled(self) -> None:
        text = GeminiSubtitleTranslator._concise_error(Exception(REQUEST_QUOTA))
        assert "SỐ REQUEST/ngày" in text

    def test_token_per_day_labelled(self) -> None:
        text = GeminiSubtitleTranslator._concise_error(Exception(TOKEN_DAY_QUOTA))
        assert "TOKEN/ngày" in text

    def test_raw_quota_id_preserved(self) -> None:
        """Giữ mã quota gốc để không phải đoán khi chẩn đoán."""
        text = GeminiSubtitleTranslator._concise_error(Exception(REQUEST_QUOTA))
        assert "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in text

    def test_per_minute_unchanged(self) -> None:
        text = GeminiSubtitleTranslator._concise_error(Exception(TOKEN_MINUTE_QUOTA))
        assert "phút" in text


class TestLimitLearningGuard:
    @staticmethod
    def _translator():
        directory = pathlib.Path(tempfile.mkdtemp())
        manager = GeminiQuotaManager(
            rate_limits={"m": RateLimit(rpm=5, tpm=250_000, rpd=70)},
            state_path=directory / "quota.json",
        )
        translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
        translator._quota_manager = manager
        translator._current_key_id = lambda: "k1"
        return translator, manager

    def test_learns_from_request_quota(self) -> None:
        translator, manager = self._translator()
        translator._log_daily_quota_context(Exception(REQUEST_QUOTA), "m")
        assert manager.get_remaining("m", key_id="k1")["rpd_limit"] == 20

    def test_ignores_token_quota_value(self) -> None:
        """Quota TOKEN/ngày có giá trị hàng triệu — không được dùng làm hạn mức request."""
        translator, manager = self._translator()
        translator._log_daily_quota_context(Exception(TOKEN_DAY_QUOTA), "m")
        assert manager.get_remaining("m", key_id="k1")["rpd_limit"] == 70

    def test_no_quota_value_keeps_limit(self) -> None:
        translator, manager = self._translator()
        translator._log_daily_quota_context(Exception("429 RESOURCE_EXHAUSTED"), "m")
        assert manager.get_remaining("m", key_id="k1")["rpd_limit"] == 70
