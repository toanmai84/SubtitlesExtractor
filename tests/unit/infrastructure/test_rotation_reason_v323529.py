"""[v3.23.529] Nhật ký phải ghi rõ AI ra quyết định xoay API key.

Dòng thời gian trong nhật ký thật: xoay key lúc 19:51:59 và 19:57:53, trong khi 429 ĐẦU TIÊN từ
server mãi 20:03:43 mới tới. Tức ứng dụng tự xoay khỏi những key VẪN CÒN quota, dồn tải vào key
cuối rồi mới thật sự chạm trần. Không ghi lý do thì không thể truy ra ai quyết định.
"""
from __future__ import annotations

import pytest

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    RateLimit,
)

MODEL = "gemini-3.8-flash"


@pytest.fixture
def manager(tmp_path) -> GeminiQuotaManager:
    return GeminiQuotaManager(
        rate_limits={MODEL: RateLimit(rpm=100, tpm=9_000_000, rpd=2)},
        state_path=tmp_path / "quota.json",
    )


class TestServerBlockFlag:
    def test_fresh_key_not_blocked(self, manager) -> None:
        assert manager.is_server_daily_blocked(MODEL, key_id="k1") is False

    def test_internal_counter_full_is_not_server_block(self, manager) -> None:
        """Bộ đếm nội bộ chạm trần KHÔNG phải là server từ chối."""
        manager.acquire(MODEL, estimated_tokens=1, key_id="k1")
        manager.acquire(MODEL, estimated_tokens=1, key_id="k1")
        assert manager.is_server_daily_blocked(MODEL, key_id="k1") is False

    def test_server_confirmation_sets_flag(self, manager) -> None:
        manager.mark_daily_exhausted(MODEL, key_id="k1")
        assert manager.is_server_daily_blocked(MODEL, key_id="k1") is True

    def test_flag_is_per_key(self, manager) -> None:
        manager.mark_daily_exhausted(MODEL, key_id="k1")
        assert manager.is_server_daily_blocked(MODEL, key_id="k2") is False

    def test_flag_is_per_model(self, manager) -> None:
        manager.mark_daily_exhausted(MODEL, key_id="k1")
        assert manager.is_server_daily_blocked("model-khac", key_id="k1") is False


class TestRotationLogsReason:
    def test_log_message_includes_source(self) -> None:
        import inspect

        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            GeminiSubtitleTranslator,
        )

        source = inspect.getsource(
            GeminiSubtitleTranslator._rotate_to_available_key_locked
        )
        assert "nguồn:" in source
        assert "SERVER từ chối" in source
        assert "bộ đếm nội bộ" in source
        assert "is_server_daily_blocked" in source
