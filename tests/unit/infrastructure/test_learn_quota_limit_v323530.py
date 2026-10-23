"""[v3.23.530] Học hạn mức NGÀY thật từ thông báo lỗi của server.

Bảng giới hạn trong ứng dụng là con số chép tay và lạc hậu ngay khi Google đổi chính sách. Nhật ký
thật: ứng dụng cấu hình 70 request/ngày trong khi server nêu rõ hạn mức là 20 — nên sau khi bị từ
chối, ứng dụng vẫn tưởng key kế tiếp "còn 20 req/ngày" rồi lại đâm vào tường.
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
        rate_limits={MODEL: RateLimit(rpm=5, tpm=250_000, rpd=70)},
        state_path=tmp_path / "quota.json",
    )


class TestLearning:
    def test_updates_limit_from_server(self, manager) -> None:
        assert manager.learn_daily_limit_from_server(MODEL, 20) is True
        assert manager.get_remaining(MODEL, key_id="k1")["rpd_limit"] == 20

    def test_same_value_is_noop(self, manager) -> None:
        manager.learn_daily_limit_from_server(MODEL, 20)
        assert manager.learn_daily_limit_from_server(MODEL, 20) is False

    @pytest.mark.parametrize("value", [0, -5])
    def test_invalid_value_ignored(self, manager, value: int) -> None:
        assert manager.learn_daily_limit_from_server(MODEL, value) is False
        assert manager.get_remaining(MODEL, key_id="k1")["rpd_limit"] == 70

    def test_keeps_other_limits(self, manager) -> None:
        """Chỉ đổi hạn mức NGÀY, không đụng giới hạn phút."""
        manager.learn_daily_limit_from_server(MODEL, 20)
        remaining = manager.get_remaining(MODEL, key_id="k1")
        assert remaining["rpd_limit"] == 20
        assert remaining.get("rpm_limit", 5) == 5

    def test_unknown_model_gets_limit_too(self, manager) -> None:
        assert manager.learn_daily_limit_from_server("model-hoan-toan-moi", 15) is True
        assert manager.get_remaining("model-hoan-toan-moi", key_id="k1")["rpd_limit"] == 15


class TestAdapterWiring:
    def test_adapter_learns_on_daily_quota_error(self) -> None:
        import inspect

        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            GeminiSubtitleTranslator,
        )

        source = inspect.getsource(GeminiSubtitleTranslator._log_daily_quota_context)
        assert "learn_daily_limit_from_server" in source
