"""[v3.23.534] Hai nguyên nhân làm ứng dụng "thỉnh thoảng treo".

1. ``rpm`` <= 0 (cấu hình sai hoặc học nhầm từ server) khiến vòng chờ quota đọc phần tử đầu của một
   hàng đợi RỖNG → ``IndexError`` làm sập tác vụ giữa chừng.
2. ``retryDelay`` của server không có trần: khi hết hạn mức ngày, server có thể yêu cầu chờ hàng
   chục phút tới hàng giờ. Ứng dụng chờ đúng ngần ấy nên đứng im — người dùng thấy như treo.
"""
from __future__ import annotations

import threading

import pytest

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    MAX_SERVER_COOLDOWN_SEC,
    GeminiQuotaManager,
    RateLimit,
)


def _manager(tmp_path, **kwargs) -> GeminiQuotaManager:
    limit = RateLimit(kwargs.pop("rpm", 100), kwargs.pop("tpm", 10_000), kwargs.pop("rpd", 50))
    return GeminiQuotaManager(rate_limits={"m": limit}, state_path=tmp_path / "quota.json")


class TestZeroRpmGuard:
    def test_zero_rpm_does_not_crash(self, tmp_path) -> None:
        manager = _manager(tmp_path, rpm=0)
        result: list[str] = []

        def run() -> None:
            try:
                manager.acquire("m", estimated_tokens=1, key_id="k")
                result.append("ok")
            except Exception as exc:  # noqa: BLE001 — test cần bắt mọi lỗi
                result.append(type(exc).__name__)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(5)
        assert result == ["ok"]

    def test_negative_rpm_does_not_crash(self, tmp_path) -> None:
        manager = _manager(tmp_path, rpm=-3)
        manager.acquire("m", estimated_tokens=1, key_id="k")

    def test_normal_rpm_still_limits(self, tmp_path) -> None:
        """Không được nới lỏng giới hạn bình thường khi thêm guard."""
        manager = _manager(tmp_path, rpm=2)
        manager.acquire("m", estimated_tokens=1, key_id="k")
        manager.acquire("m", estimated_tokens=1, key_id="k")
        assert manager.get_remaining("m", key_id="k")["rpd_used"] == 2


class TestCooldownCap:
    def test_long_server_delay_is_capped(self, tmp_path) -> None:
        manager = _manager(tmp_path)
        manager.note_rate_limited("m", 3600.0, key_id="k")
        assert manager.cooldown_remaining_s("m", key_id="k") <= MAX_SERVER_COOLDOWN_SEC + 1

    def test_short_delay_kept_as_is(self, tmp_path) -> None:
        manager = _manager(tmp_path)
        manager.note_rate_limited("m", 30.0, key_id="k")
        remaining = manager.cooldown_remaining_s("m", key_id="k")
        assert 25 <= remaining <= 31

    def test_cap_is_reasonable(self) -> None:
        """Đủ dài để tôn trọng server, đủ ngắn để không giống treo."""
        assert 60 <= MAX_SERVER_COOLDOWN_SEC <= 300

    @pytest.mark.parametrize("delay", [0, -5])
    def test_invalid_delay_ignored(self, tmp_path, delay: float) -> None:
        manager = _manager(tmp_path)
        manager.note_rate_limited("m", delay, key_id="k")
        assert manager.cooldown_remaining_s("m", key_id="k") == 0.0
