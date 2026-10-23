"""[v3.23.526] Bộ đếm quota NỘI BỘ chỉ là ước lượng — không được tự khoá khi server còn cho phép.

Nhật ký thật: ứng dụng báo "Model 'gemini-3.8-flash' đã chạm giới hạn 50 request/ngày" rồi làm SẬP
khâu phân tích, trong khi bảng điều khiển Google cùng thời điểm ghi mới dùng 0/20, 1/20 và 17/20
trên ba dự án — tức còn rất nhiều quota. Bộ đếm nội bộ lệch xa thực tế nên chặn cứng là tự khoá mình.
"""
from __future__ import annotations

import pytest

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    QuotaExhaustedError,
)

MODEL = "gemini-3.8-flash"


def _manager(tmp_path, rpd: int = 3) -> GeminiQuotaManager:
    from subtitles_extractor.infrastructure.translation.gemini_quota_manager import RateLimit

    return GeminiQuotaManager(
        rate_limits={MODEL: RateLimit(rpm=100, tpm=10_000_000, rpd=rpd)},
        state_path=tmp_path / "quota.json",
    )


def _use_all(manager: GeminiQuotaManager, times: int) -> None:
    for _ in range(times):
        manager.acquire(MODEL, estimated_tokens=1, key_id="k1")


class TestSoftBlocking:
    def test_internal_counter_does_not_hard_block(self, tmp_path) -> None:
        """Đếm nội bộ chạm trần nhưng server chưa từ chối → VẪN gửi tiếp."""
        manager = _manager(tmp_path)
        _use_all(manager, 3)
        manager.acquire(MODEL, estimated_tokens=1, key_id="k1")  # không được ném lỗi

    def test_server_confirmation_hard_blocks(self, tmp_path) -> None:
        """Server xác nhận hết quota ngày → chặn cứng như cũ."""
        manager = _manager(tmp_path)
        manager.mark_daily_exhausted(MODEL, key_id="k1")
        with pytest.raises(QuotaExhaustedError):
            manager.acquire(MODEL, estimated_tokens=1, key_id="k1")

    def test_block_is_per_key(self, tmp_path) -> None:
        """Khoá một key không được ảnh hưởng key khác — đó là lý do có nhiều key."""
        manager = _manager(tmp_path)
        manager.mark_daily_exhausted(MODEL, key_id="k1")
        manager.acquire(MODEL, estimated_tokens=1, key_id="k2")

    def test_error_message_states_server_confirmed(self, tmp_path) -> None:
        manager = _manager(tmp_path)
        manager.mark_daily_exhausted(MODEL, key_id="k1")
        with pytest.raises(QuotaExhaustedError, match="server xác nhận"):
            manager.acquire(MODEL, estimated_tokens=1, key_id="k1")
