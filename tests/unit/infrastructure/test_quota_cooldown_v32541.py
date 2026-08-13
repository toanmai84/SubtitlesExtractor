"""[v3.23.141] Test cooldown 429 + token/giây video theo media_resolution."""
from __future__ import annotations

import time

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    RateLimit,
)
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    video_tokens_per_sec,
)


def _manager() -> GeminiQuotaManager:
    return GeminiQuotaManager(
        rate_limits={"flash": RateLimit(rpm=5, tpm=250_000, rpd=20)},
    )


def test_video_tokens_per_sec_mapping() -> None:
    assert video_tokens_per_sec("low") == 100
    assert video_tokens_per_sec("medium") == 300
    assert video_tokens_per_sec("high") == 400
    assert video_tokens_per_sec("") == 300  # mặc định an toàn = medium
    assert video_tokens_per_sec("LOW") == 100  # không phân biệt hoa-thường


def test_note_rate_limited_sets_cooldown_and_acquire_waits() -> None:
    mgr = _manager()
    calls: list[tuple[float, str]] = []

    def wait_cb(seconds: float, reason: str) -> None:
        calls.append((seconds, reason))
        # Xoá cooldown ngay để test không phải ngủ thật quá lâu.
        mgr._cooldown_until.clear()

    mgr.note_rate_limited("flash", 30.0, key_id="k1")
    # acquire phải phát hiện cooldown và gọi wait_cb với lý do cooldown.
    mgr.acquire("flash", 1000, wait_cb=wait_cb, key_id="k1")

    assert calls, "acquire phải chờ do cooldown"
    seconds, reason = calls[0]
    assert "cooldown" in reason
    assert seconds > 0


def test_note_rate_limited_ignores_non_positive() -> None:
    mgr = _manager()
    mgr.note_rate_limited("flash", 0.0, key_id="k1")
    mgr.note_rate_limited("flash", -5.0, key_id="k1")
    # Không đặt cooldown -> acquire qua ngay, không chờ.
    started = time.time()
    mgr.acquire("flash", 10, key_id="k1")
    assert time.time() - started < 1.0


def test_cooldown_is_per_key() -> None:
    mgr = _manager()
    mgr.note_rate_limited("flash", 60.0, key_id="k1")
    # Key khác KHÔNG bị ảnh hưởng -> acquire qua ngay.
    started = time.time()
    mgr.acquire("flash", 10, key_id="k2")
    assert time.time() - started < 1.0


def test_cooldown_keeps_maximum() -> None:
    mgr = _manager()
    mgr.note_rate_limited("flash", 20.0, key_id="k1")
    first = mgr._cooldown_until[mgr._compose_key("k1", "flash")]
    mgr.note_rate_limited("flash", 5.0, key_id="k1")  # ngắn hơn -> giữ mốc cũ
    second = mgr._cooldown_until[mgr._compose_key("k1", "flash")]
    assert second == first
