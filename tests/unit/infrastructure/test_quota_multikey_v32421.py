"""[v3.23.121] Test: quota tách theo API key + nhiều key tự xoay khi hết quota."""

from __future__ import annotations

import pytest

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    QuotaExhaustedError,
    RateLimit,
)


def _mgr() -> GeminiQuotaManager:
    # RPD nhỏ để dễ test chạm trần.
    return GeminiQuotaManager(
        rate_limits={"gemini-x": RateLimit(rpm=100, tpm=10**9, rpd=2)}
    )


def test_quota_is_tracked_per_key_independently() -> None:
    mgr = _mgr()
    # Dùng hết RPD=2 cho keyA.
    mgr.acquire("gemini-x", 10, key_id="keyA")
    mgr.acquire("gemini-x", 10, key_id="keyA")
    with pytest.raises(QuotaExhaustedError):
        mgr.acquire("gemini-x", 10, key_id="keyA")
    # keyB vẫn còn nguyên quota (độc lập) — đây chính là bug đã sửa.
    assert mgr.acquire("gemini-x", 10, key_id="keyB") is not None
    assert mgr.get_remaining("gemini-x", key_id="keyB")["rpd_remaining"] == 1
    assert mgr.get_remaining("gemini-x", key_id="keyA")["rpd_remaining"] == 0


def test_mark_daily_exhausted_forces_next_acquire_to_raise() -> None:
    mgr = _mgr()
    mgr.mark_daily_exhausted("gemini-x", key_id="keyA")
    with pytest.raises(QuotaExhaustedError):
        mgr.acquire("gemini-x", 10, key_id="keyA")
    # Key khác không bị ảnh hưởng.
    assert mgr.get_remaining("gemini-x", key_id="keyB")["rpd_remaining"] == 2


def test_empty_key_id_backward_compatible() -> None:
    mgr = _mgr()
    # Không truyền key_id (hành vi cũ) vẫn chạy.
    assert mgr.acquire("gemini-x", 10) is not None
    assert mgr.get_remaining("gemini-x")["rpd_used"] == 1


# ── Adapter: parse + fingerprint + xoay key ──────────────────────────────


def _translator_cls():
    from subtitles_extractor.infrastructure.translation import (
        gemini_translation_adapter as _m,
    )
    return _m.GeminiSubtitleTranslator


def _adapter(api_key: str, quota_manager=None):
    return _translator_cls()(api_key=api_key, quota_manager=quota_manager)


def test_parse_multiple_keys_dedup_and_order() -> None:
    keys = _translator_cls()._parse_keys("k1\nk2, k1 ,\n k3 ")
    assert keys == ["k1", "k2", "k3"]


def test_fingerprint_stable_and_hidden() -> None:
    fp = _translator_cls()._fingerprint("super-secret-key")
    assert fp and "secret" not in fp and len(fp) == 12
    assert fp == _translator_cls()._fingerprint("super-secret-key")


def test_rotate_skips_exhausted_key() -> None:
    mgr = _mgr()
    adapter = _adapter("keyA\nkeyB", quota_manager=mgr)
    # Làm keyA cạn quota.
    fp_a = adapter._fingerprint("keyA")
    mgr.mark_daily_exhausted("gemini-x", key_id=fp_a)
    # Yêu cầu xoay → phải nhảy sang keyB (còn quota).
    assert adapter._rotate_to_available_key("gemini-x") is True
    assert adapter._api_key == "keyB"


def test_rotate_returns_false_when_all_exhausted() -> None:
    mgr = _mgr()
    adapter = _adapter("keyA\nkeyB", quota_manager=mgr)
    for k in ("keyA", "keyB"):
        mgr.mark_daily_exhausted("gemini-x", key_id=adapter._fingerprint(k))
    assert adapter._rotate_to_available_key("gemini-x") is False


def test_single_key_does_not_rotate() -> None:
    mgr = _mgr()
    adapter = _adapter("only-key", quota_manager=mgr)
    assert adapter._rotate_to_available_key("gemini-x") is False
