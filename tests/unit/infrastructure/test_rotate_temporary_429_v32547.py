"""[v3.23.147] Test xoay key NGAY khi 429 tạm thời (text-only) + cooldown key-scoped.

Trước đây khi bị 429 TPM/RPM tạm thời, adapter ngồi chờ retryDelay 30-60s trên CÙNG key dù
key khác đang rảnh. Với batch text-only (không file gắn key), xoay key là miễn phí -> chuyển
ngay sang key vừa còn quota ngày vừa KHÔNG cooldown để tiếp tục tức thời.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    RateLimit,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

MODEL = "gemini-3.1-flash-lite"


def _quota() -> GeminiQuotaManager:
    return GeminiQuotaManager(
        rate_limits={"flash-lite": RateLimit(rpm=15, tpm=250_000, rpd=500)}
    )


def _adapter(keys: str, quota: GeminiQuotaManager | None) -> GeminiSubtitleTranslator:
    return GeminiSubtitleTranslator(api_key=keys, quota_manager=quota)


def _set_rpd_used(
    quota: GeminiQuotaManager, adapter: GeminiSubtitleTranslator, idx: int, used: int
) -> None:
    key_id = adapter._fingerprint(adapter._api_keys[idx])
    model_key = quota._compose_key(key_id, MODEL)
    with quota._lock:
        quota._daily_locked(model_key)["count"] = used


# ── cooldown_remaining_s (quota manager) ─────────────────────────────────


def test_cooldown_zero_initially() -> None:
    quota = _quota()
    assert quota.cooldown_remaining_s(MODEL, key_id="k1") == 0.0


def test_cooldown_set_and_key_scoped() -> None:
    quota = _quota()
    quota.note_rate_limited(MODEL, 30.0, key_id="k1")
    # Key bị 429 có cooldown ~30s; key KHÁC không bị ảnh hưởng (cô lập theo key).
    assert quota.cooldown_remaining_s(MODEL, key_id="k1") > 25.0
    assert quota.cooldown_remaining_s(MODEL, key_id="k2") == 0.0


# ── _rotate_for_temporary_429 (adapter) ──────────────────────────────────


def test_rotates_to_free_key_on_temporary_429() -> None:
    quota = _quota()
    adapter = _adapter("KEY_A\nKEY_B", quota)
    assert adapter._rotate_for_temporary_429(MODEL) is True
    assert adapter._api_key == "KEY_B"


def test_skips_key_in_cooldown() -> None:
    quota = _quota()
    adapter = _adapter("KEY_A\nKEY_B\nKEY_C", quota)
    # B đang cooldown -> phải chọn C.
    quota.note_rate_limited(MODEL, 60.0, key_id=adapter._fingerprint("KEY_B"))
    assert adapter._rotate_for_temporary_429(MODEL) is True
    assert adapter._api_key == "KEY_C"


def test_false_when_all_others_blocked() -> None:
    quota = _quota()
    adapter = _adapter("KEY_A\nKEY_B\nKEY_C", quota)
    # B cooldown, C cạn quota ngày -> không còn key sẵn sàng -> False, giữ A.
    quota.note_rate_limited(MODEL, 60.0, key_id=adapter._fingerprint("KEY_B"))
    _set_rpd_used(quota, adapter, 2, 500)
    assert adapter._rotate_for_temporary_429(MODEL) is False
    assert adapter._api_key == "KEY_A"


def test_false_with_single_key() -> None:
    quota = _quota()
    adapter = _adapter("ONLY_KEY", quota)
    assert adapter._rotate_for_temporary_429(MODEL) is False


# ── _is_rate_limit_error ─────────────────────────────────────────────────


def test_is_rate_limit_error_detects_429() -> None:
    err = RuntimeError("429 RESOURCE_EXHAUSTED: rate limit")
    assert GeminiSubtitleTranslator._is_rate_limit_error(err) is True


def test_is_rate_limit_error_ignores_503() -> None:
    err = RuntimeError("503 UNAVAILABLE: model overloaded")
    assert GeminiSubtitleTranslator._is_rate_limit_error(err) is False
