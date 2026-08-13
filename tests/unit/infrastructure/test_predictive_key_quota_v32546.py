"""[v3.23.146] Test DỰ ĐOÁN key/quota: chọn key nhiều quota nhất + fail nhanh khi cạn.

Mục tiêu: app hành xử tối ưu, dự đoán tốt thay vì dò mù:
- ``has_any_daily_quota``: False khi MỌI key cạn -> worker fail nhanh (khỏi nén+upload).
- ``_best_available_key_index``: chọn key CÒN NHIỀU quota NHẤT (đi trọn phiên, ít xoay).
- ``ensure_viable_key``: đổi sang key tốt hơn HẲN khi key hiện tại yếu; giữ nguyên nếu sát nhau.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    RateLimit,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

MODEL = "gemini-3.5-flash"


def _quota() -> GeminiQuotaManager:
    return GeminiQuotaManager(
        rate_limits={"flash": RateLimit(rpm=5, tpm=250_000, rpd=20)}
    )


def _adapter(keys: str, quota: GeminiQuotaManager | None) -> GeminiSubtitleTranslator:
    return GeminiSubtitleTranslator(api_key=keys, quota_manager=quota)


def _set_rpd_used(
    quota: GeminiQuotaManager, adapter: GeminiSubtitleTranslator, idx: int, used: int
) -> None:
    """Set thẳng số request/ngày ĐÃ dùng cho key thứ ``idx`` (không gọi acquire -> không
    bị chặn bởi giới hạn RPM khi mô phỏng)."""
    key_id = adapter._fingerprint(adapter._api_keys[idx])
    model_key = quota._compose_key(key_id, MODEL)
    with quota._lock:
        quota._daily_locked(model_key)["count"] = used


def _exhaust(quota: GeminiQuotaManager, adapter: GeminiSubtitleTranslator, idx: int) -> None:
    _set_rpd_used(quota, adapter, idx, 20)  # rpd_limit = 20 -> cạn


def test_has_any_daily_quota_false_when_all_exhausted() -> None:
    quota = _quota()
    adapter = _adapter("KEY_A\nKEY_B\nKEY_C", quota)
    for i in range(3):
        _exhaust(quota, adapter, i)
    assert adapter.has_any_daily_quota(MODEL) is False


def test_has_any_daily_quota_true_when_one_left() -> None:
    quota = _quota()
    adapter = _adapter("KEY_A\nKEY_B\nKEY_C", quota)
    _exhaust(quota, adapter, 0)
    _exhaust(quota, adapter, 1)
    assert adapter.has_any_daily_quota(MODEL) is True


def test_has_any_daily_quota_true_without_manager() -> None:
    adapter = _adapter("KEY_A\nKEY_B", None)
    assert adapter.has_any_daily_quota(MODEL) is True


def test_best_index_picks_max_remaining() -> None:
    quota = _quota()  # rpd = 20 mỗi key
    adapter = _adapter("KEY_A\nKEY_B\nKEY_C", quota)
    # Dùng 15 ở A (còn 5), 5 ở B (còn 15), 0 ở C (còn 20) -> C nhiều nhất.
    _set_rpd_used(quota, adapter, 0, 15)
    _set_rpd_used(quota, adapter, 1, 5)
    assert adapter._best_available_key_index(MODEL) == 2


def test_best_index_none_when_all_exhausted() -> None:
    quota = _quota()
    adapter = _adapter("KEY_A\nKEY_B", quota)
    _exhaust(quota, adapter, 0)
    _exhaust(quota, adapter, 1)
    assert adapter._best_available_key_index(MODEL) is None


def test_ensure_viable_key_switches_to_much_better() -> None:
    quota = _quota()
    adapter = _adapter("KEY_A\nKEY_B", quota)
    # A còn 2, B còn 20 -> B tốt hơn hẳn -> đổi sang B.
    _set_rpd_used(quota, adapter, 0, 18)
    assert adapter._rpd_remaining_for_key(MODEL, adapter._api_keys[0]) == 2
    result = adapter.ensure_viable_key(MODEL)
    assert result == "KEY_B"
    assert adapter._api_key == "KEY_B"


def test_ensure_viable_key_keeps_current_when_close() -> None:
    quota = _quota()
    adapter = _adapter("KEY_A\nKEY_B", quota)
    # A còn 18, B còn 20 -> sát nhau -> KHÔNG churn, giữ A.
    _set_rpd_used(quota, adapter, 0, 2)
    before = adapter._current_key_id()
    adapter.ensure_viable_key(MODEL)
    assert adapter._current_key_id() == before
