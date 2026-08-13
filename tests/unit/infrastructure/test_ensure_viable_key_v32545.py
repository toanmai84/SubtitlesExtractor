"""[v3.23.145] Test ensure_viable_key: CHỦ ĐỘNG chọn API key còn quota TRƯỚC khi upload.

Log Cosmos S51E08: video upload dưới key #1, nhưng phân tích mới phát hiện #1 hết quota,
xoay #2/#3 -> file thuộc key cũ -> 403 -> tải lại (nén lại) nhiều lần (dò mù). Nay worker
gọi ensure_viable_key TRƯỚC upload để chọn đúng key, set cùng key cho video provider.
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


def test_keeps_current_key_when_quota_left() -> None:
    quota = _quota()
    adapter = _adapter("KEY_A\nKEY_B", quota)
    before = adapter._current_key_id()
    result = adapter.ensure_viable_key(MODEL)
    assert result == adapter._api_key
    assert adapter._current_key_id() == before  # còn quota -> KHÔNG xoay


def test_rotates_when_current_key_exhausted() -> None:
    quota = _quota()
    adapter = _adapter("KEY_A\nKEY_B", quota)
    first_key_id = adapter._current_key_id()
    # Đánh dấu key hiện tại (#1) hết quota ngày -> ensure_viable_key phải xoay sang #2.
    quota.mark_daily_exhausted(MODEL, key_id=first_key_id)
    result = adapter.ensure_viable_key(MODEL)
    assert adapter._current_key_id() != first_key_id  # đã xoay
    assert result == adapter._api_key
    assert adapter._api_key == "KEY_B"


def test_single_key_returns_itself() -> None:
    quota = _quota()
    adapter = _adapter("ONLY_KEY", quota)
    quota.mark_daily_exhausted(MODEL, key_id=adapter._current_key_id())
    # Một key: không thể xoay -> trả chính nó (adapter tự xử lý hết-quota ở tầng gọi).
    assert adapter.ensure_viable_key(MODEL) == "ONLY_KEY"


def test_no_quota_manager_returns_current() -> None:
    adapter = _adapter("KEY_A\nKEY_B", None)
    assert adapter.ensure_viable_key(MODEL) == adapter._api_key
