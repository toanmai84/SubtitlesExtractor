"""[v3.23.157] Test chọn key ĐỦ quota đi trọn phiên (needed_requests) + ước lượng đoạn.

Log Three Against the World: phim 11 đoạn nhưng app giữ key #1 chỉ còn 10 request
(key #2 còn 12 — "sát nhau" nên logic cũ không xoay) -> cạn giữa đoạn 5 -> xoay key
-> 403 file -> nén+upload lại. Nay worker DỰ TRÙ needed_requests = số đoạn: key hiện
tại còn ÍT HƠN mức cần mà key khác dư dả hơn -> xoay NGAY từ đầu.
"""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    RateLimit,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
    VideoContextError,
)

MODEL = "gemini-3.5-flash"


def _quota() -> GeminiQuotaManager:
    return GeminiQuotaManager(
        rate_limits={MODEL: RateLimit(rpm=5, tpm=250_000, rpd=20)}
    )


def _set_rpd_used(
    quota: GeminiQuotaManager, adapter: GeminiSubtitleTranslator, idx: int, used: int
) -> None:
    key_id = adapter._fingerprint(adapter._api_keys[idx])
    model_key = quota._compose_key(key_id, MODEL)
    with quota._lock:
        quota._daily_locked(model_key)["count"] = used


def test_switches_when_current_insufficient_for_session() -> None:
    # Đúng ca trong log: #1 còn 10, #2 còn 12, phiên cần 11 -> phải chọn #2.
    quota = _quota()
    adapter = GeminiSubtitleTranslator(api_key="KEY_A\nKEY_B", quota_manager=quota)
    _set_rpd_used(quota, adapter, 0, 10)  # A còn 10
    _set_rpd_used(quota, adapter, 1, 8)   # B còn 12
    result = adapter.ensure_viable_key(MODEL, needed_requests=11)
    assert result == "KEY_B"


def test_keeps_current_when_sufficient() -> None:
    # A còn 15 >= 11 cần -> giữ A dù B còn 20 (không much_better, không insufficient).
    quota = _quota()
    adapter = GeminiSubtitleTranslator(api_key="KEY_A\nKEY_B", quota_manager=quota)
    _set_rpd_used(quota, adapter, 0, 5)  # A còn 15
    before = adapter._current_key_id()
    adapter.ensure_viable_key(MODEL, needed_requests=11)
    assert adapter._current_key_id() == before


def test_default_needed_is_backward_compatible() -> None:
    quota = _quota()
    adapter = GeminiSubtitleTranslator(api_key="KEY_A\nKEY_B", quota_manager=quota)
    _set_rpd_used(quota, adapter, 0, 2)  # A còn 18, B còn 20 — sát nhau -> giữ A
    before = adapter._current_key_id()
    adapter.ensure_viable_key(MODEL)
    assert adapter._current_key_id() == before


def test_estimate_chunk_count_from_plan() -> None:
    provider = GeminiVideoContextProvider.__new__(GeminiVideoContextProvider)
    provider.plan_chunks = lambda _p: SimpleNamespace(chunks=[object()] * 8)  # type: ignore[method-assign]
    assert provider.estimate_chunk_count("video.mp4") == 8


def test_estimate_chunk_count_fallback_one() -> None:
    provider = GeminiVideoContextProvider.__new__(GeminiVideoContextProvider)

    def _boom(_p):
        raise VideoContextError("không đọc được thời lượng")

    provider.plan_chunks = _boom  # type: ignore[method-assign]
    assert provider.estimate_chunk_count("video.mp4") == 1
