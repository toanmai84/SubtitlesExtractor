"""[v3.23.614] Chọn key TRƯỚC khi tải video không được coi video vừa tải là "của key cũ".

app_log thật 14/10 (phân tích lẫn dịch): "Chọn API key #3 … thay vì #1 (còn 0)" → tải
video bằng key #3 → lời gọi Gemini đầu tiên ghi "Vừa đổi API key → handle video cũ chắc
chắn vô hiệu, tải lại NGAY" rồi "tải lại" đúng file vừa tải. Cờ "handle cũ" được bật khi
đổi key, nhưng đổi key ở bước chọn trước khi tải thì chưa có handle cũ nào. Với phim nhiều
đoạn, lượt "tải lại" vô ích đó còn nén lại đoạn video bằng ffmpeg.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    RateLimit,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    RemoteVideoRef,
)

MODEL = "gemini-3.8-flash"


def _adapter_with_reupload_log() -> tuple[GeminiSubtitleTranslator, list[Any]]:
    quota = GeminiQuotaManager(
        rate_limits={"flash": RateLimit(rpm=5, tpm=250_000, rpd=20)}
    )
    adapter = GeminiSubtitleTranslator(api_key="KEY_A\nKEY_B", quota_manager=quota)
    quota.mark_daily_exhausted(MODEL, key_id=adapter._current_key_id())
    reuploads: list[Any] = []

    def _resolve(refs: list[Any] | None) -> list[Any]:
        return [SimpleNamespace(name=ref.remote_name) for ref in refs or []]

    def _reupload(_key: str, ranges: list[tuple[float, float]]) -> list[Any]:
        reuploads.append(ranges)
        return [RemoteVideoRef(0, "files/AGAIN", 0.0, 177.0, "ACTIVE")]

    adapter._resolve_video_handles = _resolve  # type: ignore[method-assign]
    adapter._video_reupload_cb = _reupload
    return adapter, reuploads


def test_video_uploaded_after_key_choice_is_used_directly() -> None:
    adapter, reuploads = _adapter_with_reupload_log()
    assert adapter.ensure_viable_key(MODEL, needed_requests=3) == "KEY_B"
    refs = [RemoteVideoRef(0, "files/6iip7kcgg24f", 0.0, 177.0, "ACTIVE")]

    handles = adapter._resolve_handles_with_heal(refs, [])

    assert [handle.name for handle in handles] == ["files/6iip7kcgg24f"]
    assert reuploads == []


def test_rotation_in_the_middle_still_reuploads() -> None:
    adapter, reuploads = _adapter_with_reupload_log()
    assert adapter._rotate_to_available_key(MODEL)
    refs = [RemoteVideoRef(0, "files/OLD", 0.0, 177.0, "ACTIVE")]

    adapter._resolve_handles_with_heal(refs, [])

    assert reuploads == [[(0.0, 177.0)]]
