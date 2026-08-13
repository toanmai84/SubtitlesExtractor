"""[v3.23.132] Test: chống 403 khi xoay API key (file Gemini cô lập theo key).

Khi adapter xoay sang key khác, handle video do key cũ tải lên bị 403 → phải nhận diện
lỗi này và TẢI LẠI đoạn bằng key hiện tại (callback), khớp theo khoảng thời gian.
"""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
    RemoteVideoRef,
)


def _adapter() -> GeminiSubtitleTranslator:
    return GeminiSubtitleTranslator(api_key="KEY1")


# ── _is_file_permission_error ─────────────────────────────────────────────


def test_detects_403_file_permission() -> None:
    err = Exception(
        "403 PERMISSION_DENIED. You do not have permission to access the File abc"
    )
    assert GeminiSubtitleTranslator._is_file_permission_error(err) is True


def test_ignores_other_errors() -> None:
    f = GeminiSubtitleTranslator._is_file_permission_error
    assert f(Exception("503 UNAVAILABLE high demand")) is False
    assert f(Exception("429 quota per day exceeded")) is False
    assert f(Exception("400 INVALID_ARGUMENT")) is False


# ── _resolve_handles_with_heal ────────────────────────────────────────────


def _ref(idx: int, s: float, e: float, name: str) -> RemoteVideoRef:
    return RemoteVideoRef(idx, name, s, e, "ACTIVE")


def test_heal_reuploads_when_handles_missing() -> None:
    adapter = _adapter()
    old_refs = [_ref(0, 0.0, 100.0, "files/OLD")]
    fresh_refs = [_ref(0, 0.0, 100.0, "files/NEW")]

    # Lần resolve đầu: file cũ → 403 → trả rỗng. Sau khi tải lại (ref mới) → resolve OK.
    def fake_resolve(refs):
        if refs and getattr(refs[0], "remote_name", "") == "files/NEW":
            return [SimpleNamespace(name="files/NEW")]
        return []  # file cũ không resolve được (403)

    adapter._resolve_video_handles = fake_resolve  # type: ignore[assignment]
    adapter._video_reupload_cb = lambda _key: fresh_refs

    handles = adapter._resolve_handles_with_heal(old_refs, [])
    assert len(handles) == 1
    # old_refs được cập nhật tại chỗ sang ref mới (khớp theo khoảng thời gian).
    assert old_refs[0].remote_name == "files/NEW"


def test_heal_noop_when_handles_ok() -> None:
    adapter = _adapter()
    refs = [_ref(0, 0.0, 100.0, "files/OK")]
    adapter._resolve_video_handles = lambda r: [SimpleNamespace(name="ok")]  # type: ignore[assignment]
    called = {"reupload": False}

    def _cb(_key):
        called["reupload"] = True
        return refs

    adapter._video_reupload_cb = _cb
    handles = adapter._resolve_handles_with_heal(refs, [])
    assert len(handles) == 1
    assert called["reupload"] is False  # đủ handle → KHÔNG tải lại


# ── provider.set_active_key ───────────────────────────────────────────────


def test_provider_set_active_key_switches(tmp_path) -> None:
    prov = GeminiVideoContextProvider(api_key="KEY_A", cache_db_path=tmp_path / "c.db")
    assert prov._api_key == "KEY_A"
    prov.set_active_key("KEY_B")
    assert prov._api_key == "KEY_B"
    # Đổi sang chuỗi nhiều key → lấy key đầu.
    prov.set_active_key("KEY_C\nKEY_D")
    assert prov._api_key == "KEY_C"
