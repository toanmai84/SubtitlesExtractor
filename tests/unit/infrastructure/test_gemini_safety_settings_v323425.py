"""[v3.23.425] Test safety_settings BLOCK_NONE cho dịch phim (giảm chặn nhầm)."""

from __future__ import annotations

import types as pytypes

import pytest


def _perm(types_module):
    """Gọi hàm tĩnh mà không cần khởi tạo full adapter (tránh deps nặng)."""
    from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
        GeminiSubtitleTranslator,
    )

    return GeminiSubtitleTranslator._permissive_safety_settings(types_module)


class _FakeSafetySetting:
    def __init__(self, category, threshold):
        self.category = category
        self.threshold = threshold


def _fake_types() -> pytypes.SimpleNamespace:
    return pytypes.SimpleNamespace(
        HarmCategory=pytypes.SimpleNamespace(
            HARM_CATEGORY_HARASSMENT="harassment",
            HARM_CATEGORY_HATE_SPEECH="hate",
            HARM_CATEGORY_SEXUALLY_EXPLICIT="sexual",
            HARM_CATEGORY_DANGEROUS_CONTENT="dangerous",
        ),
        HarmBlockThreshold=pytypes.SimpleNamespace(BLOCK_NONE="BLOCK_NONE"),
        SafetySetting=_FakeSafetySetting,
    )


def test_four_categories_block_none() -> None:
    settings = _perm(_fake_types())
    assert len(settings) == 4
    assert all(s.threshold == "BLOCK_NONE" for s in settings)
    cats = {s.category for s in settings}
    assert cats == {"harassment", "hate", "sexual", "dangerous"}


def test_old_sdk_returns_empty() -> None:
    # SDK thiếu HarmCategory → rỗng, không ném lỗi.
    assert _perm(pytypes.SimpleNamespace()) == []


def test_missing_category_skipped() -> None:
    t = _fake_types()
    del t.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT  # thiếu 1 nhóm
    settings = _perm(t)
    assert len(settings) == 3  # bỏ qua nhóm thiếu, không crash


def test_safetysetting_ctor_error_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    t = _fake_types()

    def _raise(category, threshold):
        raise TypeError("ctor không nhận tham số này")

    t.SafetySetting = _raise
    assert _perm(t) == []  # lỗi ctor → bỏ qua an toàn
