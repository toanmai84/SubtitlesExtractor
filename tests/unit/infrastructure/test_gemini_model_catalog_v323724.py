"""[v3.23.724] Danh mục model Gemini sinh văn bản — miễn phí VÀ trả phí (bảng Rate Limit AI Studio 14/11/2026)."""

from __future__ import annotations

import pytest

from subtitles_extractor.infrastructure.translation import gemini_model_catalog as catalog
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    RateLimit,
    _default_limit,
    _match_free_tier_limit,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import GeminiSubtitleTranslator

#: Danh sách ``models.list`` thật của Gemini API (14/11/2026), rút gọn.
_API_NAMES = [
    "models/gemini-2.5-flash", "models/gemini-2.5-pro", "models/gemini-2.0-flash", "models/gemini-2.0-flash-001",
    "models/gemini-2.5-flash-lite", "models/gemini-flash-latest", "models/gemini-flash-lite-latest",
    "models/gemini-pro-latest", "models/gemini-3-flash-preview", "models/gemini-3.1-pro-preview",
    "models/gemini-3.1-flash-lite", "models/gemini-3.5-flash-lite", "models/gemini-3.5-flash",
    "models/gemini-3.6-flash", "models/gemini-3.7-flash", "models/gemini-3.8-flash", "models/gemini-4-argon",
    "models/gemini-4-carbon", "models/gemini-4-hydrogen", "models/gemini-2.5-flash-preview-tts",
    "models/gemini-3.1-flash-image-preview", "models/gemini-2.5-flash-native-audio-latest",
    "models/gemini-3.5-transcribe", "models/gemini-3.8-live", "models/gemini-embedding-001",
    "models/gemma-3-27b-it", "models/gemini-robotics-er-1.5-preview", "models/gemini-3-pro-image-preview",
    "models/gemini-2.5-computer-use-preview-10-2025", "models/gemini-3.6-omni",
]

_EXPECTED_TEXT = {
    "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite", "gemini-flash-latest", "gemini-flash-lite-latest",
    "gemini-pro-latest", "gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash", "gemini-4-argon", "gemini-4-carbon",
    "gemini-4-hydrogen",
}


def test_text_out_filter_keeps_free_and_paid_and_drops_others() -> None:
    kept = {name.removeprefix("models/") for name in _API_NAMES if catalog.is_text_out_model(name)}
    assert kept == _EXPECTED_TEXT


def test_old_filter_dropped_paid_and_hydrogen() -> None:
    """Tái hiện lỗi cũ: bộ lọc chỉ-flash-miễn-phí bỏ hẳn model trả phí và Gemini 4 Hydrogen (miễn phí)."""
    old = {n.removeprefix("models/") for n in _API_NAMES if GeminiSubtitleTranslator._is_free_tier_text_model(n)}
    assert "gemini-4-hydrogen" not in old and "gemini-2.5-pro" not in old
    new = {n.removeprefix("models/") for n in _API_NAMES if GeminiSubtitleTranslator._is_listed_text_model(n)}
    assert {"gemini-4-hydrogen", "gemini-4-argon", "gemini-4-carbon", "gemini-3.1-pro-preview",
            "gemini-2.5-pro"} <= new


def test_sort_free_first_then_paid_then_unknown() -> None:
    ordered = sorted(_EXPECTED_TEXT, key=catalog.sort_key)
    assert ordered[:len(catalog.TEXT_MODELS)] == catalog.default_model_ids()
    assert ordered[0] == "gemini-3.1-flash-lite"
    first_paid = next(i for i, m in enumerate(catalog.TEXT_MODELS) if m.paid_only)
    assert all(not m.paid_only for m in catalog.TEXT_MODELS[:first_paid])
    assert all(m.paid_only for m in catalog.TEXT_MODELS[first_paid:])
    tail = ordered[len(catalog.TEXT_MODELS):]
    assert tail.index("gemini-flash-latest") < tail.index("gemini-pro-latest")   # flash lạ trước


@pytest.mark.parametrize(("model", "limit"), [
    ("gemini-3.1-flash-lite", RateLimit(15, 250_000, 500)),
    ("models/gemini-3.5-flash-lite", RateLimit(15, 250_000, 500)),
    ("gemini-3.8-flash", RateLimit(5, 250_000, 20)),
    ("gemini-4-hydrogen", RateLimit(5, 250_000, 20)),
    ("gemini-2.5-flash-lite", RateLimit(10, 250_000, 20)),
    ("gemini-4-argon", catalog.PAID_TIER_LIMIT),
    ("gemini-2.5-pro", catalog.PAID_TIER_LIMIT),
])
def test_quota_defaults_follow_catalog(model: str, limit: RateLimit) -> None:
    assert _default_limit(model) == limit


def test_unknown_model_falls_back_to_name_heuristic() -> None:
    assert catalog.default_limit("gemini-9-flash") is None
    assert _default_limit("gemini-9-flash") == _match_free_tier_limit("gemini-9-flash")


def test_describe_marks_paid_only_and_free_limits() -> None:
    assert "CHỈ dùng với key trả phí" in catalog.describe("gemini-4-carbon")
    text = catalog.describe("gemini-3.1-flash-lite")
    assert "15 request/phút" in text and "500 request/ngày" in text
    assert "AI Studio" in catalog.describe("gemini-pro-latest")


def test_gemini4_uses_thinking_level() -> None:
    from subtitles_extractor.infrastructure.gemini_model_family import uses_thinking_level as checker

    for model in ("gemini-4-argon", "gemini-4-carbon", "gemini-4-hydrogen", "gemini-3.1-pro-preview"):
        assert checker(model)


def test_combo_tooltips(qapp=None) -> None:  # noqa: ANN001
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QComboBox

    from subtitles_extractor.presentation.pages.translate_widgets import apply_model_tooltips, replace_model_items

    _app = QApplication.instance() or QApplication([])
    combo = QComboBox()
    replace_model_items(combo, ["gemini-4-argon", "gemini-3.1-flash-lite"])
    tips = {combo.itemText(i): combo.itemData(i, Qt.ItemDataRole.ToolTipRole) for i in range(combo.count())}
    assert "trả phí" in tips["gemini-4-argon"]
    combo.addItem("gemini-x")
    apply_model_tooltips(combo)
    assert combo.itemData(combo.count() - 1, Qt.ItemDataRole.ToolTipRole)
