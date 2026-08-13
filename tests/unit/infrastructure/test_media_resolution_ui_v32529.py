"""[v3.23.129] Test: (1) chống tái lỗi tên chưa định nghĩa ở translate_page;
(2) cấu hình độ phân giải video phân tích (analysis_media_resolution).
"""

from __future__ import annotations

import pytest

from subtitles_extractor.infrastructure.settings.application_settings import (
    TranslationSettings,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

# ── Chống tái lỗi NameError ở translate_page (caption_style, QMessageBox) ──


def test_translate_page_has_required_names() -> None:
    import subtitles_extractor.presentation.pages.translate_page as tp

    # Hai tên này từng gây NameError lúc mở hộp thoại (chưa import).
    assert hasattr(tp, "caption_style"), "thiếu import caption_style"
    assert hasattr(tp, "QMessageBox"), "thiếu import QMessageBox"


def test_no_undefined_names_in_presentation_pages() -> None:
    # Bắt mọi lỗi F821 (tên chưa định nghĩa) ở các trang presentation.
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "F821",
         "src/subtitles_extractor/presentation/pages"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Còn lỗi tên chưa định nghĩa:\n{result.stdout}"


# ── analysis_media_resolution ─────────────────────────────────────────────


def test_settings_accepts_valid_levels() -> None:
    for lvl in ("low", "medium", "high"):
        cfg = TranslationSettings(analysis_media_resolution=lvl)
        assert cfg.analysis_media_resolution == lvl


def test_settings_rejects_invalid_level() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TranslationSettings(analysis_media_resolution="ultra")


def test_settings_default_is_low() -> None:
    # [v3.23.141] Mặc định đổi medium -> low: an toàn TPM free-tier (low ~100 token/s,
    # medium ~300 token/s; một đoạn ~1063s ở medium = ~319K > 250K TPM -> 429).
    assert TranslationSettings().analysis_media_resolution == "low"


def test_adapter_stores_media_resolution() -> None:
    tr = GeminiSubtitleTranslator(api_key="x", analysis_media_resolution="high")
    assert tr._analysis_media_resolution == "high"


def test_adapter_defaults_and_sanitizes() -> None:
    assert GeminiSubtitleTranslator(api_key="x")._analysis_media_resolution == "medium"
    # Giá trị lạ → quy về medium (không vỡ).
    bad = GeminiSubtitleTranslator(api_key="x", analysis_media_resolution="bogus")
    assert bad._analysis_media_resolution == "medium"
