"""[v3.23.511] Chỉ gửi tham số lấy mẫu cho họ model CŨ còn hỗ trợ.

Tài liệu Gemini (7/2026): từ Gemini 3.6 Flash, ``temperature``/``top_p``/``top_k`` bị BỎ QUA và các
thế hệ model SAU sẽ trả HTTP 400 nếu vẫn gửi. Cách nhận diện cũ ("không phải gemini-3 thì gửi")
khiến model tương lai như ``gemini-4-flash`` vẫn bị gửi temperature → hỏng toàn bộ khâu dịch.
"""
from __future__ import annotations

import pytest

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


@pytest.mark.parametrize(
    "model_name",
    ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemma-3-27b-it"],
)
def test_legacy_models_receive_sampling_params(model_name: str) -> None:
    assert GeminiSubtitleTranslator._accepts_sampling_params(model_name) is True


@pytest.mark.parametrize(
    "model_name",
    ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"],
)
def test_gemini_3_family_excluded(model_name: str) -> None:
    assert GeminiSubtitleTranslator._accepts_sampling_params(model_name) is False


@pytest.mark.parametrize("model_name", ["gemini-4-flash", "gemini-5-pro", "model-la", ""])
def test_unknown_or_future_models_excluded(model_name: str) -> None:
    """Mặc định AN TOÀN: model lạ/mới không nhận tham số đã bị khai tử.

    Bỏ sót tham số vốn đã bị bỏ qua thì vô hại; gửi thừa thì hỏng cả tác vụ (HTTP 400).
    """
    assert GeminiSubtitleTranslator._accepts_sampling_params(model_name) is False


def test_build_config_omits_temperature_for_future_model() -> None:
    import inspect

    source = inspect.getsource(GeminiSubtitleTranslator._build_config)
    assert "self._accepts_sampling_params(model_name)" in source
    assert "if not self._is_gemini_3_family(model_name):" not in source


def test_top_p_and_top_k_never_sent() -> None:
    """Không bao giờ gửi top_p/top_k — đã khai tử với mọi model hiện hành."""
    import inspect

    source = inspect.getsource(GeminiSubtitleTranslator._build_config)
    assert "top_p" not in source
    assert "top_k" not in source
