"""[v3.23.610] Tắt "Thinking" phải giảm suy nghĩ thật sự (học từ gemini-srt-translator).

Trước bản này, bỏ chọn ô Thinking thì ứng dụng KHÔNG gửi cấu hình suy nghĩ nào, nên
model dùng mức MẶC ĐỊNH của nó. Theo tài liệu Gemini API (10/2026), mặc định của
gemini-3.8-flash và 3.6-flash là Medium, của 3.1-flash-lite là High — tức là "tắt" lại
suy nghĩ NHIỀU HƠN mức "Thấp" mà người dùng chọn khi bật. Gemini 2.5 Flash mặc định
cũng suy nghĩ (động), chỉ tắt được bằng ``thinking_budget=0``.

gemini-srt-translator xử lý đúng: Gemini 3 không tắt hẳn được → gửi mức THẤP NHẤT model
nhận (``minimal``; bản Pro chỉ từ ``low``); Gemini 2.5 Pro không tắt được → ngân sách tối
thiểu 128.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import types as genai_types

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


def _translator() -> GeminiSubtitleTranslator:
    translator = GeminiSubtitleTranslator(api_key="KEY")
    translator._types_module = genai_types
    return translator


def _thinking(config: Any) -> dict[str, Any]:
    thinking = getattr(config, "thinking_config", None)
    if thinking is None:
        return {}
    result: dict[str, Any] = {}
    if thinking.thinking_level is not None:
        result["level"] = str(thinking.thinking_level.value).lower()
    if thinking.thinking_budget is not None:
        result["budget"] = thinking.thinking_budget
    return result


def _config(model: str, *, enabled: bool, budget: int = -1, level: str = "low") -> Any:
    return _translator()._build_config(
        0.3, {}, "sys", enable_thinking=enabled, thinking_budget=budget,
        model_name=model, thinking_level=level,
    )


class TestThinkingOff:
    @pytest.mark.parametrize(
        "model", ["gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.5-flash-lite"]
    )
    def test_gemini_3_flash_gets_minimal(self, model: str) -> None:
        assert _thinking(_config(model, enabled=False)) == {"level": "minimal"}

    @pytest.mark.parametrize(
        "model", ["gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-3-pro-preview"]
    )
    def test_models_without_minimal_get_low(self, model: str) -> None:
        assert _thinking(_config(model, enabled=False)) == {"level": "low"}

    def test_budget_zero_counts_as_off(self) -> None:
        cfg = _config("gemini-3.1-flash-lite", enabled=True, budget=0, level="high")
        assert _thinking(cfg) == {"level": "minimal"}

    def test_gemini_25_flash_is_really_turned_off(self) -> None:
        assert _thinking(_config("gemini-2.5-flash", enabled=False)) == {"budget": 0}

    def test_gemini_25_pro_gets_the_minimum_budget(self) -> None:
        assert _thinking(_config("gemini-2.5-pro", enabled=False)) == {"budget": 128}

    def test_model_without_thinking_gets_no_config(self) -> None:
        assert _thinking(_config("gemini-2.0-flash", enabled=False)) == {}
        assert _thinking(_config("gemini-2.0-flash", enabled=True, budget=512)) == {}


class TestThinkingOnLimits:
    def test_gemini_25_pro_budget_below_minimum_is_raised(self) -> None:
        cfg = _config("gemini-2.5-pro", enabled=True, budget=64)
        assert _thinking(cfg) == {"budget": 128}

    def test_gemini_25_budgets_are_capped(self) -> None:
        assert _thinking(_config("gemini-2.5-pro", enabled=True, budget=40000)) == {
            "budget": 32768
        }
        assert _thinking(_config("gemini-2.5-flash", enabled=True, budget=30000)) == {
            "budget": 24576
        }

    def test_pro_cannot_use_minimal(self) -> None:
        cfg = _config("gemini-3.1-pro-preview", enabled=True, level="minimal")
        assert _thinking(cfg) == {"level": "low"}

    def test_chosen_level_is_kept(self) -> None:
        cfg = _config("gemini-3.8-flash", enabled=True, level="high")
        assert _thinking(cfg) == {"level": "high"}


class _Response:
    text = '{"subtitles": []}'
    usage_metadata = None


def _fake_client(calls: list[Any], reject_minimal: bool) -> Any:
    def generate(*, model: str, contents: Any, config: Any) -> _Response:
        calls.append(config)
        level = _thinking(config).get("level")
        if reject_minimal and level == "minimal":
            raise RuntimeError(
                "400 INVALID_ARGUMENT. {'error': {'message': 'Thinking level MINIMAL is "
                "not supported for this model.', 'status': 'INVALID_ARGUMENT'}}"
            )
        return _Response()

    return SimpleNamespace(models=SimpleNamespace(generate_content=generate))


def test_model_rejecting_minimal_falls_back_to_low_and_remembers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translator = _translator()
    calls: list[Any] = []
    monkeypatch.setattr(translator, "_get_client", lambda: _fake_client(calls, True))
    config = translator._build_config(
        0.3, {}, "sys", enable_thinking=False, model_name="gemini-3.9-flash",
        media_resolution="low",
    )
    assert translator._call_gemini("gemini-3.9-flash", "p", config, lambda _d: None) == {
        "subtitles": []
    }
    assert [_thinking(c) for c in calls] == [{"level": "minimal"}, {"level": "low"}]
    # Bước hạ mức chỉ đổi phần suy nghĩ: độ phân giải video và bộ lọc an toàn còn nguyên.
    assert calls[1].media_resolution == calls[0].media_resolution
    assert calls[1].safety_settings == calls[0].safety_settings
    # Lô sau của cùng model gửi thẳng "low" — không tốn thêm lượt bị từ chối.
    later = translator._build_config(
        0.3, {}, "sys", enable_thinking=False, model_name="gemini-3.9-flash"
    )
    assert _thinking(later) == {"level": "low"}


def test_stripping_thinking_keeps_video_resolution_and_safety(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model không nhận cấu hình suy nghĩ: bỏ phần đó, giữ nguyên phần còn lại."""
    translator = _translator()
    calls: list[Any] = []

    def generate(*, model: str, contents: Any, config: Any) -> _Response:
        calls.append(config)
        if len(calls) == 1:
            raise RuntimeError("400 INVALID_ARGUMENT: thinking_config is not supported")
        return _Response()

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    monkeypatch.setattr(translator, "_get_client", lambda: client)
    config = translator._build_config(
        0.3, {}, "sys", enable_thinking=True, model_name="gemini-3.9-flash",
        thinking_level="high", media_resolution="low",
    )
    translator._call_gemini("gemini-3.9-flash", "p", config, lambda _d: None)
    assert calls[1].thinking_config is None
    assert calls[1].media_resolution == calls[0].media_resolution
    assert calls[1].safety_settings
    assert calls[1].safety_settings == calls[0].safety_settings
