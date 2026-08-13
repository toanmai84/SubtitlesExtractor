"""[v3.23.250] Không set temperature thấp cho Gemini 3.x (theo tài liệu Google).

**Cải tiến chất lượng dịch có căn cứ tài liệu.** Gemini 3 Developer Guide khuyến
nghị MẠNH:
*"we strongly recommend keeping the temperature parameter at its default value of 1.0. If
your existing code explicitly sets temperature (especially to low values for deterministic
outputs), we recommend removing this parameter... to avoid potential looping issues or
performance degradation on complex tasks."*

Code cũ clamp temperature về [0,1] và truyền cho MỌI model — kể cả Gemini 3.x với giá trị
rất thấp (0.1-0.25 cho dịch tất định). Điều này đúng thứ tài liệu cảnh báo.

**Sửa:** với Gemini 3.x, KHÔNG truyền temperature (để model dùng mặc định 1.0, tránh
looping/giảm chất lượng). Với Gemini 2.5.x giữ nguyên hành vi cũ (model cũ được hiệu chỉnh
với temperature thấp cho dịch).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


def _adapter_with_fake_types() -> tuple[GeminiSubtitleTranslator, dict]:
    adapter = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    captured: dict = {}

    def fake_config(**kwargs: object) -> MagicMock:
        captured.clear()
        captured.update(kwargs)
        return MagicMock()

    fake_types = MagicMock()
    fake_types.GenerateContentConfig = fake_config
    adapter._types_module = fake_types
    return adapter, captured


def test_gemini_3_không_set_temperature() -> None:
    adapter, captured = _adapter_with_fake_types()
    adapter._build_config(
        temperature=0.25,
        response_schema={"type": "OBJECT"},
        system_instruction="dịch",
        model_name="gemini-3.1-flash-lite",
    )
    # Gemini 3.x: KHÔNG được truyền temperature (để mặc định 1.0).
    assert "temperature" not in captured


def test_gemini_25_vẫn_set_temperature() -> None:
    adapter, captured = _adapter_with_fake_types()
    adapter._build_config(
        temperature=0.25,
        response_schema={"type": "OBJECT"},
        system_instruction="dịch",
        model_name="gemini-2.5-flash-lite",
    )
    # Gemini 2.5.x: giữ hành vi cũ (set temperature clamp [0,1]).
    assert captured["temperature"] == 0.25


def test_model_rỗng_vẫn_set_temperature() -> None:
    # Không rõ model (rỗng) -> an toàn, giữ hành vi cũ.
    adapter, captured = _adapter_with_fake_types()
    adapter._build_config(
        temperature=0.5,
        response_schema={"type": "OBJECT"},
        system_instruction="x",
    )
    assert captured["temperature"] == 0.5


def test_core_keys_giữ_nguyên_cả_hai_họ() -> None:
    # response_mime_type + response_schema luôn có, bất kể họ model.
    for model in ("gemini-3.1-flash-lite", "gemini-2.5-flash-lite"):
        adapter, captured = _adapter_with_fake_types()
        adapter._build_config(
            temperature=0.3,
            response_schema={"type": "OBJECT"},
            system_instruction="x",
            model_name=model,
        )
        assert captured["response_mime_type"] == "application/json"
        assert "response_schema" in captured
