"""Test [v3.23.13] _build_config KHÔNG đặt media_resolution (gây 400 INVALID_ARGUMENT
trên v1beta — GitHub python-genai #793). Gemini 3 đã mặc định ~70 token/frame."""

from __future__ import annotations

from unittest.mock import MagicMock

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


class TestBuildConfigNoMediaResolution:
    def _adapter_with_fake_types(self) -> tuple[GeminiSubtitleTranslator, dict]:
        adapter = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
        captured: dict = {}

        def fake_config(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        fake_types = MagicMock()
        fake_types.GenerateContentConfig = fake_config
        adapter._types_module = fake_types
        return adapter, captured

    def test_no_media_resolution_key(self) -> None:
        adapter, captured = self._adapter_with_fake_types()
        adapter._build_config(
            temperature=0.3,
            response_schema={"type": "OBJECT"},
            system_instruction="dịch",
        )
        # MẤU CHỐT: không được có media_resolution (tránh 400 trên v1beta).
        assert "media_resolution" not in captured

    def test_keeps_core_keys(self) -> None:
        adapter, captured = self._adapter_with_fake_types()
        adapter._build_config(
            temperature=0.5,
            response_schema={"type": "OBJECT"},
            system_instruction="x",
        )
        assert captured["response_mime_type"] == "application/json"
        assert "response_schema" in captured
        assert captured["system_instruction"] == "x"

    def test_temperature_chi_gui_cho_ho_model_cu(self) -> None:
        """[v3.23.511] Chỉ họ model CŨ mới nhận ``temperature``.

        Tài liệu Gemini (7/2026): từ 3.6 Flash trở đi tham số lấy mẫu bị bỏ qua, và các
        thế hệ sau TRẢ HTTP 400 nếu vẫn gửi. Nên mặc định là KHÔNG gửi; chỉ gửi cho các
        họ đã biết chắc còn hỗ trợ.
        """
        adapter, captured = self._adapter_with_fake_types()
        adapter._build_config(
            temperature=0.5,
            response_schema={"type": "OBJECT"},
            system_instruction="x",
            model_name="gemini-2.5-flash",
        )
        assert captured["temperature"] == 0.5

    def test_temperature_khong_gui_cho_model_moi(self) -> None:
        """Model mới/lạ (kể cả khi không truyền tên) thì TUYỆT ĐỐI không gửi."""
        for model_name in ("", "gemini-3.5-flash", "gemini-4-flash"):
            adapter, captured = self._adapter_with_fake_types()
            adapter._build_config(
                temperature=0.5,
                response_schema={"type": "OBJECT"},
                system_instruction="x",
                model_name=model_name,
            )
            assert "temperature" not in captured, model_name
