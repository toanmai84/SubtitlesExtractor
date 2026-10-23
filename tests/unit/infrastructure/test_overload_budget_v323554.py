"""[v3.23.554] Hồi quy: Gemini QUÁ TẢI (503) không được làm mất cả lượt dịch.

Nhật ký thật 2026-09-23 (app_log 38c6acc5): hai lần dịch kết thúc bằng ``Dịch phụ đề thất
bại: Gọi Gemini thất bại: 503 UNAVAILABLE`` sau đúng 5 lượt thử — tổng thời gian chờ chỉ
~83 giây (7.9 + 12.3 + 22.5 + 42.2), trong khi đợt quá tải của Google kéo dài lâu hơn thế.
Người dùng mất cả lượt dịch đã chạy hàng chục phút.

Lỗi quá tải KHÁC lỗi hết quota: xoay API key vô ích (server bận chứ không phải key
cạn), nên cách duy nhất là CHỜ. Nay có ngân sách chờ riêng cho nhóm lỗi này.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    OVERLOAD_MAX_DELAY_S,
    OVERLOAD_RETRY_BUDGET_S,
    GeminiSubtitleTranslator,
)


class TestOverloadClassification:
    def test_real_503_message_is_overload(self) -> None:
        message = (
            "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is "
            "currently experiencing high demand. Spikes in demand are usually "
            "temporary. Please try again later.', 'status': 'UNAVAILABLE'}}"
        )
        assert GeminiSubtitleTranslator._is_overload_error(Exception(message)) is True

    def test_http_code_attribute_is_enough(self) -> None:
        error = Exception("server busy")
        error.code = 503
        assert GeminiSubtitleTranslator._is_overload_error(error) is True

    def test_quota_error_is_not_overload(self) -> None:
        """429 hết quota phải đi nhánh XOAY KEY, không phải nhánh chờ dài."""
        quota = Exception("429 RESOURCE_EXHAUSTED: quota exceeded for model")
        assert GeminiSubtitleTranslator._is_overload_error(quota) is False

    def test_permanent_error_is_not_overload(self) -> None:
        error = ValueError("sai tham số")
        assert GeminiSubtitleTranslator._is_overload_error(error) is False

    def test_overload_is_still_retryable(self) -> None:
        error = Exception("503 UNAVAILABLE")
        assert GeminiSubtitleTranslator._is_retryable(error) is True


class TestBudget:
    def test_budget_outlives_the_real_outage(self) -> None:
        """Đợt quá tải thật dài hơn ~83s — ngân sách phải đủ lớn để vượt qua."""
        assert OVERLOAD_RETRY_BUDGET_S >= 300.0

    def test_delay_is_capped_so_we_do_not_hammer_a_busy_server(self) -> None:
        assert 30.0 <= OVERLOAD_MAX_DELAY_S <= 300.0

    def test_budget_allows_several_attempts(self) -> None:
        assert OVERLOAD_RETRY_BUDGET_S / OVERLOAD_MAX_DELAY_S >= 4


class TestRetryLoop:
    """Chạy THẬT vòng lặp thử lại của ``_call_gemini`` với đồng hồ giả."""

    @staticmethod
    def _translator(monkeypatch, failures: int):
        from types import SimpleNamespace

        translator = GeminiSubtitleTranslator(api_key="dummy", retry_count=5)
        attempts: list[float] = []

        class _Resp:
            text = '{"subtitles": []}'
            usage_metadata = None

        def fake_generate(model, contents, config):
            attempts.append(0.0)
            if len(attempts) <= failures:
                raise RuntimeError("503 UNAVAILABLE. This model has high demand.")
            return _Resp()

        monkeypatch.setattr(
            translator,
            "_get_client",
            lambda: SimpleNamespace(
                models=SimpleNamespace(generate_content=fake_generate)
            ),
        )
        slept: list[float] = []
        monkeypatch.setattr(
            translator, "_interruptible_sleep", lambda s, cb: slept.append(s) or False
        )
        return translator, attempts, slept

    def _config(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            temperature=0.2, response_mime_type="application/json",
            response_schema={}, system_instruction="x",
        )

    def test_survives_outage_longer_than_normal_retries(self, monkeypatch) -> None:
        """Ca thật: 503 kéo dài quá 5 lượt thử — trước đây mất cả lượt dịch."""
        translator, attempts, slept = self._translator(monkeypatch, failures=8)
        data = translator._call_gemini("flash", "prompt", self._config(), lambda d: None)
        assert data == {"subtitles": []}
        assert len(attempts) == 9
        assert sum(slept) <= OVERLOAD_RETRY_BUDGET_S + OVERLOAD_MAX_DELAY_S

    def test_gives_up_when_budget_is_spent(self, monkeypatch) -> None:
        import pytest

        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            SubtitleTranslationError,
        )

        translator, _attempts, slept = self._translator(monkeypatch, failures=10_000)
        with pytest.raises(SubtitleTranslationError):
            translator._call_gemini("flash", "prompt", self._config(), lambda d: None)
        assert sum(slept) >= OVERLOAD_RETRY_BUDGET_S

    def test_permanent_error_still_fails_fast(self, monkeypatch) -> None:
        from types import SimpleNamespace

        import pytest

        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            SubtitleTranslationError,
        )

        translator = GeminiSubtitleTranslator(api_key="dummy", retry_count=5)
        attempts: list[int] = []

        def fake_generate(model, contents, config):
            attempts.append(1)
            raise RuntimeError("400 INVALID_ARGUMENT: sai tham số")

        monkeypatch.setattr(
            translator,
            "_get_client",
            lambda: SimpleNamespace(
                models=SimpleNamespace(generate_content=fake_generate)
            ),
        )
        monkeypatch.setattr(translator, "_interruptible_sleep", lambda s, cb: False)
        with pytest.raises(SubtitleTranslationError):
            translator._call_gemini("flash", "prompt", self._config(), lambda d: None)
        assert len(attempts) == 1  # lỗi vĩnh viễn: KHÔNG thử lại
