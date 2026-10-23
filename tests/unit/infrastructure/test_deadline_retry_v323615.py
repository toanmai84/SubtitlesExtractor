"""[v3.23.615] Lời gọi Gemini hết thời hạn: thử lại (kể cả 499) và nới thời hạn lượt sau.

app_log thật 14/10 (phim "天骄女友退婚" tập 1, gemini-3.8-flash, thời hạn mỗi lời gọi 300
giây): lô dịch thô thứ hai chạy đúng 300 giây rồi nhận ``499 CANCELLED`` — mã này không
nằm trong nhóm thử lại nên cả lượt dịch thất bại ("Dịch phụ đề thất bại: Gọi Gemini thất
bại: 499 CANCELLED"), người dùng phải bấm dịch lại. Trên 44 tệp app_log, 56 lượt thử hỏng
đúng ở mốc 300 giây (504 DEADLINE_EXCEEDED, 499, "timed out"); lượt kế tiếp với CÙNG thời
hạn lại chạm mốc 17/62 lần.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
)
from subtitles_extractor.infrastructure.translation import (
    gemini_translation_adapter as adapter_module,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.infrastructure.translation.overload_retry_policy import (
    REQUEST_TIMEOUT_CEILING_S,
    deadline_was_hit,
    escalated_request_timeout,
)

#: Thông điệp nguyên văn trong app_log 14/10 14:26:45.
REAL_499 = (
    "499 CANCELLED. {'error': {'code': 499, 'message': 'The operation was cancelled.', "
    "'status': 'CANCELLED'}}"
)


class _ApiError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class _Response:
    text = '{"subtitles": []}'
    usage_metadata = None


class TestClassification:
    def test_real_499_is_retried(self) -> None:
        assert GeminiSubtitleTranslator._is_retryable(Exception(REAL_499)) is True

    def test_499_code_is_overload_like_504(self) -> None:
        error = _ApiError(499, REAL_499)
        assert GeminiSubtitleTranslator._is_retryable(error) is True
        assert GeminiSubtitleTranslator._is_overload_error(error) is True

    def test_timeout_message_does_not_blame_the_network_only(self) -> None:
        concise = GeminiSubtitleTranslator._concise_error(
            Exception("The read operation timed out")
        )
        assert "máy chủ" in concise


class TestPolicy:
    def test_deadline_hit_only_near_the_limit(self) -> None:
        assert deadline_was_hit(300.0, 300.0)
        assert deadline_was_hit(299.0, 300.0)
        assert not deadline_was_hit(12.0, 300.0)
        assert not deadline_was_hit(100.0, 0.0)

    def test_escalation_grows_then_stops_at_ceiling(self) -> None:
        assert escalated_request_timeout(300.0) == 450.0
        assert escalated_request_timeout(450.0) == REQUEST_TIMEOUT_CEILING_S
        assert escalated_request_timeout(REQUEST_TIMEOUT_CEILING_S) == (
            REQUEST_TIMEOUT_CEILING_S
        )

    def test_user_timeout_above_ceiling_is_never_shortened(self) -> None:
        assert escalated_request_timeout(900.0) == 900.0


def _translator(
    monkeypatch: pytest.MonkeyPatch, errors: list[Exception], durations: list[float]
) -> tuple[GeminiSubtitleTranslator, list[Any]]:
    """Adapter mà lượt gửi thứ i chạy ``durations[i]`` giây rồi ném ``errors[i]``."""
    translator = GeminiSubtitleTranslator(
        api_key="dummy", retry_count=5, request_timeout_s=300.0
    )
    configs: list[Any] = []
    clock = {"now": 1000.0}

    def fake_generate(model: str, contents: list[Any], config: Any) -> _Response:
        index = len(configs)
        configs.append(config)
        clock["now"] += durations[index] if index < len(durations) else 5.0
        if index < len(errors):
            raise errors[index]
        return _Response()

    monkeypatch.setattr(
        translator,
        "_get_client",
        lambda: SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate)),
    )
    monkeypatch.setattr(translator, "_interruptible_sleep", lambda s, cb: False)
    monkeypatch.setattr(
        adapter_module, "time", SimpleNamespace(monotonic=lambda: clock["now"])
    )
    return translator, configs


def _real_config(translator: GeminiSubtitleTranslator) -> Any:
    types = translator._types_module
    if types is None:
        pytest.skip("google-genai chưa cài")
    return types.GenerateContentConfig(temperature=0.2, system_instruction="x")


def _timeout_ms(config: Any) -> int | None:
    http_options = getattr(config, "http_options", None)
    return None if http_options is None else http_options.timeout


class TestRetryLoop:
    def test_499_at_deadline_no_longer_fails_the_whole_job(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        translator, configs = _translator(
            monkeypatch, [_ApiError(499, REAL_499)], [300.0]
        )
        data = translator._call_gemini(
            "gemini-3.8-flash", "prompt", _real_config(translator), lambda d: None
        )
        assert data == {"subtitles": []}
        assert len(configs) == 2

    def test_next_attempt_gets_a_longer_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        errors = [
            _ApiError(504, "504 DEADLINE_EXCEEDED"),
            Exception("The read operation timed out"),
        ]
        translator, configs = _translator(monkeypatch, errors, [300.0, 450.0])
        translator._call_gemini(
            "gemini-3.8-flash", "prompt", _real_config(translator), lambda d: None
        )
        assert [_timeout_ms(config) for config in configs] == [None, 450_000, 600_000]

    def test_quick_failure_keeps_the_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        translator, configs = _translator(
            monkeypatch, [_ApiError(503, "503 UNAVAILABLE")], [3.0]
        )
        translator._call_gemini(
            "gemini-3.8-flash", "prompt", _real_config(translator), lambda d: None
        )
        assert [_timeout_ms(config) for config in configs] == [None, None]

    def test_permanent_error_still_fails_fast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        translator, configs = _translator(
            monkeypatch, [_ApiError(400, "400 INVALID_ARGUMENT")], [300.0]
        )
        with pytest.raises(SubtitleTranslationError):
            translator._call_gemini(
                "gemini-3.8-flash", "prompt", _real_config(translator), lambda d: None
            )
        assert len(configs) == 1
